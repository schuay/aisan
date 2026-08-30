# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Say what a box is, in the terms bwrap will actually apply.

The audit artifact. A confinement profile is only as good as somebody's ability
to check it against intent, and a bind list in Python source is not that: the
question is always "what does the box END UP with", which is a function of mount
ORDER, of hoisting, of which optional sources happen to exist on this host, and
of two deferred remounts that sit far from the thing they seal.

So this renders the resolved list and then the exact argv, and it renders them
from the same `BoxSpec` the real path uses -- never from a second construction
that agrees with the first only until someone edits one of them. That fidelity
is the whole reason the mode catches drift, and it is why `Box.staged()` exists:
describing a profile must put the same files on disk the real path does, without
minting a credential or opening a socket.

Two readings, deliberately both:

- the classified mount list, which is how a human checks the policy (what is
  writable, what is pinned ro on top of it, what is sealed);
- the argv, which is what actually happens, and the thing a snapshot test can
  hold still.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .box import Box
from .sandbox import Sandbox


@dataclass(frozen=True)
class Mount:
    """One mount spec read back out of the wrapper argv. `path` is the mount
    destination; for binds src==dst at the real absolute path."""

    kind: str  # system | proc | dev | symlink | ro | ro-pin | ro-shadow | ro-sub
    #           | rw-root | rw | tmpfs | overlay | seal-ro
    path: str
    idx: int  # token index of the mount spec (for ordering)
    size: str | None = None  # tmpfs only


# Kind -> ANSI, grouped by what a reviewer scans FOR, never decoration:
#   red/bold  the box can write here, and the one it must not miss -- ro-shadow,
#             an ro bind a later rw defeats, reads as protected but is writable;
#   yellow    writable by design (the rw root, binds, tmpfs, overlay);
#   green     a guard doing its job (a pin, a seal);
#   cyan      a substitution -- the box reads a DIFFERENT file than the host's;
#   dim       inert surface (plain ro, the fixed system mounts).
_KIND_ANSI = {
    "ro-shadow": "1;31",
    "rw-root": "33",
    "rw": "33",
    "tmpfs": "33",
    "overlay": "33",
    "ro-pin": "32",
    "seal-ro": "32",
    "ro-sub": "36",
    "ro": "2",
    "system": "2",
    "proc": "2",
    "dev": "2",
    "symlink": "2",
}

# The fixed system surface, collapsed to one disclosed line: identical in every
# box, so per-line it is noise between a reviewer and the policy. Named, not
# hidden -- a change to it still shows as a changed path in the one line.
_SURFACE_KINDS = frozenset({"system", "proc", "dev", "symlink"})


def _kind_field(kind: str, color: bool, width: int = 9) -> str:
    """The kind column, padded to `width`, wrapped in its colour when `color`.

    Padded before colouring so the ANSI bytes never enter the column math and
    the paths stay aligned; plain and identical to the old output when off, so
    a pipe or a snapshot sees no escape codes."""
    field = f"{kind:<{width}}"
    ansi = _KIND_ANSI.get(kind)
    return f"\x1b[{ansi}m{field}\x1b[0m" if color and ansi else field


@dataclass(frozen=True)
class Profile:
    mounts: list[Mount]
    tmpfs: list[Mount]
    env: dict[str, str]
    chdir: str


def _policy_dests(sb: Sandbox) -> set[Path]:
    """Mount destinations the profile's own bind list asks for.

    Everything else in the argv came from the fixed system surface (/usr, /etc,
    the journal socket), which is emitted outside the list. Distinguishing them
    by membership rather than by argv position means the inspector keeps naming
    them correctly through any reordering."""
    dests: set[Path] = set()
    for m in sb.resolve():
        try:
            dests.add(Path(m.dst).resolve())
        except OSError:
            continue
    return dests


def _pins_and_shadows(sb: Sandbox) -> tuple[set[Path], set[Path]]:
    """The two ways an ro bind is not what "ro" alone says, read off the ORDER.

    Both are security-relevant and both are invisible in a flat "ro" label, so a
    reviewer scanning a dump has to see them named:

    - a PIN is an ro bind written after an rw one that covers it -- the .git
      config guards. The box has it read-only, protecting a writable region.
    - a SHADOW is the reverse: an ro bind that a LATER writable mount (rw, tmpfs
      or overlay) covers, so the box actually has it WRITABLE. Labelling that
      "ro" is the misleading rendering the audit flagged -- the reviewer reads
      protection where there is none.

    Mutually exclusive for one destination (the covering rw is either before it
    or after it), and both come from the resolved order rather than a field, so
    neither can disagree with the box the argv builds.
    """
    mounts = sb.resolve()
    resolved = [(m, Path(m.dst).resolve()) for m in mounts]
    pinned: set[Path] = set()
    shadowed: set[Path] = set()
    writable: list[Path] = []
    for i, (m, dst) in enumerate(resolved):
        if m.op == "rw":
            writable.append(dst)
        elif m.op == "ro":
            if any(dst.is_relative_to(w) for w in writable):
                pinned.add(dst)
            elif any(
                later.covers
                and later.op in ("rw", "tmpfs", "overlay")
                and dst.is_relative_to(ld)
                for later, ld in resolved[i + 1 :]
            ):
                shadowed.add(dst)
    return pinned, shadowed


def parse_wrapper(argv: list[str], sb: Sandbox) -> Profile:
    """Tokenize the bwrap argv into classified mounts + env. Recognizes only
    the spec forms `Sandbox` emits (--size/--tmpfs, --ro-bind/--bind S D,
    --overlay-src S --tmp-overlay D, --remount-ro D, --setenv K V, --chdir P);
    anything else is skipped.

    The argv is what bwrap is actually handed, so ordering here is the real
    precedence; the resolved list is consulted only to name the ro binds that
    are pins."""
    mounts: list[Mount] = []
    env: dict[str, str] = {}
    chdir = ""
    pinned, shadowed = _pins_and_shadows(sb)
    policy = _policy_dests(sb)
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--tmpfs" and i + 1 < n:
            # Emitted as `--size N --tmpfs MNT`, so the cap sits two tokens
            # back and its value one; not all tmpfs carry one (a seal's does
            # not), so look it up defensively.
            size = argv[i - 1] if i >= 2 and argv[i - 2] == "--size" else None
            mounts.append(Mount("tmpfs", argv[i + 1], i, size))
            i += 2
        elif a == "--tmp-overlay" and i + 1 < n:
            mounts.append(Mount("overlay", argv[i + 1], i))
            i += 2
        elif a == "--remount-ro" and i + 1 < n:
            # The second half of a seal: the tmpfs above went down empty, this
            # closes it once the holes through it have been mounted.
            mounts.append(Mount("seal-ro", argv[i + 1], i))
            i += 2
        elif a == "--ro-bind" and i + 2 < n:
            src, dst = argv[i + 1], Path(argv[i + 2])
            if src != argv[i + 2]:
                # A BindOver: src is substituted at dst, so this is NOT "the
                # host's dst is mounted" -- the box reads a different file there
                # (ReapiBackend's /etc/hosts redirect, a per-box .sisoenv).
                kind = "ro-sub"
            else:
                try:
                    d = dst.resolve()
                    kind = (
                        "ro-pin"
                        if d in pinned
                        else "ro-shadow"
                        if d in shadowed
                        else "ro"
                        if d in policy
                        else "system"
                    )
                except OSError:
                    kind = "ro"
            mounts.append(Mount(kind, argv[i + 2], i))
            i += 3
        elif a == "--proc" and i + 1 < n:
            mounts.append(Mount("proc", argv[i + 1], i))
            i += 2
        elif a == "--dev" and i + 1 < n:
            mounts.append(Mount("dev", argv[i + 1], i))
            i += 2
        elif a == "--symlink" and i + 2 < n:
            # `--symlink TARGET LINKPATH`: the merged-usr links (/bin -> usr/bin)
            # that stand in for a bound /. Named by the link path, since that is
            # what exists in the box.
            mounts.append(Mount("symlink", argv[i + 2], i))
            i += 3
        elif a == "--bind" and i + 2 < n:
            dst = argv[i + 2]
            d = Path(dst).resolve()
            kind = "rw-root" if d == Path(str(sb.root)).resolve() else "rw"
            mounts.append(Mount(kind, dst, i))
            i += 3
        elif a == "--setenv" and i + 2 < n:
            env[argv[i + 1]] = argv[i + 2]
            i += 3
        elif a == "--chdir" and i + 1 < n:
            chdir = argv[i + 1]
            i += 2
        else:
            i += 1
    tmpfs = [m for m in mounts if m.kind == "tmpfs"]
    return Profile(mounts=mounts, tmpfs=tmpfs, env=env, chdir=chdir)


def _anc_eq(child: str, ancestor: str) -> bool:
    """Is `ancestor` equal to or a parent of `child` (both resolved)? The
    shadowing direction we care about: a later bind whose path covers the tmpfs
    mount point. is_relative_to is True for equality too."""
    try:
        return Path(child).resolve().is_relative_to(Path(ancestor).resolve())
    except (OSError, ValueError):
        return False


def assembly_refusal(box: Box) -> Exception | None:
    """The refusal box assembly raises for this spec, or None if it resolves.

    `_sandbox()` refuses a spec whose binds would shadow a tmpfs or the rw root
    or expose a backend credential; `wrapper()` refuses a bind whose mandatory
    source is missing. Both are the finding `--explain` exists to surface: this
    is where `explain` reads the refusal to render it, and where the CLI reads
    it to exit nonzero, so a scripted `--explain && run` does not treat a
    refused profile as a passed review. Both builders are pure, so probing them
    here mints nothing and opens no socket."""
    try:
        box._sandbox()
        box.wrapper()
    except (ValueError, FileNotFoundError) as exc:
        return exc
    return None


def explain(
    box: Box,
    *,
    inputs: tuple[tuple[str, str], ...] = (),
    argv: bool = True,
    color: bool = False,
) -> str:
    """The whole report for `box`, as text.

    Text rather than prints, because the two consumers want different things
    with it: an operator wants it on a terminal, and the snapshot test wants to
    compare it. A function that printed would force the test to capture stdout,
    which is the kind of indirection that makes an audit artifact feel optional.

    `inputs` is the caller's own context (which config, which worktree) -- the
    one section aisan cannot fill in, because the whole point of a preset is
    that aisan does not know what its arguments meant.
    """
    out = io.StringIO()

    def section(title: str) -> None:
        out.write(f"\n== {title} ==\n")

    # A leak or a missing mandatory source surfaces as a raise from the
    # builders, not as an argv to inspect. Report it as the finding -- this IS
    # the diagnosis the mode exists to give, and a traceback would bury it. The
    # CLI reads the same predicate to exit nonzero.
    refusal = assembly_refusal(box)
    if refusal is not None:
        section("BOX ASSEMBLY REFUSED")
        out.write(f"  {refusal}\n")
        return out.getvalue()
    sandbox = box._sandbox()
    wrapper = box.wrapper()
    prof = parse_wrapper(wrapper, sandbox)
    isolated = box.spec.unshare_net
    shared_egress = bool(box.spec.egress) and not isolated
    effective_env = dict(prof.env)
    if shared_egress:
        for backend in box.spec.egress:
            effective_env.update(backend.shared_client_env_description())
    home = effective_env.get("HOME", "")

    section("inputs")
    for key, value in inputs:
        out.write(f"  {key:<9} {value}\n")
    # Rendered from unshare_net directly, unconditionally: the network mode is
    # the box's most consequential boundary, and gating this line on egress once
    # let a shared-host-network box with no backends print nothing here while the
    # egress section claimed "no route off the machine".
    if isolated:
        out.write("  network   own namespace (no route off the machine)\n")
    else:
        out.write("  network   shared host namespace (full host network)\n")
    out.write(f"  box_id    {box.box_id}\n")
    out.write(f"  HOME(env) {home}\n")
    out.write(f"  chdir     {prof.chdir}\n")

    if not shared_egress:
        section("egress backends (host half on a socket, in-box on loopback)")
    else:
        section("egress backends (authenticated host-loopback TCP)")
    if box.spec.egress:
        for backend in box.spec.egress:
            if not shared_egress:
                sock = backend.socket_path(box.runtime_dir)
                out.write(f"  {backend.name:<8} 127.0.0.1:{backend.port} -> {sock}\n")
            else:
                out.write(f"  {backend.name:<8} 127.0.0.1:(assigned at launch)\n")
    elif isolated:
        out.write("  (none; the box has no route off the machine)\n")
    else:
        out.write("  (none; egress off -- the box shares the host network)\n")

    section("tmpfs mounts (mounted before binds; intended writable scratch)")
    if prof.tmpfs:
        for m in prof.tmpfs:
            tag = "  <- $HOME" if home and _anc_eq(home, m.path) else ""
            out.write(f"  [{m.idx:>3}] {m.path}  ({m.size or 'no size'}{tag})\n")
    else:
        out.write("  (none)\n")

    section("binds in argv order (later shadows earlier on overlap)")
    ordered = [m for m in sorted(prof.mounts, key=lambda x: x.idx) if m.kind != "tmpfs"]
    surface = [m for m in ordered if m.kind in _SURFACE_KINDS]
    if surface:
        paths = " ".join(m.path for m in surface)
        out.write(f"  {_kind_field('system', color)} {paths}\n")
    for m in ordered:
        if m.kind in _SURFACE_KINDS:
            continue
        out.write(f"  [{m.idx:>3}] {_kind_field(m.kind, color)} {m.path}\n")

    # A seal is two ops far apart in the argv (an empty tmpfs where the directory
    # was, then a ro remount after the holes through it are mounted), so neither
    # line alone says "sealed". Name them together or a reader has to reconstruct
    # the pairing from the ordering.
    sealed = [m for m in prof.mounts if m.kind == "seal-ro"]
    if sealed:
        section("sealed directories (empty in the box; nothing creatable)")
        for m in sealed:
            out.write(f"  {m.path}\n")

    if not shared_egress:
        section("environment (the box's complete environment; --clearenv first)")
    else:
        section("effective environment (backend values injected by launcher)")
    for k in sorted(effective_env):
        out.write(f"  {k}={effective_env[k]}\n")

    if argv:
        # The exact argv, one token per line. Not shell-quoted: this is a list
        # bwrap is handed directly, and rendering it as a copy-pasteable command
        # would invite someone to run a re-parsed approximation of it.
        section("argv (exactly what is exec'd, one token per line)")
        for token in [*wrapper, *box.launch_prefix()]:
            out.write(f"  {token}\n")
        out.write("  <payload>\n")
    return out.getvalue()


_AISAN_MARK = "  <AISAN RUNTIME>"


def _collapse_aisan_runtime(text: str) -> str:
    """Replace the run of lines describing aisan's own runtime with one marker.

    Aisan binds its interpreter, its venv and its editable source roots into
    every box, unconditionally (see `Box._sandbox`). How MANY paths that is, and
    what they are called, is a property of how aisan happens to be installed on
    this machine -- one site-packages tree for a released install, one line per
    workspace member for a checkout like this one. It is identical in every box
    by construction, so it can never be the thing a preset diff changed, and a
    snapshot carrying it would fail on somebody's install layout while staying
    green through an actual policy change.

    Matched by asking `launcher_binds` for the paths rather than by pattern, so
    this collapses exactly what the Box added and nothing that looks like it.
    """
    from .launch import launcher_binds

    paths = {str(b.path) for b in launcher_binds()}
    lines = text.split("\n")
    drop = [any(ln.endswith(p) for p in paths) for ln in lines]
    for i, dropped in enumerate(drop):
        # In the argv section a bind is three lines (--ro-bind, src, dst), so the
        # flag introducing a dropped path goes with it or a bare --ro-bind is
        # left behind pointing at nothing.
        if dropped and i and lines[i - 1].strip() in ("--ro-bind", "--bind"):
            drop[i - 1] = True
    out: list[str] = []
    # strict: drop is built per line above, so a length mismatch would mean
    # the mask no longer describes the text it is masking.
    for line, dropped in zip(lines, drop, strict=True):
        if not dropped:
            out.append(line)
        elif not out or out[-1] != _AISAN_MARK:
            out.append(_AISAN_MARK)
    return "\n".join(out)


def normalise(
    text: str,
    *,
    root: Path | None = None,
    paths: tuple[tuple[Path, str], ...] = (),
) -> str:
    """Replace this host's paths in an explain report with stable placeholders.

    For snapshot tests. The report describes one host in four ways and only
    four: the home directory, the system temp dir (which the runtime dir's
    digest hangs off), the checkout the caller passed as the root, and aisan's
    own runtime binds. None of them is a property of the policy, and all of them
    differ between a developer machine and CI, so a raw snapshot would fail
    everywhere except where it was generated -- and a snapshot that fails for a
    reason nobody caused is a snapshot people regenerate reflexively, which is
    how a preset drifts behind a green test.

    The mount indices go too. They are byte offsets into an argv whose first
    thirty-odd tokens are a fixed system preamble, so they carry no information
    the line ORDER does not already carry, and they shift wholesale when
    anything earlier gains a mount -- turning one real change into a diff
    against every line below it.

    `paths` is the caller's own: a profile names paths aisan cannot classify,
    because it does not know what they meant. The V8 preset's is the sibling
    MAIN checkout, reached through the worktree's dep symlinks and its .git --
    under $HOME on a real host, so covered, but under a per-run temp dir in a
    test, where nothing built in would catch it. Applied under the same
    longest-first rule as the rest, so a caller path inside $HOME still wins.

    Deliberately NOT normalised: mount order, mount kinds, sizes, env values,
    and the bwrap flags. Those are the policy. If one of them changes, the test
    is supposed to fail.
    """
    import re
    import sys
    import tempfile

    subs = [
        *((str(Path(p).resolve()), name) for p, name in paths),
        # The launcher's interpreter, which appears in the argv of any box with
        # egress (`<python> -m aisan.launch`). Same reasoning as the runtime
        # binds `_collapse_aisan_runtime` drops: it is wherever aisan happens to
        # be installed -- a checkout's venv here, a vendored submodule's venv in
        # a consumer -- so a snapshot carrying it fails for whoever runs the
        # suite from the other one. The token stays, because "a launcher runs
        # here" IS policy; only the install path goes.
        (str(Path(sys.executable)), "<AISAN PYTHON>"),
        (str(root.resolve()), "<ROOT>") if root else None,
        (str(Path.home()), "<HOME>"),
        (tempfile.gettempdir(), "<TMP>"),
    ]
    text = _collapse_aisan_runtime(text)
    # Longest first: a home under the temp dir (some CI images) would otherwise
    # have its prefix eaten by the shorter match and never hit the right rule.
    for pair in sorted([s for s in subs if s], key=lambda s: -len(s[0])):
        text = text.replace(*pair)
    return re.sub(r"^(\s*)\[\s*\d+\]", r"\1[..]", text, flags=re.MULTILINE)


def _cli_inputs(argv: list[str]) -> tuple[tuple[str, str], ...]:
    return (("argv", " ".join(argv)),)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: explain a preset without a config file or a running job.

    `--dry-run` is the point of this entry: a preset is a pure function from
    arguments to a spec, so explaining one needs no deployment, no config, no
    job and no credential -- which means an operator on a fresh host, or a
    reviewer reading a diff, can see the profile a change produces. Anything
    that needed a config to answer "what does this preset mount" would be
    answering a question about a deployment instead.
    """
    import argparse
    import sys

    from .presets import PRESETS

    p = argparse.ArgumentParser(prog="aisan explain")
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("root", type=Path, help="the box's rw root (e.g. a worktree)")
    p.add_argument("--box-id", default="explain", help="opaque box identity")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="describe only; never start a backend (the default and only mode"
        " today, kept explicit so a future --run cannot be the default)",
    )
    p.add_argument("--no-argv", action="store_true", help="omit the argv section")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    spec = PRESETS[args.preset](args.root)
    box = Box(spec, box_id=args.box_id)
    with box.staged():
        print(
            explain(
                box,
                inputs=(("preset", args.preset), ("root", str(args.root))),
                argv=not args.no_argv,
                color=sys.stdout.isatty(),
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
