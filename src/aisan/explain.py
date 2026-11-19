# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Describe the effective bubblewrap confinement profile.

The destination tree, optional sources, and deferred remounts determine the
final filesystem view. Reports use the same staged ``BoxSpec`` as a real launch
and show both a classified mount list and the exact argument vector. Staging
creates bind sources without minting credentials or opening sockets.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .box import Box
from .sandbox import Sandbox, Seal
from .seatbelt import (
    SeatbeltSandbox,
    _ere_escape,
    darwin_system_ro_roots,
    seatbelt_lexpath,
    seatbelt_path,
)


@dataclass(frozen=True)
class MountLine:
    """A mount parsed from the wrapper arguments, keyed by destination path."""

    kind: str  # system | proc | dev | symlink | ro | ro-pin | ro-sub
    #           | rw-root | rw | tmpfs | overlay | seal-ro
    path: str
    idx: int  # token index of the mount spec (for ordering)
    size: str | None = None  # tmpfs only
    guard: bool = False  # nothing plain may be mounted below it


# Colors distinguish effective access: yellow for writable mounts, green for
# pins and seals, cyan for substitutions, and dim for fixed or read-only
# surface.
_KIND_ANSI = {
    "rw-root": "33",
    "rw": "33",
    "tmpfs": "33",
    "overlay": "33",
    "write-re": "33",
    "ro-pin": "32",
    "seal-ro": "32",
    "ro-sub": "36",
    "ro": "2",
    "system": "2",
    "proc": "2",
    "dev": "2",
    "symlink": "2",
}

# Collapse the fixed system surface into one line while retaining its paths.
_SURFACE_KINDS = frozenset({"system", "proc", "dev", "symlink"})


def _kind_field(kind: str, color: bool, width: int = 8) -> str:
    """Format a padded mount kind with optional ANSI color.

    The width leaves one column after the longest kind for the guard marker, so
    paths align whether or not a line carries one.
    """
    field = f"{kind:<{width}}"
    ansi = _KIND_ANSI.get(kind)
    return f"\x1b[{ansi}m{field}\x1b[0m" if color and ansi else field


@dataclass(frozen=True)
class Profile:
    mounts: list[MountLine]
    tmpfs: list[MountLine]
    env: dict[str, str]
    chdir: str


def _policy_dests(sb: Sandbox) -> set[Path]:
    """Return destinations from the profile's bind list.

    Other wrapper mounts belong to the fixed system surface. Membership remains
    accurate if wrapper argument order changes.
    """
    dests: set[Path] = set()
    for m in sb.resolve():
        try:
            dests.add(Path(m.dst).resolve())
        except OSError:
            continue
    return dests


def _pins_and_guards(sb: Sandbox) -> tuple[set[Path], set[Path]]:
    """Return read-only destinations below a writable mount, and guards.

    A read-only mount below a writable one pins that subtree; depth decides,
    so no order is consulted. Guards come from the resolved mounts directly.
    """
    mounts = sb.resolve()
    writable = [m.dst for m in mounts if m.op in ("rw", "overlay")]
    pinned = {
        m.dst
        for m in mounts
        if m.op == "ro" and any(m.dst.is_relative_to(w) for w in writable)
    }
    guards = {m.dst for m in mounts if m.guard}
    return pinned, guards


def parse_wrapper(argv: list[str], sb: Sandbox) -> Profile:
    """Parse supported sandbox arguments into mounts, environment, and cwd.

    The argument vector is the sorted destination tree, so ancestors precede
    descendants. The resolved mount list identifies pins, guards, and profile
    mounts.
    """
    mounts: list[MountLine] = []
    env: dict[str, str] = {}
    chdir = ""
    pinned, guards = _pins_and_guards(sb)
    policy = _policy_dests(sb)
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--tmpfs" and i + 1 < n:
            # A size, when present, immediately precedes --tmpfs.
            size = argv[i - 1] if i >= 2 and argv[i - 2] == "--size" else None
            dst = Path(argv[i + 1])
            mounts.append(MountLine("tmpfs", argv[i + 1], i, size, dst in guards))
            i += 2
        elif a == "--tmp-overlay" and i + 1 < n:
            dst = Path(argv[i + 1])
            mounts.append(MountLine("overlay", argv[i + 1], i, guard=dst in guards))
            i += 2
        elif a == "--remount-ro" and i + 1 < n:
            # Seal the empty tmpfs after mounting its allowed descendants.
            mounts.append(MountLine("seal-ro", argv[i + 1], i, guard=True))
            i += 2
        elif a == "--ro-bind" and i + 2 < n:
            src, dst = argv[i + 1], Path(argv[i + 2])
            if src != argv[i + 2]:
                # BindOver substitutes a different source at this destination.
                kind = "ro-sub"
            elif dst in pinned:
                kind = "ro-pin"
            else:
                try:
                    kind = "ro" if dst.resolve() in policy else "system"
                except OSError:
                    kind = "ro"
            mounts.append(MountLine(kind, argv[i + 2], i, guard=dst in guards))
            i += 3
        elif a == "--proc" and i + 1 < n:
            mounts.append(MountLine("proc", argv[i + 1], i))
            i += 2
        elif a == "--dev" and i + 1 < n:
            mounts.append(MountLine("dev", argv[i + 1], i))
            i += 2
        elif a == "--symlink" and i + 2 < n:
            # Report merged-usr symlinks by the link path visible in the box.
            mounts.append(MountLine("symlink", argv[i + 2], i))
            i += 3
        elif a == "--bind" and i + 2 < n:
            dst = Path(argv[i + 2])
            kind = "rw-root" if dst.resolve() == sb.root.resolve() else "rw"
            mounts.append(MountLine(kind, argv[i + 2], i, guard=dst in guards))
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


def parse_seatbelt(argv: list[str], sb: SeatbeltSandbox) -> tuple[Profile, str]:
    """Parse a `SeatbeltSandbox` wrapper and resolved mounts into a `Profile`."""
    env: dict[str, str] = {}
    chdir = ""
    i = 0
    n = len(argv)
    in_env = False
    while i < n:
        tok = argv[i]
        if tok == "-C" and i + 1 < n and not in_env:
            chdir = argv[i + 1]
            i += 2
        elif tok == "--" and not in_env:
            in_env = True
            i += 1
        elif tok == "/usr/bin/sandbox-exec":
            break
        elif in_env and "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
            i += 1
        else:
            i += 1

    mounts: list[MountLine] = []
    idx = 0
    for sys_root in sorted(darwin_system_ro_roots(), key=str):
        if sys_root.exists():
            mounts.append(MountLine("system", seatbelt_path(sys_root), idx))
            idx += 1

    resolved = [m for m in sb.resolve() if m.op != "seal-ro"]
    writable = [
        Path(seatbelt_path(m.dst)) for m in resolved if m.op in ("rw", "write-regex")
    ]
    root_sb = Path(seatbelt_path(sb.root))
    for m in resolved:
        dst_sb = seatbelt_path(m.dst)
        dst_path = Path(dst_sb)
        match m.op:
            case "tmpfs":
                mounts.append(MountLine("seal-ro", dst_sb, idx, guard=True))
            case "rw":
                kind = "rw-root" if dst_path == root_sb else "rw"
                mounts.append(MountLine(kind, dst_sb, idx, guard=m.guard))
            case "write-regex":
                mounts.append(MountLine("write-re", dst_sb, idx, guard=m.guard))
            case "ro":
                is_pin = any(
                    dst_path != w and dst_path.is_relative_to(w) for w in writable
                )
                kind = "ro-pin" if is_pin else "ro"
                mounts.append(MountLine(kind, dst_sb, idx, guard=m.guard))
        idx += 1

    for sym in sb._pinned_symlinks():
        label = f"{seatbelt_lexpath(sym.dst)} -> {seatbelt_path(sym.src)}"
        mounts.append(MountLine("ro-sub", label, idx, guard=True))
        idx += 1

    return Profile(mounts=mounts, tmpfs=[], env=env, chdir=chdir), sb.render_sbpl()


def _anc_eq(child: str, ancestor: str) -> bool:
    """Return whether resolved ``ancestor`` contains ``child``, including equality."""
    try:
        return Path(child).resolve().is_relative_to(Path(ancestor).resolve())
    except (OSError, ValueError):
        return False


_ASSEMBLY_ERRORS = (ValueError, FileNotFoundError, FileExistsError, NotADirectoryError)


def assembly_refusal(box: Box) -> Exception | None:
    """Return the error raised while assembling ``box``, if any.

    Assembly detects credential exposure, missing required sources, mount
    conflicts, and plain mounts below a guard. Both builders are pure, so this
    check doesn't mint credentials or open sockets.
    """
    try:
        box._sandbox()
        box.wrapper()
    except _ASSEMBLY_ERRORS as exc:
        return exc
    return None


def explain(
    box: Box,
    *,
    inputs: tuple[tuple[str, str], ...] = (),
    argv: bool = True,
    color: bool = False,
) -> str:
    """Render the confinement report for ``box``.

    ``inputs`` supplies caller-specific context such as the selected config and
    worktree. Returning text supports both terminal output and snapshot tests.
    """
    out = io.StringIO()

    def section(title: str) -> None:
        out.write(f"\n== {title} ==\n")

    # Assembly failures have no wrapper arguments to inspect, so report the error.
    try:
        sandbox = box._sandbox()
        wrapper = box.wrapper()
    except _ASSEMBLY_ERRORS as refusal:
        section("BOX ASSEMBLY REFUSED")
        out.write(f"  {refusal}\n")
        return out.getvalue()
    is_seatbelt = isinstance(sandbox, SeatbeltSandbox)
    sbpl_text = ""
    if isinstance(sandbox, SeatbeltSandbox):
        prof, sbpl_text = parse_seatbelt(wrapper, sandbox)
    elif isinstance(sandbox, Sandbox):
        prof = parse_wrapper(wrapper, sandbox)
    else:
        raise TypeError(f"unsupported sandbox backend: {type(sandbox)!r}")
    isolated = box.spec.unshare_net
    shared_backends = [b for b in box.spec.egress if box._uses_shared_tcp(b)]
    shared_egress = bool(shared_backends)
    effective_env = dict(prof.env)
    for backend in shared_backends:
        effective_env.update(backend.shared_client_env_description())
    home = effective_env.get("HOME", "")

    section("inputs")
    for key, value in inputs:
        out.write(f"  {key:<9} {value}\n")
    # Always report the network confinement mode, including boxes without backends.
    if is_seatbelt:
        if isolated and box.spec.allow_local_loopback:
            out.write(
                "  network   macOS Seatbelt (deny network*, localhost TCP + pinned endpoints)\n"
            )
        elif isolated:
            out.write(
                "  network   macOS Seatbelt (deny network*, pinned endpoints only)\n"
            )
        else:
            out.write("  network   shared host namespace (Seatbelt allow network*)\n")
    elif isolated:
        out.write("  network   own namespace (no route off the machine)\n")
    else:
        out.write("  network   shared host namespace (full host network)\n")
    out.write(f"  box_id    {box.box_id}\n")
    out.write(f"  HOME(env) {home}\n")
    out.write(f"  chdir     {prof.chdir}\n")

    if not shared_egress:
        if is_seatbelt:
            section("egress backends (direct Unix domain sockets)")
        else:
            section("egress backends (host half on a socket, in-box on loopback)")
    elif len(shared_backends) == len(box.spec.egress):
        section("egress backends (authenticated host-loopback TCP)")
    else:
        section("egress backends (authenticated host-loopback TCP + Unix sockets)")
    if box.spec.egress:
        for backend in box.spec.egress:
            if not box._uses_shared_tcp(backend):
                sock = backend.socket_path(box.runtime_dir)
                if is_seatbelt:
                    out.write(f"  {backend.name:<8} unix://{seatbelt_lexpath(sock)}\n")
                else:
                    out.write(
                        f"  {backend.name:<8} 127.0.0.1:{backend.port} -> {sock}\n"
                    )
            else:
                out.write(f"  {backend.name:<8} 127.0.0.1:(assigned at launch)\n")
            if (route := backend.transport.describe()) is not None:
                out.write(f"  {'':<8} upstream {route}\n")
    elif isolated:
        out.write("  (none; the box has no route off the machine)\n")
    else:
        out.write("  (none; egress off -- the box shares the host network)\n")

    section("tmpfs mounts (intended writable scratch; binds below land on top)")
    if prof.tmpfs:
        for m in prof.tmpfs:
            tag = "  <- $HOME" if home and _anc_eq(home, m.path) else ""
            out.write(f"  [{m.idx:>3}] {m.path}  ({m.size or 'no size'}{tag})\n")
    else:
        out.write("  (none)\n")

    section("binds by destination (deeper wins; * marks a guard: nothing plain below)")
    ordered = [m for m in sorted(prof.mounts, key=lambda x: x.idx) if m.kind != "tmpfs"]
    surface = [m for m in ordered if m.kind in _SURFACE_KINDS]
    if surface:
        paths = " ".join(m.path for m in surface)
        out.write(f"  {_kind_field('system', color)}  {paths}\n")
    for m in ordered:
        if m.kind in _SURFACE_KINDS:
            continue
        mark = "*" if m.guard else " "
        out.write(f"  [{m.idx:>3}] {_kind_field(m.kind, color)}{mark} {m.path}\n")

    # A seal combines an empty tmpfs with a later read-only remount. Report each
    # seal's holes too: they are the only sources allowed to republish its
    # contents, so they bound the guarantee the seal otherwise makes.
    sealed = [m for m in prof.mounts if m.kind == "seal-ro"]
    if sealed:
        section(
            "sealed directories (empty in the box, nothing creatable, no alias elsewhere)"
        )
        holes: dict[str, tuple[Path, ...]] = {}
        for b in sandbox.binds:
            if not isinstance(b, Seal):
                continue
            key = seatbelt_path(b.path) if is_seatbelt else str(b.path)
            allow_items = (
                tuple(Path(seatbelt_path(h)) for h in b.allow)
                if is_seatbelt
                else b.allow
            )
            prior = holes.get(key)
            # Two seals may share a destination, and each refuses on its own
            # list. A hole is effective only where every one of them allows it.
            holes[key] = (
                allow_items
                if prior is None
                else tuple(h for h in prior if h in allow_items)
            )
        for m in sealed:
            allow = holes.get(m.path, ())
            out.write(f"  {m.path}\n")
            for hole in allow:
                out.write(f"    hole {hole}\n")

    if box.spec.confidential:
        section("confidential host paths (no mount may publish them)")
        for path in box.spec.confidential:
            out.write(f"  {path}\n")

    if not shared_egress:
        if is_seatbelt:
            section("environment (the box's complete environment; env -i first)")
        else:
            section("environment (the box's complete environment; --clearenv first)")
    else:
        section("effective environment (backend values injected by launcher)")
    for k in sorted(effective_env):
        out.write(f"  {k}={effective_env[k]}\n")

    if is_seatbelt and sbpl_text:
        section("seatbelt profile (SBPL)")
        for sb_line in sbpl_text.splitlines():
            out.write(f"  {sb_line}\n")

    if argv:
        # Show direct exec arguments one per line, without shell quoting.
        section("argv (exactly what is exec'd, one token per line)")
        for token in [*wrapper, *box.launch_prefix()]:
            out.write(f"  {token}\n")
        out.write("  <payload>\n")
    return out.getvalue()


def normalise(
    text: str,
    *,
    root: Path | None = None,
    paths: tuple[tuple[Path, str], ...] = (),
) -> str:
    """Replace host-specific report paths with stable snapshot placeholders.

    Normalize home, temp, private root, checkout, runtime, interpreter, caller
    paths, and mount indices. Preserve every mount line, its order, kind, and
    size, plus environment values and bubblewrap flags, because they define the
    policy. Apply path replacements longest first so nested caller paths retain
    their own labels.
    """
    import re
    import sys
    import tempfile

    from .launch import own_source_root
    from .private import private_root
    from .sandbox import system_ro_roots
    from .spec import NESTING_ENV

    _NESTED_ROOT = NESTING_ENV["AISAN_PRIVATE_ROOT"]

    # Interpreter prefixes vary with the aisan installation layout. Label them
    # rather than eliding the binds: a reader of this report must be able to
    # see every mount the box receives, including the system surface. A prefix
    # that is itself a system root gets no label, or every path under it would
    # be rewritten and the system surface would disappear from the report.
    all_sys_roots = system_ro_roots() | darwin_system_ro_roots()
    prefixes = [
        (Path(p), name)
        for p, name in (
            (sys.prefix, "<AISAN PREFIX>"),
            (sys.base_prefix, "<AISAN BASE PREFIX>"),
        )
        if Path(p).resolve() not in all_sys_roots
    ]

    raw_subs = [
        *((str(Path(p).resolve()), name) for p, name in paths),
        *((seatbelt_path(Path(p)), name) for p, name in paths),
        (str(Path(sys.executable)), "<AISAN PYTHON>"),
        *((str(p), name) for p, name in prefixes),
        *((seatbelt_path(p), name) for p, name in prefixes),
        (str(own_source_root()), "<AISAN SRC>") if own_source_root() else None,
        (seatbelt_path(own_source_root()), "<AISAN SRC>")
        if own_source_root()
        else None,
        (str(root.resolve()), "<ROOT>") if root else None,
        (seatbelt_path(root), "<ROOT>") if root else None,
        (str(Path.home()), "<HOME>"),
        (seatbelt_path(Path.home()), "<HOME>"),
        (str(private_root()), "<AISAN PRIVATE>"),
        (seatbelt_path(private_root()), "<AISAN PRIVATE>"),
        # The nested private root contains the current uid.
        (_NESTED_ROOT, "<AISAN NESTED ROOT>"),
        (seatbelt_path(Path(_NESTED_ROOT)), "<AISAN NESTED ROOT>"),
        (tempfile.gettempdir(), "<TMP>"),
        (seatbelt_path(Path(tempfile.gettempdir())), "<TMP>"),
    ]
    subs: list[tuple[str, str]] = []
    for pair in raw_subs:
        if pair is None:
            continue
        subs.append(pair)
        esc = _ere_escape(pair[0])
        if esc != pair[0]:
            subs.append((esc, pair[1]))
    # Replace nested paths before their parents.
    for pair in sorted([s for s in subs if s], key=lambda s: -len(s[0])):
        text = text.replace(*pair)
    return re.sub(r"^(\s*)\[\s*\d+\]", r"\1[..]", text, flags=re.MULTILINE)


def _cli_inputs(argv: list[str]) -> tuple[tuple[str, str], ...]:
    return (("argv", " ".join(argv)),)


def main(argv: list[str] | None = None) -> int:
    """Explain a preset without starting a job or reading deployment config."""
    import argparse
    import sys

    from .presets import EGRESS_PROFILES, GRANTS, PRESETS
    from .session import LaunchRefused, resolve_launcher_flags

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
    # Include launcher composition options in the reviewed profile.
    p.add_argument(
        "--egress",
        action="append",
        choices=sorted(EGRESS_PROFILES),
        metavar="NAME",
        help="add an egress profile's backends and binds (repeatable)",
    )
    p.add_argument(
        "--grant",
        action="append",
        choices=sorted(GRANTS),
        metavar="NAME",
        help="add a named grant's mounts, PATH entries and env (repeatable)",
    )
    p.add_argument(
        "--binds",
        action="append",
        type=Path,
        metavar="FILE",
        help="apply a user bind spec file, later-wins (repeatable)",
    )
    p.add_argument("--no-argv", action="store_true", help="omit the argv section")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    root = args.root.resolve()
    spec = PRESETS[args.preset](root)
    try:
        # This command explains a preset, which imports no MCP declarations, so
        # a spec file's `mcp` key has nothing to select here.
        spec = resolve_launcher_flags(
            root,
            base_egress=spec.egress,
            unshare_net=spec.unshare_net,
            egress_profiles=args.egress,
            grants=args.grant,
            binds=args.binds,
        ).apply(spec)
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code
    box = Box(spec, box_id=args.box_id)
    with box.staged():
        print(
            explain(
                box,
                inputs=(
                    ("preset", args.preset),
                    ("root", str(root)),
                    ("egress", ", ".join(args.egress) if args.egress else "(none)"),
                    (
                        "binds",
                        ", ".join(map(str, args.binds)) if args.binds else "(none)",
                    ),
                ),
                argv=not args.no_argv,
                color=sys.stdout.isatty(),
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
