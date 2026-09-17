# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Describe the effective bubblewrap confinement profile.

Mount order, optional sources, hoisting, and deferred remounts determine the
final filesystem view. Reports use the same staged ``BoxSpec`` as a real launch
and show both a classified mount list and the exact argument vector. Staging
creates bind sources without minting credentials or opening sockets.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .box import Box
from .sandbox import Sandbox


@dataclass(frozen=True)
class MountLine:
    """A mount parsed from the wrapper arguments, keyed by destination path."""

    kind: str  # system | proc | dev | symlink | ro | ro-pin | ro-shadow | ro-sub
    #           | rw-root | rw | tmpfs | overlay | seal-ro
    path: str
    idx: int  # token index of the mount spec (for ordering)
    size: str | None = None  # tmpfs only


# Colors distinguish effective access: red for a shadowed read-only bind, yellow
# for writable mounts, green for guards, cyan for substitutions, and dim for
# fixed or read-only surface.
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

# Collapse the fixed system surface into one line while retaining its paths.
_SURFACE_KINDS = frozenset({"system", "proc", "dev", "symlink"})


def _kind_field(kind: str, color: bool, width: int = 9) -> str:
    """Format a padded mount kind with optional ANSI color."""
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


def _pins_and_shadows(sb: Sandbox) -> tuple[set[Path], set[Path]]:
    """Classify read-only mounts affected by a covering writable mount.

    A later read-only mount pins a path inside a writable region. A later
    writable mount shadows the earlier read-only mount. Derive both sets from
    resolved mount order.
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
    """Parse supported sandbox arguments into mounts, environment, and cwd.

    Argument order defines mount precedence. The resolved bind list identifies
    read-only pins and profile mounts.
    """
    mounts: list[MountLine] = []
    env: dict[str, str] = {}
    chdir = ""
    pinned, shadowed = _pins_and_shadows(sb)
    policy = _policy_dests(sb)
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--tmpfs" and i + 1 < n:
            # A size, when present, immediately precedes --tmpfs.
            size = argv[i - 1] if i >= 2 and argv[i - 2] == "--size" else None
            mounts.append(MountLine("tmpfs", argv[i + 1], i, size))
            i += 2
        elif a == "--tmp-overlay" and i + 1 < n:
            mounts.append(MountLine("overlay", argv[i + 1], i))
            i += 2
        elif a == "--remount-ro" and i + 1 < n:
            # Seal the empty tmpfs after mounting its allowed descendants.
            mounts.append(MountLine("seal-ro", argv[i + 1], i))
            i += 2
        elif a == "--ro-bind" and i + 2 < n:
            src, dst = argv[i + 1], Path(argv[i + 2])
            if src != argv[i + 2]:
                # BindOver substitutes a different source at this destination.
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
            mounts.append(MountLine(kind, argv[i + 2], i))
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
            dst = argv[i + 2]
            d = Path(dst).resolve()
            kind = "rw-root" if d == Path(str(sb.root)).resolve() else "rw"
            mounts.append(MountLine(kind, dst, i))
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
    """Return whether resolved ``ancestor`` contains ``child``, including equality."""
    try:
        return Path(child).resolve().is_relative_to(Path(ancestor).resolve())
    except (OSError, ValueError):
        return False


def assembly_refusal(box: Box) -> Exception | None:
    """Return the error raised while assembling ``box``, if any.

    Assembly detects credential exposure, missing required sources, and mounts
    that shadow tmpfs or the writable root. Both builders are pure, so this
    check doesn't mint credentials or open sockets.
    """
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
    """Render the confinement report for ``box``.

    ``inputs`` supplies caller-specific context such as the selected config and
    worktree. Returning text supports both terminal output and snapshot tests.
    """
    out = io.StringIO()

    def section(title: str) -> None:
        out.write(f"\n== {title} ==\n")

    # Assembly failures have no wrapper arguments to inspect, so report the error.
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
    # Always report the network namespace, including boxes without backends.
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

    # A seal combines an empty tmpfs with a later read-only remount.
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
    from .spec import NESTING_ENV

    _NESTED_ROOT = NESTING_ENV["AISAN_PRIVATE_ROOT"]

    subs = [
        *((str(Path(p).resolve()), name) for p, name in paths),
        # Interpreter paths vary with the aisan installation layout. Label them
        # rather than eliding the binds: a reader of this report must be able to
        # see every mount the box receives, including the system surface.
        (str(Path(sys.executable)), "<AISAN PYTHON>"),
        (str(Path(sys.prefix)), "<AISAN PREFIX>"),
        (str(Path(sys.base_prefix)), "<AISAN BASE PREFIX>"),
        (str(own_source_root()), "<AISAN SRC>") if own_source_root() else None,
        (str(root.resolve()), "<ROOT>") if root else None,
        (str(Path.home()), "<HOME>"),
        (str(private_root()), "<AISAN PRIVATE>"),
        # The nested private root contains the current uid.
        (_NESTED_ROOT, "<AISAN NESTED ROOT>"),
        (tempfile.gettempdir(), "<TMP>"),
    ]
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
