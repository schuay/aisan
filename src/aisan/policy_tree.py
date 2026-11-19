# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Platform-neutral destination tree, guard rules, and exposure checks.

Both the Linux bubblewrap backend and the macOS Seatbelt backend validate the
same mount specifications before rendering platform-specific confinement rules:

* A mount applies to its destination and everything below it that no other
  mount names. Deeper destinations win where two overlap.
* Two identity binds at one destination merge to the stricter mode: read-only
  over overlay over writable. Any other pair at one destination is a conflict.
* A guard admits only other guards below it, so a plain mount from a user bind
  file cannot reopen part of a preset guard.
* A seal hides a directory's host contents and admits only its declared holes;
  republishing sealed data through an alias elsewhere is refused.
"""

from __future__ import annotations

import enum
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Mode(enum.Enum):
    """Whether a bind is read-only or writable."""

    RO = "ro"
    RW = "rw"


RO = Mode.RO
RW = Mode.RW


@dataclass(frozen=True)
class Bind:
    """Mount `path` into the box at its own absolute path.

    A bind covers its path and everything below it that no other mount names.
    A read-only bind below a writable one pins that subtree; a writable bind
    below a read-only one opens a hole. Two binds at the same path merge to
    the stricter mode.

    Set `optional` only when a missing source is safe to omit, such as a vanished
    dependency symlink target. A missing guard must fail because omitting it can
    leave the underlying path writable. Sources are mandatory by default.

    A bind is a guard unless `guard=False`: nothing but another guard may be
    mounted below it, so a plain mount cannot reopen part of a Git hooks
    directory, a credential store, or a hole through a `Seal`. Declare a bind
    plain only when it is an operator mount that other operator mounts may
    refine, as bind files do.
    """

    path: Path
    mode: Mode
    optional: bool = False
    guard: bool = True


@dataclass(frozen=True)
class BindOver:
    """Mount `src` read-only at a different path, `dst`.

    This supplies a file where software already expects it, such as a per-job
    `/etc/hosts` or a build configuration in another checkout. It is always
    mandatory because omitting it would silently expose the original file, and
    always a guard because it substitutes content that nothing may reopen.
    """

    src: Path
    dst: Path


@dataclass(frozen=True)
class Overlay:
    """Expose the host contents at `path` but discard changes made in the box.

    bwrap places a per-box tmpfs above the host directory. This supports shared
    tool caches that require writes even for reads, where a read-only bind fails
    on lock files and an absent cache triggers an offline rebuild that hangs.

    An overlay remains writable inside the box. Use a read-only `Bind` below it
    for files that must not change even temporarily, such as pinned Git config
    or credentials. Overlays are mandatory because silently omitting a cache can
    hang the tool that needs it. `guard` has the same meaning as on `Bind`.
    """

    path: Path
    guard: bool = True


@dataclass(frozen=True)
class Seal:
    """Replace `path` with an empty directory nothing can be created in.

    A read-only directory still exposes existing entries and may permit new
    mount points. A seal hides the contents and blocks creation, restoring only
    the holes named in `allow`.

    Git worktrees require this stronger operation. Pinning the sibling configs
    found during assembly captures only a snapshot; a job could later add a
    sibling or invent a directory containing `config.worktree`. Sealing the
    entire worktrees directory hides every sibling, while a guarded writable
    bind below it restores the current worktree's directory.

    A seal is always a guard: only guards may be mounted below it.

    A seal also declares the host data confidential, not merely the pathname:
    `sealed_exposure` refuses the profile when any other mount publishes that
    data under a second name. Sealing one destination cannot hide an alias,
    because a bind resolves its source and publishes an independent view.
    """

    path: Path
    # Missing destinations usually indicate caller error because bwrap may
    # create them through a writable host bind. Reserved internal paths opt in
    # because they must be hidden before their first use.
    allow_missing: bool = False
    # Sources at or below these paths are the seal's intentional holes: the
    # current worktree's private directory, this box's runtime directory. A
    # hole must be named here as well as mounted, because the guard permits any
    # guard below the seal and the sealed tree's siblings are what the seal
    # protects. An ancestor publishing protected siblings is still refused.
    allow: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        # A hole outside the sealed tree exempts data the seal never covered.
        # `/` would exempt everything and silently disable the check.
        outside = [
            p for p in self.allow if p == self.path or not p.is_relative_to(self.path)
        ]
        if outside:
            raise ValueError(
                f"seal {self.path} allows sources it does not strictly contain:"
                f" {outside}"
            )


@dataclass(frozen=True)
class PinnedSymlink:
    """Publish read-only `src` at `dst` via a host symlink pinned in Seatbelt.

    macOS Seatbelt cannot mount one path at another. For state directories owned
    by aisan, `Box` creates `dst -> src` before launch, removes `dst` on exit,
    and `SeatbeltSandbox` denies unlinking or replacing `dst` and writing `src`.
    Always a guard. Refused by `BwrapSandbox`, which uses `BindOver`.
    """

    src: Path
    dst: Path

    def __post_init__(self) -> None:
        validate_destination(self.src)
        validate_destination(self.dst)


@dataclass(frozen=True)
class WriteRegex:
    """Mount `path` read-only while permitting writes to matching subpaths.

    Seatbelt has no copy-on-write overlay mount. Shared caches that only touch
    lock or timestamp files during reads can stay at their host path read-only
    with narrow `file-write*` regular expressions anchored strictly below
    `path`. Each entry in `suffixes` must begin with `/` and contain no NUL or
    newline characters. Refused by `BwrapSandbox`, which uses `Overlay`.
    """

    path: Path
    suffixes: tuple[str, ...]
    guard: bool = True

    def __post_init__(self) -> None:
        validate_destination(self.path)
        if '"' in str(self.path):
            raise ValueError(f"WriteRegex path cannot contain '\"': {self.path!r}")
        if not self.suffixes:
            raise ValueError(f"WriteRegex at {self.path} requires at least one suffix")
        for suffix in self.suffixes:
            if (
                not suffix.startswith("/")
                or len(suffix) < 2
                or any(c in suffix for c in '\x00\r\n"')
                or "(?" in suffix
                or any(seg == ".." for seg in suffix.split("/"))
            ):
                raise ValueError(
                    f"WriteRegex suffix {suffix!r} must be a non-empty POSIX ERE"
                    " starting with '/' with no quotes, '..', control lines, or"
                    " '(?' groups"
                )
            try:
                re.compile(suffix[1:])
            except re.error as e:
                raise ValueError(
                    f"WriteRegex suffix {suffix!r} is not a valid POSIX ERE: {e}"
                ) from e


BindSpec = Bind | BindOver | Overlay | Seal | PinnedSymlink | WriteRegex


@dataclass(frozen=True)
class EnsurePath:
    """A host path that must exist before bind resolution.

    The box creates a missing path empty and removes it afterward if it remains
    safe to do so. This supports Git guard pins whose source files may not exist
    in a fresh checkout. Those pins prevent the box from planting config or
    hooks that host-side Git would later read.

    Declaring creation here keeps presets pure: `Box` performs the host mutation
    only while staging or running and then undoes it. `is_dir` selects `mkdir`
    and conditional `rmdir` instead of touching and unlinking a file.
    """

    path: Path
    is_dir: bool


class MountConflict(ValueError):
    """Two mounts name one destination and neither can yield to the other."""


class GuardViolation(ValueError):
    """A plain mount sits below a guard, which would reopen part of it."""


class SymlinkDestination(ValueError):
    """A destination passes through a symlink the box can see."""


class Op(enum.Enum):
    """A mount operation at one destination.

    `SEAL` and `TMPFS` hide the host path, and `PROC` and `DEV` are the
    kernel filesystems bwrap creates. The other three expose the host path
    with decreasing restriction: `RO` forbids writes, `OVERLAY` discards them,
    `RW` passes them through. That order is the tie-break when two identity
    binds name one destination.
    """

    SEAL = "seal"
    TMPFS = "tmpfs"
    PROC = "proc"
    DEV = "dev"
    RO = "ro"
    OVERLAY = "overlay"
    WRITE_REGEX = "write-regex"
    RW = "rw"


# Identity binds that may merge at one destination, strictest first.
_MERGEABLE = (Op.RO, Op.OVERLAY, Op.WRITE_REGEX, Op.RW)


@dataclass(frozen=True)
class Entry:
    """One declared mount, normalized for the destination tree.

    Every declared thing becomes an entry: the fixed system roots, tmpfs
    mounts, the writable root, and each `BindSpec`. The tree is a dict from
    `dst` to the entry that wins there.

    `fixed` marks an entry whose operation cannot be changed by merging: the
    root, because the payload runs in it; tmpfs and seals, because they hide
    rather than expose; the system surface and bind-overs, because their
    content is not the host path. Two identical fixed entries still collapse.

    The root, tmpfs mounts, and the system surface are plain, not guards:
    operator binds live below them by design, such as a cache under the home
    tmpfs, a pin inside the worktree, or a device node under `/dev`. They are
    subject to the guard rule like any other plain entry, so a guard above the
    root is a policy error.
    """

    dst: Path
    op: Op
    src: Path | None = None
    guard: bool = False
    optional: bool = False
    fixed: bool = False
    size: int = 0
    allow_missing: bool = False
    suffixes: tuple[str, ...] = ()

    @property
    def _identity(self) -> tuple[object, ...]:
        return (self.op, self.src, self.size, self.allow_missing, self.suffixes)

    def merge(self, other: Entry) -> Entry:
        """Return the entry that wins where `self` and `other` share `dst`.

        Identical entries collapse. A fixed entry defines its destination, so a
        restatement takes its guard status; a system root restated by a
        resolved interpreter prefix stays plain. Otherwise the result is a
        guard if either side was. The optional flag survives only if both had
        it. Two mergeable identity binds take the stricter mode under the same
        flag rules. Anything else is a conflict: the operator wrote two
        different things for one path.
        """
        optional = self.optional and other.optional
        if self._identity == other._identity:
            fixed = self if self.fixed else other if other.fixed else None
            guard = fixed.guard if fixed else (self.guard or other.guard)
            return Entry(
                self.dst, self.op, self.src, guard, optional, fixed is not None,
                self.size, self.allow_missing, self.suffixes,
            )  # fmt: skip
        mergeable = (
            not self.fixed
            and not other.fixed
            and self.op in _MERGEABLE
            and other.op in _MERGEABLE
            and not (self.op is Op.WRITE_REGEX and other.op is Op.WRITE_REGEX)
        )
        if not mergeable:
            raise MountConflict(
                f"mount conflict at {self.dst}: {self._describe()} and"
                f" {other._describe()} cannot both apply; remove one"
            )
        winner = min(self, other, key=lambda item: _MERGEABLE.index(item.op))
        return Entry(
            self.dst,
            winner.op,
            self.dst,
            self.guard or other.guard,
            optional,
            suffixes=winner.suffixes,
        )

    def _describe(self) -> str:
        if self.src is None:
            return self.op.value
        if self.src != self.dst:
            return f"{self.op.value} from {self.src}"
        return self.op.value


@dataclass(frozen=True)
class Mount:
    """One resolved mount operation, ordered ancestor-first.

    The leak check, argv/profile renderers, and inspector consume this
    representation so validation and display cannot drift from the active
    confinement policy.
    """

    op: str  # ro | rw | tmpfs | overlay | write-regex | seal-ro
    dst: Path
    src: Path | None = None
    size: int = 0
    guard: bool = False
    suffixes: tuple[str, ...] = ()

    @property
    def covers(self) -> bool:
        """Return whether this operation can shadow an earlier mount at `dst`."""
        return self.op != "seal-ro"


class SandboxBackend(Protocol):
    """The confinement-backend interface consumed by `Box` and `explain`."""

    root: Path
    binds: tuple[BindSpec, ...]
    env: tuple[tuple[str, str], ...]

    def exposed_path(
        self,
        paths: Iterable[Path],
        *,
        allowed_sources: Iterable[Path] = (),
    ) -> tuple[Path, Path] | None:
        """Return the first protected path left readable in the box, if any."""
        ...

    def sealed_exposure(self) -> tuple[Path, Path] | None:
        """Return the first mount source republishing a sealed directory."""
        ...

    def assert_runtime_visible(self, runtime_dir: Path) -> None:
        """Raise `ValueError` if `runtime_dir` did not survive resolution."""
        ...

    def wrapper(self) -> list[str]:
        """Return the command prefix that confines a payload."""
        ...


def host_platform() -> str:
    """Return the native confinement platform (`'darwin'` or `'linux'`)."""
    return "darwin" if sys.platform == "darwin" else "linux"


def bindable(p: Path) -> bool:
    """Return whether an optional bind source can be statted.

    `Path.exists()` suppresses only selected errors. Other failures, such as
    ENOKEY from a credential-gated network mount, would escape during sandbox
    assembly. An optional source is unusable whether it is absent or merely
    unreachable, so any stat failure omits it.

    Mandatory sources use separate checks that preserve errors. Silently
    omitting one could drop a guard and leave its underlying path writable.
    """
    try:
        p.stat()
    except OSError:
        return False
    return True


def resolved_path(p: Path) -> Path:
    """Return the canonical host path, or `p` when it cannot be resolved."""
    try:
        return p.resolve()
    except OSError:
        return p


def paths_overlap(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are the same path or one is inside the other.

    Containment in either direction exposes protected data: mounting an ancestor
    exposes the whole store, while mounting a descendant exposes part of it.
    Resolve both host paths so symlink aliases cannot evade the check.
    """
    ra, rb = resolved_path(a), resolved_path(b)
    return ra.is_relative_to(rb) or rb.is_relative_to(ra)


def reachable_through(mounts: Iterable[Mount], path: Path) -> tuple[Path, ...]:
    """Return mount sources that leave any part of `path` readable in the box.

    Process mounts in order because a later tmpfs, seal, or unrelated bind can
    hide an earlier publication. Track each box destination separately: one host
    path may be mounted under several names, and hiding one does not hide the
    others.

    Resolve sources because they belong to the host filesystem, where symlink
    aliases identify the same data. Keep destinations literal because they
    belong to the box: `check_symlink_free` refuses any destination that passes
    through a visible symlink.
    """
    target = resolved_path(path)
    visible: dict[Path, Path] = {}
    for m in mounts:
        if not m.covers:
            continue
        for shadowed in [p for p in visible if p.is_relative_to(m.dst)]:
            del visible[shadowed]
        if m.src is None:
            continue
        src = resolved_path(m.src)
        if not paths_overlap(target, src):
            continue
        # Preserve the protected path's offset when its ancestor is mounted.
        inside = target.relative_to(src) if target.is_relative_to(src) else Path()
        visible[m.dst / inside] = m.src
    # Reassigning an existing key records the source of the winning mount.
    return tuple(visible.values())


def first_exposure(
    mounts: Iterable[Mount],
    paths: Iterable[Path],
    allowed_sources: Iterable[Path],
) -> tuple[Path, Path] | None:
    """Return the first of `paths` readable through a source outside `allowed`."""
    materialized = list(mounts)
    allowed = tuple(resolved_path(p) for p in allowed_sources)
    for path in paths:
        for source in reachable_through(materialized, path):
            resolved_source = resolved_path(source)
            if any(resolved_source.is_relative_to(p) for p in allowed):
                continue
            return source, path
    return None


def validate_destination(path: Path) -> None:
    """Require a box path whose kernel meaning matches its written shape.

    Relative destinations depend on the launcher cwd. Normalizing parent
    traversal is unsafe when an earlier component is a symlink, while leaving it
    intact makes policy checks and the kernel interpret different paths.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"mount destination must be absolute and contain no '..': {path}"
        )


def entry_for(spec: BindSpec) -> Entry:
    """Normalize one declared bind into a tree entry."""
    match spec:
        case Bind(path=p, mode=mode, optional=optional, guard=guard):
            return Entry(p, Op(mode.value), p, guard=guard, optional=optional)
        case BindOver(src=src, dst=dst):
            return Entry(dst, Op.RO, src, guard=True, fixed=True)
        case Overlay(path=p, guard=guard):
            return Entry(p, Op.OVERLAY, p, guard=guard)
        case Seal(path=p, allow_missing=allow_missing):
            return Entry(
                p, Op.SEAL, guard=True, fixed=True, allow_missing=allow_missing
            )
    raise TypeError(f"unsupported bind spec for entry_for: {spec!r}")


def merge_entries(entries: Iterable[Entry]) -> dict[Path, Entry]:
    """Build the destination tree, merging entries that share a destination.

    Destinations are literal box paths. Two spellings that resolve to one host
    directory stay separate entries because the box needs both names; a venv's
    unversioned interpreter link and its target are the usual case.
    """
    tree: dict[Path, Entry] = {}
    for e in entries:
        validate_destination(e.dst)
        held = tree.get(e.dst)
        tree[e.dst] = e if held is None else held.merge(e)
    return tree


def check_guards(tree: Mapping[Path, Entry]) -> None:
    """Reject a plain entry below a guard.

    Walk every ancestor, not only the nearest named one: a guard higher up
    still forbids the entry when an unguarded mount sits in between. The
    check runs on declared entries, before optional sources are dropped, so an
    absent optional guard keeps protecting its subtree.
    """
    for dst, e in tree.items():
        if e.guard:
            continue
        for parent in dst.parents:
            above = tree.get(parent)
            if above is not None and above.guard:
                raise GuardViolation(
                    f"{e._describe()} mount at {dst} lies below the guard at"
                    f" {parent} ({above._describe()}); only guards may be"
                    " mounted below a guard, so remove the nested entry or"
                    " declare the guard plain"
                )


def check_surface(
    tree: Mapping[Path, Entry],
    surface: AbstractSet[Path],
    *,
    tool: str = "bwrap",
) -> None:
    """Refuse an entry above a fixed system surface path."""
    for dst in tree:
        if dst in surface:
            continue
        for below in surface:
            if below != dst and below.is_relative_to(dst):
                raise MountConflict(
                    f"mount at {dst} lies above the fixed system surface at"
                    f" {below}, which {tool} mounts first; nothing can be"
                    " mounted above it, so name a path below it instead"
                )


def _host_view(tree: Mapping[Path, Entry], c: Path) -> Path | None:
    """Return the host path the box shows at `c` before anything mounts there."""
    for parent in c.parents:
        above = tree.get(parent)
        if above is None:
            continue
        if above.src is None:
            return None
        return above.src / c.relative_to(parent)
    return None


def check_symlink_free(
    tree: Mapping[Path, Entry],
    system_symlinks: Mapping[Path, Path] | None = None,
    *,
    tool: str = "bwrap",
) -> None:
    """Refuse a destination that passes through a symlink the box can see."""
    links = system_symlinks or {}
    for dst in tree:
        for c in (dst, *dst.parents):
            if c != dst and c in tree:
                continue
            if c in links:
                target = f" -> {links[c]}"
            else:
                host = _host_view(tree, c)
                if host is None or not host.is_symlink():
                    continue
                try:
                    target = f" -> {host.readlink()}"
                except OSError:
                    target = ""
            raise SymlinkDestination(
                f"mount destination {dst} passes through the symlink {c}"
                f"{target}, which the box sees; {tool} would mount at the"
                " link's target, so name that path instead"
            )


def mount_for(e: Entry) -> Mount | None:
    """Turn one entry into a mount, or `None` for an unreachable optional one.

    Mandatory sources raise so a missing guard cannot silently leave its
    underlying path writable.
    """
    match e.op:
        case Op.TMPFS:
            return Mount("tmpfs", e.dst, size=e.size)
        case Op.SEAL:
            if not e.allow_missing and not e.dst.is_dir():
                raise FileNotFoundError(f"seal source missing: {e.dst}")
            if e.dst.exists() and not e.dst.is_dir():
                raise NotADirectoryError(f"seal path is not a directory: {e.dst}")
            return Mount("tmpfs", e.dst, guard=True)
        case Op.OVERLAY:
            assert e.src is not None
            if not e.src.is_dir():
                raise FileNotFoundError(f"tmp-overlay source missing: {e.src}")
            return Mount("overlay", e.dst, e.src, guard=e.guard)
        case Op.WRITE_REGEX:
            assert e.src is not None
            if not e.src.exists():
                raise FileNotFoundError(f"write-regex source missing: {e.src}")
            if not e.src.is_dir():
                raise NotADirectoryError(
                    f"write-regex source is not a directory: {e.src}"
                )
            return Mount(
                "write-regex", e.dst, e.src, guard=e.guard, suffixes=e.suffixes
            )
    assert e.src is not None
    if e.src != e.dst:
        if not e.src.exists():
            raise FileNotFoundError(f"bind-over source missing: {e.src}")
    elif e.optional:
        if not bindable(e.src):
            return None
    else:
        try:
            e.src.stat()
        except (FileNotFoundError, NotADirectoryError):
            raise FileNotFoundError(f"bind source missing: {e.src}") from None
    return Mount(e.op.value, e.dst, e.src, guard=e.guard)
