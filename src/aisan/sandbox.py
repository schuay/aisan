# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Compile a mount policy into a bubblewrap command line.

Each call runs in a fresh mount namespace. The job worktree is writable at its
absolute host path, declared dependencies are read-only, and undeclared paths
such as the host home and other worktrees are absent. An optional systemd scope
applies resource limits.

The policy is a tree keyed by box destination. The written order of the bind
list carries no meaning. Four rules decide what the box sees:

* A mount applies to its destination and everything below it that no other
  mount names. Mounts are emitted ancestor first, so the deeper destination
  wins where two overlap.
* Two identity binds at one destination merge to the stricter mode: read-only
  over overlay over writable. Any other pair at one destination is a conflict.
* A guard admits only other guards below it. Every mount is a guard unless
  declared plain, so policy written in Python is closed by default and a plain
  mount from a bind file cannot reopen part of it. Plain mounts are for
  operator declarations that other operator declarations may refine.
* A seal is an empty directory that becomes read-only after every hole through
  it has been mounted.

The fixed system surface takes part in the tree as plain entries, so a bind at
one of its paths is a conflict and a bind below one lands on top of it. bwrap
mounts that surface before anything else, so a bind above one of its paths
cannot be ordered and is refused.

Destinations are literal box paths. One that passes through a symlink the box
can see is refused, because bwrap would mount at the link's target and the
rules above would have judged the wrong path.

This layer has three known limits:

* Without `unshare_net`, the box shares the host network. With it, the box has
  no external route and reaches host-side proxies through mounted Unix sockets.
* Only tmpfs mounts have size caps; the writable worktree has no disk quota.
* `/tmp` is a fresh tmpfs for each call. Durable scratch requires an explicit
  persistent bind.
"""

from __future__ import annotations

import enum
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The fixed system surface. Bind /usr and /etc read-only and recreate the usual
# merged-/usr symlinks without exposing other directories under `/`.
_SYSTEM_ARGS = (
    "--ro-bind", "/usr", "/usr",
    "--symlink", "usr/bin", "/bin",
    "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib64", "/lib64",
    "--symlink", "usr/sbin", "/sbin",
    "--ro-bind", "/etc", "/etc",
    "--proc", "/proc",
    "--dev", "/dev",
)  # fmt: skip

# See `_journal_socket_mount`.
_JOURNAL_STDOUT_SOCK = Path("/run/systemd/journal/stdout")

_ISOLATION_ARGS = (
    "--unshare-pid",
    "--unshare-ipc",
    "--unshare-uts",
    "--cap-drop",
    "ALL",
    "--die-with-parent",
)


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
    tool caches that require writes even for reads. In particular, vpython needs
    a lock file to use its multi-gigabyte cache; a read-only bind fails, while an
    absent cache triggers an offline rebuild that hangs.

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
    mount points. A seal hides the contents with an empty tmpfs, then remounts it
    read-only after all later mounts have created their holes.

    Git worktrees require this stronger operation. Pinning the sibling configs
    found during assembly captures only a snapshot; a job could later add a
    sibling or invent a directory containing `config.worktree`. Sealing the
    entire worktrees directory hides every sibling, while a guarded writable
    bind below it restores the current worktree's directory.

    A seal is always a guard: only guards may be mounted below it.

    bwrap cannot create a mount point inside an already read-only tmpfs. The
    read-only remount therefore runs after the complete bind list. This behavior
    was verified with bwrap 0.11.2.
    """

    path: Path
    # Missing destinations usually indicate caller error because bwrap may
    # create them through a writable host bind. Reserved internal paths opt in
    # because they must be hidden before their first use.
    allow_missing: bool = False


BindSpec = Bind | BindOver | Overlay | Seal


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
    """A destination passes through a symlink the box can see.

    bwrap follows it, so the mount would land somewhere the tree did not
    reason about; a relative link can redirect a plain bind into a guard.
    """


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
    RW = "rw"


# Identity binds that may merge at one destination, strictest first.
_MERGEABLE = (Op.RO, Op.OVERLAY, Op.RW)


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

    @property
    def _identity(self) -> tuple[object, ...]:
        return (self.op, self.src, self.size, self.allow_missing)

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
                self.size, self.allow_missing,
            )  # fmt: skip
        mergeable = (
            not self.fixed
            and not other.fixed
            and self.op in _MERGEABLE
            and other.op in _MERGEABLE
        )
        if not mergeable:
            raise MountConflict(
                f"mount conflict at {self.dst}: {self._describe()} and"
                f" {other._describe()} cannot both apply; remove one"
            )
        op = min(self.op, other.op, key=_MERGEABLE.index)
        return Entry(self.dst, op, self.dst, self.guard or other.guard, optional)

    def _describe(self) -> str:
        if self.src is None:
            return self.op.value
        if self.src != self.dst:
            return f"{self.op.value} from {self.src}"
        return self.op.value


@dataclass(frozen=True)
class Mount:
    """One resolved mount operation, in the order bwrap will apply it.

    The leak check, argv renderer, and inspector all consume this representation
    so validation and display cannot drift from the mounted policy.
    """

    op: str  # ro | rw | tmpfs | overlay | seal-ro
    dst: Path
    src: Path | None = None
    size: int = 0
    guard: bool = False

    @property
    def covers(self) -> bool:
        """Return whether this operation can shadow an earlier mount at `dst`."""
        return self.op != "seal-ro"


def _bindable(p: Path) -> bool:
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


# Read-only binds already emitted by `_SYSTEM_ARGS`, parsed as source and
# destination pairs so deduplication and reachability checks use the same list.
_SYSTEM_RO_BINDS = tuple(
    (Path(_SYSTEM_ARGS[i + 1]), Path(_SYSTEM_ARGS[i + 2]))
    for i in range(len(_SYSTEM_ARGS) - 2)
    if _SYSTEM_ARGS[i] == "--ro-bind"
)


def _surface_entries() -> tuple[Entry, ...]:
    """Return the fixed system surface as tree entries.

    Everything `wrapper()` mounts before the policy is declared here, so a
    bind at one of these paths is a conflict rather than a silent cover, and
    a bind above one is refused. The journal socket is declared whether or
    not the host has it: the tree is pure, and a bind there is wrong either
    way. Built per call because tests substitute the system roots.
    """
    out = [Entry(dst, Op.RO, src, fixed=True) for src, dst in _SYSTEM_RO_BINDS]
    for i, arg in enumerate(_SYSTEM_ARGS):
        if arg == "--proc":
            out.append(Entry(Path(_SYSTEM_ARGS[i + 1]), Op.PROC, fixed=True))
        elif arg == "--dev":
            out.append(Entry(Path(_SYSTEM_ARGS[i + 1]), Op.DEV, fixed=True))
    sock = _JOURNAL_STDOUT_SOCK
    out.append(Entry(sock, Op.RO, sock, fixed=True))
    return tuple(out)


def _surface_dsts() -> frozenset[Path]:
    return frozenset(e.dst for e in _surface_entries())


def system_ro_roots() -> frozenset[Path]:
    """Return identity-mounted system roots that need no second read-only bind.

    Callers that build bind lists skip these so the reported policy does not
    list a mount the box receives anyway.

    A non-identity system bind does not qualify because it leaves the source's
    own path unmounted.
    """
    return frozenset(dst for src, dst in _SYSTEM_RO_BINDS if src == dst)


# Symlinks `_SYSTEM_ARGS` creates inside the box, link to absolute target. They
# exist only in the box, so a host-side check cannot see them.
_SYSTEM_SYMLINKS = {
    Path(_SYSTEM_ARGS[i + 2]): Path("/") / _SYSTEM_ARGS[i + 1]
    for i in range(len(_SYSTEM_ARGS) - 2)
    if _SYSTEM_ARGS[i] == "--symlink"
}


def through_system_symlink(path: Path) -> Path:
    """Return `path` with a leading system symlink replaced by its target.

    `/bin/python3` is `/usr/bin/python3` inside the box. A caller that derives
    a destination from a host path uses this so the destination names the
    directory the box has, not the link bwrap would follow.
    """
    for link, target in _SYSTEM_SYMLINKS.items():
        if path.is_relative_to(link):
            return target / path.relative_to(link)
    return path


def _journal_socket_mount() -> Mount | None:
    """The journal socket bind, when the host has one.

    A login shell may invoke `systemd-cat` from `/etc/profile.d`. `/run` is
    otherwise absent, so this bind lets that connect without a harmless
    journal error in every command result.
    """
    if _JOURNAL_STDOUT_SOCK.is_socket():
        return Mount("ro", _JOURNAL_STDOUT_SOCK, _JOURNAL_STDOUT_SOCK)
    return None


def _system_mounts() -> list[Mount]:
    """Return the host paths the fixed system surface exposes, in bwrap order.

    Policy checks need the mounts the box receives implicitly as well as those
    requested by the caller.
    """
    mounts = [Mount("ro", dst, src) for src, dst in _SYSTEM_RO_BINDS]
    journal = _journal_socket_mount()
    return mounts if journal is None else [*mounts, journal]


def _resolved(p: Path) -> Path:
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
    ra, rb = _resolved(a), _resolved(b)
    return ra.is_relative_to(rb) or rb.is_relative_to(ra)


def _reachable_through(mounts: list[Mount], path: Path) -> tuple[Path, ...]:
    """Return mount sources that leave any part of `path` readable in the box.

    Process mounts in order because a later tmpfs, seal, or unrelated bind can
    hide an earlier publication. Track each box destination separately: one host
    path may be mounted under several names, and hiding one does not hide the
    others.

    Resolve sources because they belong to the host filesystem, where symlink
    aliases identify the same data. Keep destinations literal because they
    belong to the box: `_check_symlink_free` refuses any destination bwrap
    would redirect, so host-side resolution would only invent masks that the
    box never receives.
    """
    target = _resolved(path)
    visible: dict[Path, Path] = {}
    for m in mounts:
        if not m.covers:
            continue
        for shadowed in [p for p in visible if p.is_relative_to(m.dst)]:
            del visible[shadowed]
        if m.src is None:
            continue
        src = _resolved(m.src)
        if not paths_overlap(target, src):
            continue
        # Preserve the protected path's offset when its ancestor is mounted.
        inside = target.relative_to(src) if target.is_relative_to(src) else Path()
        visible[m.dst / inside] = m.src
    # Reassigning an existing key records the source of the winning mount.
    return tuple(visible.values())


def _validate_destination(path: Path) -> None:
    """Require a box path whose kernel meaning matches its written shape.

    Relative destinations depend on the launcher cwd. Normalizing parent
    traversal is unsafe when an earlier component is a symlink, while leaving it
    intact makes policy checks and the kernel interpret different paths.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"mount destination must be absolute and contain no '..': {path}"
        )


def _entry(spec: BindSpec) -> Entry:
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
    raise TypeError(f"not a bind spec: {spec!r}")


def _merge(entries: Iterable[Entry]) -> dict[Path, Entry]:
    """Build the destination tree, merging entries that share a destination.

    Destinations are literal box paths. Two spellings that resolve to one host
    directory stay separate entries because the box needs both names; a venv's
    unversioned interpreter link and its target are the usual case.
    """
    tree: dict[Path, Entry] = {}
    for e in entries:
        _validate_destination(e.dst)
        held = tree.get(e.dst)
        tree[e.dst] = e if held is None else held.merge(e)
    return tree


def _check_guards(tree: dict[Path, Entry]) -> None:
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


def _check_surface(tree: dict[Path, Entry]) -> None:
    """Refuse an entry above a fixed system surface path.

    bwrap mounts the surface before the policy, so a bind above `/dev` or the
    journal socket would land on top of it whatever the tree says. A bind
    below one follows it in the usual order and is fine.
    """
    surface = _surface_dsts()
    for dst in tree:
        if dst in surface:
            continue
        for below in surface:
            if below != dst and below.is_relative_to(dst):
                raise MountConflict(
                    f"mount at {dst} lies above the fixed system surface at"
                    f" {below}, which bwrap mounts first; nothing can be"
                    " mounted above it, so name a path below it instead"
                )


def _host_view(tree: dict[Path, Entry], c: Path) -> Path | None:
    """Return the host path the box shows at `c` before anything mounts there.

    The nearest entry strictly above `c` decides. None, or a tmpfs or seal,
    means bwrap creates plain directories on the way to a deeper mount, so
    nothing at `c` comes from the host. Any entry with a source shows that
    source's content, offset by `c`'s position below it.
    """
    for parent in c.parents:
        above = tree.get(parent)
        if above is None:
            continue
        if above.src is None:
            return None
        return above.src / c.relative_to(parent)
    return None


def _check_symlink_free(tree: dict[Path, Entry]) -> None:
    """Refuse a destination that passes through a symlink the box can see.

    bwrap resolves each destination inside the box it is building. A relative
    symlink on the way, or at the destination itself, is followed, and the
    mount lands at the link's target. The tree reasons about the written path,
    so that landing spot escapes the deeper-wins, same-destination, and guard
    rules: a plain bind through `wt/tools -> ../main/.git` would reopen the
    guarded hooks directory. An absolute link fails inside bwrap instead;
    refusing it here gives the same message.

    Only links the box shows matter. A component below an unmounted path or a
    tmpfs is a directory bwrap creates, whatever the host has there, which
    keeps a symlinked home directory usable. A proper ancestor that is itself
    a destination is a mount point, not a link. A component `lstat` cannot
    read counts as a directory: bwrap runs as the same user and could not
    traverse it either. The merged-`/usr` links the system surface creates are
    links in the box whatever the host has, so they are checked by name.

    The check reads the host at resolution time. A link planted afterwards by
    another host process is outside the model; the host is trusted.
    """
    for dst in tree:
        for c in (dst, *dst.parents):
            if c != dst and c in tree:
                continue
            if c in _SYSTEM_SYMLINKS:
                target = f" -> {_SYSTEM_SYMLINKS[c]}"
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
                f"{target}, which the box sees; bwrap would mount at the"
                " link's target, so name that path instead"
            )


def _mount(e: Entry) -> Mount | None:
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
    assert e.src is not None
    if e.src != e.dst:
        if not e.src.exists():
            raise FileNotFoundError(f"bind-over source missing: {e.src}")
    elif e.optional:
        if not _bindable(e.src):
            return None
    elif not e.src.exists():
        raise FileNotFoundError(f"bind source missing: {e.src}")
    return Mount(e.op.value, e.dst, e.src, guard=e.guard)


@dataclass(frozen=True)
class Sandbox:
    """An immutable confinement profile for one job.

    `wrapper()` returns the command prefix for running inside the profile.
    """

    # The writable job root and cwd. It retains its absolute host path because
    # remote execution resolves build inputs by that path. Guards below it,
    # such as the .git gitdir pointer, make paths inside it read-only.
    root: Path
    # Everything else the box may touch. Order carries no meaning; see the
    # module docstring for the rules that decide overlaps.
    binds: tuple[BindSpec, ...] = ()
    # (mount point, size in bytes). A bind below a tmpfs lands on top of it, so
    # the worktree stays visible under the blanked $HOME.
    tmpfs: tuple[tuple[str, int], ...] = ()
    # The complete box environment, applied after `--clearenv` so host
    # credentials cannot leak through inheritance. Callers add noninteractive
    # build defaults explicitly from `spec.DEFANG_ENV`.
    env: tuple[tuple[str, str], ...] = ()
    # cgroup limits, applied via `systemd-run --user --scope` when available.
    # Empty string / 0 skips that property.
    memory_max: str = ""
    cpu_quota: str = ""
    tasks_max: int = 0
    use_cgroup: bool = True
    # systemd slice to place the scope under (`--slice=`). A stable cgroup anchor
    # for host-side accounting; empty leaves systemd's default.
    slice_unit: str = ""
    # Give the box a private network namespace with its own loopback and no
    # external route. An L3 allowlist cannot provide the same boundary because
    # RBE and other Google services share frontend address ranges.
    #
    # Host loopback listeners are unreachable from this namespace. Proxies need
    # an in-box TCP relay connected to a bind-mounted Unix socket; omitting that
    # relay disables their service.
    unshare_net: bool = False

    def entries(self) -> list[Entry]:
        """Return every declared mount as a tree entry, system surface included.

        The surface takes part so a bind at `/usr` collapses into the one the
        box already has, and a bind at `/dev` or a writable one at `/usr` is a
        conflict.
        """
        out = list(_surface_entries())
        out += [
            Entry(Path(m), Op.TMPFS, fixed=True, size=size) for m, size in self.tmpfs
        ]
        out.append(Entry(self.root, Op.RW, self.root, fixed=True))
        out += [_entry(spec) for spec in self.binds]
        return out

    def tree(self) -> dict[Path, Entry]:
        """Return the merged, guard-checked destination tree.

        Pure: no filesystem access. Raises `MountConflict` or `GuardViolation`
        for a policy that has no single meaning.
        """
        tree = _merge(self.entries())
        # Before the guard check: a guard above the surface would otherwise be
        # told to declare itself plain, which is refused as well.
        _check_surface(tree)
        _check_guards(tree)
        return tree

    def resolve(self) -> list[Mount]:
        """Compile the policy into ordered mount operations.

        `wrapper()`, the exposure checks, and the inspector consume the result.
        Resolution omits unreachable optional sources and raises for missing
        mandatory sources, so the returned operations describe the actual box.
        The system surface is excluded because `wrapper()` emits it separately.

        Ancestors precede descendants, so the deeper destination wins where two
        overlap. Seals remount read-only after the complete list because bwrap
        cannot create a mount point inside a read-only tmpfs.

        Destinations are checked against the host here, not in `tree()`, which
        stays free of filesystem access.
        """
        tree = self.tree()
        _check_symlink_free(tree)
        mounts: list[Mount] = []
        seal_ro: list[Mount] = []
        surface = _surface_dsts()
        for e in sorted(tree.values(), key=lambda e: e.dst.parts):
            if e.dst in surface:
                continue
            m = _mount(e)
            if m is None:
                continue
            mounts.append(m)
            if e.op is Op.SEAL:
                seal_ro.append(Mount("seal-ro", e.dst, guard=True))
        return [*mounts, *seal_ro]

    def exposed_path(
        self,
        paths: Iterable[Path],
        *,
        allowed_sources: Iterable[Path] = (),
    ) -> tuple[Path, Path] | None:
        """Return the first protected path left readable through a mount source.

        Evaluate the resolved mounts after the fixed system surface because that
        ordered result determines whether the payload can read a path. The input
        bind list alone omits both implicit mounts and later masks.

        `allowed_sources` names intentional holes inside a protected directory.
        Ignore a source at or below an allowed path, but still reject an ancestor
        that also publishes protected siblings. Examine every publication so an
        allowed mount cannot hide another alias to the protected path.

        Checking whether a requested bind contains a credential is insufficient
        when home resides below a system root such as `/usr/local`:

        * The implicit `/usr` bind exposes the credential directory until the
          home tmpfs below it hides it.
        * A requested `/usr` bind compiles to no operation because the system
          surface already mounted it.

        The final mount list decides whether to refuse the box.
        """
        mounts = [*_system_mounts(), *self.resolve()]
        allowed = tuple(_resolved(p) for p in allowed_sources)
        for path in paths:
            for source in _reachable_through(mounts, path):
                resolved_source = _resolved(source)
                if any(resolved_source.is_relative_to(p) for p in allowed):
                    continue
                return source, path
        return None

    def wrapper(self) -> list[str]:
        """Return the systemd and bwrap argv prefix for a command."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root missing: {self.root}")
        mounts = self.resolve()
        argv = list(self._cgroup_args())
        argv += ["bwrap", *_SYSTEM_ARGS]
        if self.unshare_net:
            argv += ["--unshare-net"]
        journal = _journal_socket_mount()
        argv += _mount_args([journal] if journal else [])
        argv += _mount_args(mounts)
        argv += _ISOLATION_ARGS
        argv += ["--clearenv"]
        for k, v in self.env:
            argv += ["--setenv", k, v]
        argv += ["--chdir", str(self.root)]
        return argv

    def _cgroup_args(self) -> list[str]:
        # Hosts without a systemd user manager still use bwrap without cgroups.
        if not self.use_cgroup or shutil.which("systemd-run") is None:
            return []
        args = ["systemd-run", "--user", "--scope", "-q", "--collect"]
        if self.slice_unit:
            args += [f"--slice={self.slice_unit}"]
        if self.memory_max:
            args += ["-p", f"MemoryMax={self.memory_max}"]
        if self.cpu_quota:
            args += ["-p", f"CPUQuota={self.cpu_quota}"]
        if self.tasks_max:
            args += ["-p", f"TasksMax={self.tasks_max}"]
        return [*args, "--"]

    # The mount namespace enforces containment. A second in-process copy of the
    # policy could drift from the emitted argv; tools needing confinement must
    # run inside the wrapper.


def _mount_args(mounts: list[Mount]) -> list[str]:
    """Format resolved mounts as ordered bwrap arguments."""
    argv: list[str] = []
    for m in mounts:
        match m.op:
            case "ro" | "rw":
                flag = "--ro-bind" if m.op == "ro" else "--bind"
                argv += [flag, str(m.src), str(m.dst)]
            case "tmpfs":
                if m.size:
                    argv += ["--size", str(m.size)]
                argv += ["--tmpfs", str(m.dst)]
            case "overlay":
                argv += ["--overlay-src", str(m.src), "--tmp-overlay", str(m.dst)]
            case "seal-ro":
                argv += ["--remount-ro", str(m.dst)]
    return argv
