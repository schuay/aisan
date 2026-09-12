# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Render an ordered mount policy as a bubblewrap command line.

Each call runs in a fresh mount namespace. The job worktree is writable at its
absolute host path, declared dependencies are read-only, and undeclared paths
such as the host home and other worktrees are absent. An optional systemd scope
applies resource limits.

Mount order is the policy: when paths overlap, the later bwrap mount wins. No
separate rule gives read-only or writable mounts priority, so reading the list
from top to bottom reveals the effective policy.

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

# A login shell may invoke `systemd-cat` from `/etc/profile.d`. Because `/run` is
# otherwise absent, expose this socket when present to prevent harmless journal
# connection errors from appearing in every command result.
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
    """Whether a bind is read-only or writable.

    Its position in the bind list determines precedence.
    """

    RO = "ro"
    RW = "rw"


RO = Mode.RO
RW = Mode.RW


@dataclass(frozen=True)
class Bind:
    """Mount `path` into the box at its own absolute path.

    Later overlapping binds win. A pin, a hole through it, and another pin
    inside the hole are represented as three binds in that order; the modes do
    not alter their precedence.

    Set `optional` only when a missing source is safe to omit, such as a vanished
    dependency symlink target. A missing guard must fail because omitting it can
    leave the underlying path writable. Sources are mandatory by default.
    """

    path: Path
    mode: Mode
    optional: bool = False


@dataclass(frozen=True)
class BindOver:
    """Mount `src` read-only at a different path, `dst`.

    This supplies a file where software already expects it, such as a per-job
    `/etc/hosts` or a build configuration in another checkout. It is always
    mandatory because omitting it would silently expose the original file.
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

    An overlay remains writable inside the box. Use a later read-only `Bind` for
    files that must not change even temporarily, such as pinned Git config or
    credentials. Overlays are mandatory because silently omitting a cache can
    hang the tool that needs it.
    """

    path: Path


@dataclass(frozen=True)
class Seal:
    """Replace `path` with an empty directory nothing can be created in.

    A read-only directory still exposes existing entries and may permit new
    mount points. A seal hides the contents with an empty tmpfs, then remounts it
    read-only after all later mounts have created their holes.

    Git worktrees require this stronger operation. Pinning the sibling configs
    found during assembly captures only a snapshot; a job could later add a
    sibling or invent a directory containing `config.worktree`. Sealing the
    entire worktrees directory hides every sibling, while a later writable bind
    restores the current worktree's directory.

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


def _system_ro_roots() -> frozenset[Path]:
    """Return identity-mounted system roots that need no second read-only bind.

    Re-emitting one after a tmpfs could expose host contents again. This once
    happened when a resolved interpreter root caused `/usr` to be mounted after
    the home tmpfs.

    A non-identity system bind does not qualify because it leaves the source's
    own path unmounted.
    """
    return frozenset(dst for src, dst in _SYSTEM_RO_BINDS if src == dst)


def _system_mounts() -> list[Mount]:
    """Return the complete fixed system surface in bwrap order.

    Policy checks need the mounts the box receives implicitly as well as those
    requested by the caller.
    """
    return [Mount("ro", dst, src) for src, dst in _SYSTEM_RO_BINDS]


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
    belong to the box. bwrap rejects a symlink destination rather than following
    it, so host-side resolution would invent masks that the box never receives.
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


def _strict_ancestor(a: Path, b: Path) -> bool:
    """Return whether resolved path `a` is a strict ancestor of `b`."""
    try:
        ra, rb = a.resolve(), b.resolve()
    except OSError:
        return False
    return ra != rb and rb.is_relative_to(ra)


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


@dataclass(frozen=True)
class Sandbox:
    """An immutable confinement profile for one job.

    `wrapper()` returns the command prefix for running inside the profile.
    """

    # The writable job root and cwd. It retains its absolute host path because
    # remote execution resolves build inputs by that path. It is mounted after
    # tmpfs entries and before `binds`, allowing later guards such as the .git
    # gitdir pointer to make paths inside it read-only.
    root: Path
    # Everything else the box may touch, in mount order: later wins.
    #
    # One safety rule adjusts the written order: a read-only strict ancestor of
    # the writable root or a tmpfs is hoisted before that mount. Such ancestors
    # often depend on host layout, including editable source roots and resolved
    # interpreter chains. Requiring callers to predict them would make the same
    # profile unsafe on another host. This rule fixed `/usr` being remounted
    # after the home tmpfs and exposing the host home read-only. All other binds
    # retain their relative order.
    binds: tuple[BindSpec, ...] = ()
    # (mount point, size in bytes). Mounted before the root and the binds so a
    # bind under a tmpfs (the worktree under the blanked $HOME) lands on top.
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

    def resolve(self) -> list[Mount]:
        """Compile the bind list into ordered mount operations.

        `wrapper()`, `_assert_no_leak`, and the inspector consume the result.
        Resolution omits unreachable optional sources and raises for missing
        mandatory sources, so the returned operations describe the actual box.
        """
        tmpfs_mounts = [Path(m) for m, _ in self.tmpfs]
        mounts: list[Mount] = []
        # A seal becomes read-only only after all holes through it are mounted.
        seal_ro: list[Mount] = []
        # Track the effective mode at each literal destination to avoid duplicate
        # binds. Do not resolve these paths: a venv's unversioned interpreter
        # symlink and its target need separate destinations even when they resolve
        # to one host directory. A covering mount invalidates entries below it so
        # a guard repeated after a hole is preserved.
        in_effect: dict[Path, Mode] = {}

        def emit(m: Mount) -> None:
            _validate_destination(m.dst)
            mounts.append(m)
            if not m.covers:
                return
            for k in [k for k in in_effect if k == m.dst or k.is_relative_to(m.dst)]:
                del in_effect[k]

        early: list[Bind] = []  # Read-only ancestors mounted before tmpfs entries.
        mid: list[Bind] = []  # Read-only ancestors mounted before the root.
        rest: list[BindSpec] = []
        for spec in self.binds:
            if isinstance(spec, Bind):
                # Validate every bind in this pass so hoisted and ordinary binds
                # handle missing sources consistently.
                if spec.optional:
                    if not _bindable(spec.path):
                        continue
                elif not spec.path.exists():
                    raise FileNotFoundError(f"bind source missing: {spec.path}")
                if spec.mode is RO:
                    if spec.path.resolve() in _system_ro_roots():
                        continue
                    if any(_strict_ancestor(spec.path, t) for t in tmpfs_mounts):
                        early.append(spec)
                        continue
                    if _strict_ancestor(spec.path, self.root):
                        mid.append(spec)
                        continue
            rest.append(spec)

        def emit_bind(b: Bind) -> None:
            if in_effect.get(b.path) is b.mode:
                return
            emit(Mount(b.mode.value, b.path, b.path))
            in_effect[b.path] = b.mode

        for b in early:
            emit_bind(b)
        for mnt, size in self.tmpfs:
            emit(Mount("tmpfs", Path(mnt), size=size))
        for b in mid:
            emit_bind(b)
        emit(Mount("rw", self.root, self.root))
        in_effect[self.root] = RW
        for spec in rest:
            match spec:
                case Bind():
                    emit_bind(spec)
                case Overlay(path=p):
                    if not p.is_dir():
                        raise FileNotFoundError(f"tmp-overlay source missing: {p}")
                    emit(Mount("overlay", p, p))
                case Seal(path=p, allow_missing=allow_missing):
                    if not allow_missing and not p.is_dir():
                        raise FileNotFoundError(f"seal source missing: {p}")
                    if p.exists() and not p.is_dir():
                        raise NotADirectoryError(f"seal path is not a directory: {p}")
                    emit(Mount("tmpfs", p))
                    seal_ro.append(Mount("seal-ro", p))
                case BindOver(src=src, dst=dst):
                    if not src.exists():
                        raise FileNotFoundError(f"bind-over source missing: {src}")
                    emit(Mount("ro", dst, src))
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
          later home tmpfs hides it.
        * A requested `/usr` bind compiles to no operation because the system
          surface already mounted it.

        The final mount order, rather than a bind's spelling, therefore decides
        whether to refuse the box.
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
        # A later mount covering an earlier tmpfs or the writable root can expose
        # the host home, make the worktree read-only, or replace it with other
        # host contents. Reject such profiles before launch.
        self._assert_no_leak(mounts)
        argv = list(self._cgroup_args())
        argv += ["bwrap", *_SYSTEM_ARGS]
        if self.unshare_net:
            argv += ["--unshare-net"]
        argv += self._journal_socket_args()
        argv += _mount_args(mounts)
        argv += _ISOLATION_ARGS
        argv += ["--clearenv"]
        for k, v in self.env:
            argv += ["--setenv", k, v]
        argv += ["--chdir", str(self.root)]
        return argv

    def _assert_no_leak(self, mounts: list[Mount]) -> None:
        """Reject a later mount that covers an earlier tmpfs or writable root.

        Validate the resolved mount list used by both rendering and inspection
        so the checked representation cannot differ from the displayed one.
        """
        root = self.root.resolve()
        protected = [
            (i, m.dst.resolve())
            for i, m in enumerate(mounts)
            if m.op == "tmpfs" or (m.op == "rw" and m.dst.resolve() == root)
        ]
        for pi, pp in protected:
            for mi, m in enumerate(mounts):
                if mi <= pi or not m.covers:
                    continue
                # Equality also shadows the earlier mount.
                if pp.is_relative_to(m.dst.resolve()):
                    raise ValueError(
                        f"sandbox mount order leaks: {m.dst} (index {mi}) covers"
                        f" protected {pp} (index {pi}) -- a later bind shadows"
                        f" a tmpfs or the rw root"
                    )

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

    @staticmethod
    def _journal_socket_args() -> list[str]:
        # Let login-shell `systemd-cat` calls connect without exposing `/run`.
        if _JOURNAL_STDOUT_SOCK.is_socket():
            return ["--ro-bind", str(_JOURNAL_STDOUT_SOCK), str(_JOURNAL_STDOUT_SOCK)]
        return []

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
