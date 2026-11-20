# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Compile a mount policy into a bubblewrap command line.

Each call runs in a fresh mount namespace. The job worktree is writable at its
absolute host path, declared dependencies are read-only, and undeclared paths
such as the host home and other worktrees are absent. An optional systemd scope
applies resource limits.

The destination tree, merge rules, guard invariants, and exposure checks live in
`policy_tree` so platform backends enforce identical rules. The fixed Linux
system surface takes part in the tree as plain entries, so a bind at one of its
paths is a conflict and a bind below one lands on top of it. bwrap mounts that
surface before anything else, so a bind above one of its paths cannot be ordered
and is refused.

This layer has three known limits:

* Without `unshare_net`, the box shares the host network. With it, the box has
  no external route and reaches host-side proxies through mounted Unix sockets.
* Only tmpfs mounts have size caps; the writable worktree has no disk quota.
* `/tmp` is a fresh tmpfs for each call. Durable scratch requires an explicit
  persistent bind.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .policy_tree import (
    DARWIN_DEFAULT_BINS,
    DARWIN_DEFAULT_PATH,
    RO,
    RW,
    Bind,
    BindOver,
    BindSpec,
    EnsurePath,
    Entry,
    GuardViolation,
    Mode,
    Mount,
    MountConflict,
    Op,
    Overlay,
    PinnedSymlink,
    SandboxBackend,
    Seal,
    SymlinkDestination,
    WriteRegex,
    check_guards,
    check_surface,
    check_symlink_free,
    entry_for,
    first_exposure,
    host_platform,
    merge_entries,
    mount_for,
    paths_overlap,
)

__all__ = [
    "DARWIN_DEFAULT_BINS",
    "DARWIN_DEFAULT_PATH",
    "RO",
    "RW",
    "Bind",
    "BindOver",
    "BindSpec",
    "BwrapSandbox",
    "EnsurePath",
    "Entry",
    "GuardViolation",
    "Mode",
    "Mount",
    "MountConflict",
    "Op",
    "Overlay",
    "PinnedSymlink",
    "Sandbox",
    "SandboxBackend",
    "Seal",
    "SymlinkDestination",
    "WriteRegex",
    "host_platform",
    "paths_overlap",
    "system_ro_roots",
    "through_system_symlink",
]

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


@dataclass(frozen=True)
class Sandbox:
    """An immutable Linux bubblewrap confinement profile for one job.

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
    # Size in bytes of a tmpfs mounted at `root` in place of the host directory.
    # The payload's writes then stay in memory, capped by this size, and vanish
    # with the box; binds below `root` supply its inputs. Zero binds the host
    # directory.
    root_tmpfs: int = 0

    def __post_init__(self) -> None:
        if self.root_tmpfs < 0:
            raise ValueError(f"root_tmpfs must not be negative: {self.root_tmpfs}")

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
        if self.root_tmpfs:
            out.append(Entry(self.root, Op.TMPFS, fixed=True, size=self.root_tmpfs))
        else:
            out.append(Entry(self.root, Op.RW, self.root, fixed=True))
        for spec in self.binds:
            match spec:
                case PinnedSymlink(src=src, dst=dst):
                    raise ValueError(
                        f"PinnedSymlink ({src} -> {dst}) is a macOS Seatbelt"
                        " construct; use BindOver on Linux"
                    )
                case WriteRegex(path=p):
                    raise ValueError(
                        f"WriteRegex ({p}) is a macOS Seatbelt construct; use"
                        " Overlay on Linux"
                    )
                case _:
                    out.append(entry_for(spec))
        return out

    def tree(self) -> dict[Path, Entry]:
        """Return the merged, guard-checked destination tree.

        Pure: no filesystem access. Raises `MountConflict` or `GuardViolation`
        for a policy that has no single meaning.
        """
        tree = merge_entries(self.entries())
        # Before the guard check: a guard above the surface would otherwise be
        # told to declare itself plain, which is refused as well.
        check_surface(tree, _surface_dsts(), tool="bwrap")
        check_guards(tree)
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
        check_symlink_free(tree, _SYSTEM_SYMLINKS, tool="bwrap")
        mounts: list[Mount] = []
        seal_ro: list[Mount] = []
        surface = _surface_dsts()
        for e in sorted(tree.values(), key=lambda e: e.dst.parts):
            if e.dst in surface:
                continue
            m = mount_for(e)
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
        ordered result determines whether the payload can read a path. Checking
        only the requested bind list misses implicit mounts (such as `/usr`
        when home lives under `/usr/local`) and later masks.
        """
        paths = tuple(paths)
        if not paths:
            return None
        return first_exposure(self._published(), paths, allowed_sources)

    def _published(self) -> list[Mount]:
        """The ordered mounts a payload sees, implicit surface included."""
        return [*_system_mounts(), *self.resolve()]

    def sealed_exposure(self) -> tuple[Path, Path] | None:
        """Return the first mount source republishing a sealed directory.

        A `Seal` empties one destination; any other mount whose source resolves
        into the sealed host directory publishes the same data under a second
        name. Each seal carries its own `allow` hole list, so each receives its
        own pass over the shared mount list.
        """
        seals = [b for b in self.binds if isinstance(b, Seal)]
        if not seals:
            return None
        mounts = self._published()
        for seal in seals:
            hit = first_exposure(mounts, (seal.path,), seal.allow)
            if hit is not None:
                return hit
        return None

    def assert_runtime_visible(self, runtime_dir: Path) -> None:
        """Verify that the optional runtime-directory bind survived resolution."""
        if not any(m.dst == runtime_dir and m.op == "ro" for m in self.resolve()):
            raise ValueError(
                f"spec has egress backends but does not bind the runtime dir"
                f" {runtime_dir} ro -- the launcher would have no egress"
                " control data"
            )

    def wrapper(self) -> list[str]:
        """Return the systemd and bwrap argv prefix for a command."""
        if not self.root_tmpfs and not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root missing: {self.root}")
        hit = self.sealed_exposure()
        if hit is not None:
            src, path = hit
            raise ValueError(
                f"mount {src} would republish the sealed directory {path} under"
                " a second name, where the seal does not reach"
            )
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


BwrapSandbox = Sandbox


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
