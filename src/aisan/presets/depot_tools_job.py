# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Build confinement profiles for depot_tools workspaces.

The profile mounts the workspace read-write, external dependency targets
read-only, the Git metadata needed in the workspace, depot_tools, and a private
copy-on-write vpython cache. It also disables depot_tools operations that need
network access.

The V8-specific helpers configure remote builds. Host-side model and RBE
backends attach credentials to requests, while the box receives only local
endpoints. Without RBE, builds use Siso's offline fallback.

The default private network has no external route or access to host loopback.
Relays listen on Unix sockets mounted from the runtime directory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..egress.base import Backend, EgressProfile
from ..gitbinds import GC_ENV, external_symlink_targets, git_binds, git_host_files
from ..sandbox import RO, Bind, BindSpec, Overlay
from ..spec import DEFANG_ENV, NESTING_ENV, BoxSpec, Grant, Limits

# The wrappers otherwise update the read-only depot_tools checkout on every run.
# DEPOT_TOOLS_UPDATE disables the update, CIPD sync, and Python bootstrap. Its
# file-based alternative cannot be added beneath the read-only bind (measured
# with bubblewrap 0.12.0).
#
# `git cl presubmit` fetches the live CL description and validates OWNERS
# through authenticated Gerrit endpoints. PRESUBMIT_SKIP_NETWORK skips those
# checks because the box has no SSO cookie, while keeping local lint, formatting,
# and copyright checks.
#
# An in-box presubmit therefore doesn't validate OWNERS or the CL description.
# CQ runs the complete checks.
_DEPOT_TOOLS_ENV = (
    ("DEPOT_TOOLS_UPDATE", "0"),
    ("PRESUBMIT_SKIP_NETWORK", "1"),
)


def _vpython_cache() -> Path:
    """Return the host vpython cache path used by depot_tools."""
    return Path.home() / ".cache" / f"vpython-root.{os.getuid()}"


def _tool_cache_overlays() -> list[Path]:
    """Return host tool caches that need private copy-on-write mounts.

    vpython locks its cache even for reads, so a read-only mount fails. Without
    the cache it tried to rebuild offline and produced no output for 120 seconds
    in testing. A shared writable mount would let jobs modify roughly 4 GB of
    executable tool state. An overlay keeps the warm cache readable and confines
    writes to the box.
    """
    cache = _vpython_cache()
    return [cache] if cache.is_dir() else []


def depot_tools_root() -> Path | None:
    """Find the depot_tools checkout through ``autoninja`` on ``PATH``."""
    autoninja = shutil.which("autoninja")
    return Path(autoninja).parent if autoninja else None


def depot_tools_grant(depot_tools: Path | None = None) -> Grant:
    """Grant access to depot_tools and its copy-on-write caches.

    The grant also adds depot_tools to ``PATH`` and disables operations that
    need network access. ``depot_tools`` overrides PATH-based discovery. Return
    an empty grant when no checkout is available.
    """
    root = depot_tools if depot_tools is not None else depot_tools_root()
    if root is None:
        # The environment settings apply only when the tool tree is mounted.
        return Grant()
    return Grant(
        binds=(
            *(Overlay(p) for p in _tool_cache_overlays()),
            Bind(root, RO, optional=True),
        ),
        path=(root,),
        env=_DEPOT_TOOLS_ENV,
    )


def sisoenv_path(worktree: Path) -> Path:
    """Return the V8 checkout's ``.sisoenv`` path."""
    return (worktree / "build" / "config" / "siso" / ".sisoenv").resolve()


# The shared instance used by V8 checkouts in this deployment.
RBE_PROJECT = "rbe-chromium-untrusted"


def sisoenv_paths(root: Path) -> list[Path]:
    """Find distinct ``.sisoenv`` files in ``root`` and its direct children."""
    children = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    found = {p for p in map(sisoenv_path, (root, *children)) if p.exists()}
    return sorted(found)


def v8_rbe(root: Path) -> EgressProfile:
    """Configure host-proxied V8 remote builds for an isolated network.

    Return an empty profile when no ``.sisoenv`` exists. Import the backend here
    to keep proxy dependencies out of preset imports.
    """
    from ..egress.reapi import ReapiBackend

    paths = sisoenv_paths(root)
    if not paths:
        return EgressProfile()
    return EgressProfile(backends=(ReapiBackend(project=RBE_PROJECT, sisoenv=paths),))


def v8_rbe_shared_net(_root: Path) -> EgressProfile:
    """Configure V8 remote builds for a box sharing the host network.

    A plaintext proxy on host loopback would be available to every local
    process. This profile instead mounts the LUCI credential store read-only and
    lets Siso authenticate directly. The resulting cloud token represents the
    user's full account, including chromium-review, for the session lifetime.
    """
    from ..egress.reapi import LUCI_STORE

    if not LUCI_STORE.is_dir():
        return EgressProfile()
    return EgressProfile(
        binds=(Bind(LUCI_STORE, RO),),
        notice=(
            f"WARNING: {LUCI_STORE} is mounted in the box: the session holds a"
            " luci credential that resolves to your full account, chromium-review"
            " included, for as long as it runs."
        ),
    )


def depot_tools_job(
    worktree: Path,
    *,
    egress: tuple[Backend, ...] = (),
    extra_ro: tuple[Path, ...] = (),
    extra_path: tuple[str, ...] = (),
    depot_tools: Path | None = None,
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 4096,
    home_size_mb: int = 1024,
    memory_max: str = "32G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """Build a confinement spec rooted at ``worktree``.

    Dependency targets and ``extra_ro`` paths are optional. Git pins and the
    read-write worktree are required because they enforce the confinement
    boundary. Validate the worktree here because ``BoxSpec`` doesn't access the
    filesystem.
    """
    if not worktree.is_dir():
        raise ValueError(
            f"box root does not exist: {worktree}"
            " (a box is rooted at a real directory: bound rw, and its cwd)"
        )
    grant = depot_tools_grant(depot_tools)
    ro = [*external_symlink_targets(worktree), *extra_ro]
    home = Path.home()
    # Keep depot_tools first and include launcher directories for bare MCP commands.
    path_env = ":".join([*(str(d) for d in grant.path), *extra_path, "/usr/bin"])
    # Later mounts take precedence. Git pins follow the writable worktree during
    # resolution; Box then appends runtime, interpreter, and backend mounts.
    binds: list[BindSpec] = [
        # Consumer read-only binds may shadow paths from the grant.
        *grant.binds,
        *(Bind(p, RO, optional=True) for p in ro),
        *git_binds(worktree, pin_packs=unshare_net),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        # A size-capped tmpfs hides the real home while allowing tool caches.
        # The box also receives a private, size-capped /tmp.
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            # Sandbox clears the environment before applying these values.
            *DEFANG_ENV.items(),
            *NESTING_ENV.items(),
            ("HOME", str(home)),
            ("PATH", path_env),
            *GC_ENV,
            *grant.env,
        ),
        egress=egress,
        unshare_net=unshare_net,
        ensure=git_host_files(worktree, pin_packs=unshare_net),
        limits=Limits(
            memory_max=memory_max,
            cpu_quota=cpu_quota,
            tasks_max=tasks_max,
            slice_unit=cgroup_slice,
        ),
    )


def depot_tools_job_default(worktree: Path) -> BoxSpec:
    """Build the deployment-independent profile used by ``explain --dry-run``."""
    return depot_tools_job(worktree)
