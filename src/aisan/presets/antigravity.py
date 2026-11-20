# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Confine Antigravity CLI with isolated per-repository state.

``--gemini_dir`` points agy at a per-repository writable state directory in
place of the host's ``~/.gemini``. Sessions, settings and workspace trust
persist there.

The box doesn't mount the host login, which lives in the keyring, and reaches
no D-Bus session bus. agy then reads its login from a token file in the state
directory; the egress backend binds a placeholder over that path, and the host
proxy replaces the placeholder with the host credential.

``agy_argv`` tells agy that bubblewrap already confines the client and every
command it launches. Without the flag agy runs commands in its own terminal
sandbox, nested inside this one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import DARWIN_DEFAULT_PATH, RO, RW, Bind, BindSpec, host_platform
from ..spec import DEFANG_ENV, NESTING_ENV, BoxSpec, Limits

# Where agy keeps its login below its state directory when it has no keyring.
TOKEN_FILE = Path("antigravity-cli") / "antigravity-oauth-token"


def agy_binary() -> Path | None:
    """Return the host ``agy`` entry point, if installed."""
    found = shutil.which("agy")
    return Path(found) if found else None


def agy_argv(extra: tuple[str, ...] = (), *, state: Path) -> list[str]:
    """Build arguments for agy running in the outer sandbox."""
    return [
        "agy",
        "--gemini_dir",
        str(state),
        "--dangerously-skip-permissions",
        *extra,
    ]


def antigravity(
    worktree: Path,
    *,
    state: Path,
    egress: tuple[Backend, ...] = (),
    extra_ro: tuple[Path, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 2048,
    home_size_mb: int = 1024,
    memory_max: str = "8G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
    platform: str | None = None,
) -> BoxSpec:
    """Build an Antigravity CLI confinement spec for ``worktree``."""
    home = Path.home()
    is_darwin = (host_platform() if platform is None else platform) == "darwin"
    binds: list[BindSpec] = [
        *(Bind(p, RO, optional=True, guard=False) for p in extra_ro),
        # Keep shared Git objects writable, pin steering files read-only, and
        # hide sibling worktrees. Pin packs read-only in an isolated network
        # so Git GC cannot prune objects only a hidden sibling reaches.
        *git_binds(worktree, pin_packs=unshare_net),
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        # No display or browser variable is set: an agy that finds no login
        # would otherwise start an OAuth flow in the host browser.
        env=(
            *DEFANG_ENV.items(),
            *NESTING_ENV.items(),
            *GC_ENV,
            ("HOME", str(home)),
            ("PATH", DARWIN_DEFAULT_PATH if is_darwin else "/usr/bin"),
            *extra_env,
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
        allow_local_loopback=is_darwin,
    )


def antigravity_default(worktree: Path) -> BoxSpec:
    """The deployment-free registry shape used by ``explain``."""
    return antigravity(worktree, state=worktree / ".aisan-antigravity-state")
