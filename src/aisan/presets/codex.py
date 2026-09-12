# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Confine Codex with isolated per-repository state.

``CODEX_HOME`` points at a per-repository writable state directory, never the
host's ``~/.codex``. Sessions and local preferences persist there, while the
Responses backend supplies highest-precedence CLI config for the provider route
and disabled non-model egress. The user config remains writable for repository
trust and TUI preferences.

The box doesn't mount the host credential. Codex sends a placeholder bearer to
the local backend, which replaces it with the host credential.

``codex_argv`` tells Codex that bubblewrap already confines the client and every
command it launches. This avoids a nested command sandbox.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import RO, RW, Bind, BindSpec
from ..spec import DEFANG_ENV, NESTING_ENV, BoxSpec, Limits


def codex_binary() -> Path | None:
    """Return the host ``codex`` entry point, if installed."""
    found = shutil.which("codex")
    return Path(found) if found else None


def codex_argv(
    extra: tuple[str, ...] = (), *, overrides: tuple[str, ...] = ()
) -> list[str]:
    """Build arguments for Codex running in the outer sandbox."""
    config = [arg for value in overrides for arg in ("--config", value)]
    return [
        "codex",
        "--dangerously-bypass-approvals-and-sandbox",
        *config,
        *extra,
    ]


def codex(
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
) -> BoxSpec:
    """Build a Codex confinement spec for ``worktree``."""
    home = Path.home()
    binds: list[BindSpec] = [
        *(Bind(p, RO, optional=True) for p in extra_ro),
        # Keep shared Git objects writable, pin steering files read-only, and
        # hide sibling worktrees. Plain checkouts receive the same steering pins.
        #
        # Hiding sibling refs makes their unique objects appear unreachable.
        # Pin packs read-only in an isolated network to prevent Git GC pruning
        # those objects from the shared store.
        *git_binds(worktree, pin_packs=unshare_net),
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            *DEFANG_ENV.items(),
            *NESTING_ENV.items(),
            *GC_ENV,
            ("HOME", str(home)),
            ("PATH", "/usr/bin"),
            ("CODEX_HOME", str(state)),
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
    )


def codex_default(worktree: Path) -> BoxSpec:
    """The deployment-free registry shape used by ``explain``."""
    return codex(worktree, state=worktree / ".aisan-codex-state")
