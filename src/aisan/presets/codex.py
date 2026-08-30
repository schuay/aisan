# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Codex profile: an externally sandboxed CLI with isolated state.

``CODEX_HOME`` points at a per-repository writable state directory, never the
host's ``~/.codex``. Sessions and local preferences persist there, while the
Responses backend supplies highest-precedence CLI config for the provider route
and disabled non-model egress. The user config remains writable for repository
trust and TUI preferences.

The host credential remains absent by subtraction. Codex sees only the
backend's placeholder environment variable, and the host-side proxy replaces
that bearer after the request has crossed the network namespace.

Codex normally creates its own command sandbox. This profile already runs the
whole client inside bubblewrap, including every command it launches, so callers
must use ``codex_argv``: its external-sandbox flag avoids nesting a second
sandbox whose namespace operations may be unavailable and whose writable-root
model would duplicate the outer box.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import RO, RW, Bind, BindSpec
from ..spec import DEFANG_ENV, BoxSpec, Limits


def codex_binary() -> Path | None:
    """The ``codex`` entry point on this host, or None."""
    found = shutil.which("codex")
    return Path(found) if found else None


def codex_argv(
    extra: tuple[str, ...] = (), *, overrides: tuple[str, ...] = ()
) -> list[str]:
    """Codex argv for a client already confined by this profile."""
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
    """The spec for a Codex session on ``worktree``."""
    home = Path.home()
    binds: list[BindSpec] = [
        *(Bind(p, RO, optional=True) for p in extra_ro),
        # The .git policy for a linked worktree: the common dir rw so git works,
        # its steering files pinned ro, sibling worktrees sealed away, and this
        # session's own dir punched back through. Empty for a plain checkout,
        # whose .git is inside the rw root already.
        #
        # `pin_packs` under unshare_net: the seal removes the siblings' HEAD and
        # index as reachability roots, so an in-box `git gc` would prune objects
        # only they reference out of the SHARED store (measured -- gitbinds
        # documents it). A ro objects/pack stops that; it also stops any in-box
        # fetch, which a box with its own network namespace cannot do anyway.
        *git_binds(worktree, pin_packs=unshare_net),
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            *DEFANG_ENV.items(),
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
