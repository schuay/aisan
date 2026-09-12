# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Confine Claude Code with isolated state and host-side credentials.

The profile mounts a writable worktree and state directory plus an optional
read-only settings file. It doesn't mount ``~/.claude/.credentials.json``.
Claude Code sends a per-box placeholder to the local relay, which replaces it
with the host OAuth token or API key.

``CLAUDE_CONFIG_DIR`` points to writable session state and user instructions.
Operator settings remain read-only through ``--settings``. Standard installs
need no extra runtime bind because Claude Code and Node live beneath the
read-only ``/usr`` mount; other installations must supply ``extra_ro`` paths.

Interactive use with bubblewrap's ``--new-session`` remains untested. The
measurements for this profile used headless ``-p`` turns.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import RO, RW, Bind, BindSpec
from ..spec import DEFANG_ENV, NESTING_ENV, BoxSpec, Limits

# Return promptly when the relay is unavailable. Testing reached no answer after
# 180 seconds with retries and returned code 1 in one second with retries disabled.
_RETRY_ENV = (("CLAUDE_CODE_MAX_RETRIES", "0"),)

# Disable the updater because the isolated network can't reach npm.
_UPDATER_ENV = (("DISABLE_AUTOUPDATER", "1"),)

# Disable direct feature-flag, telemetry, and error-reporting connections to
# api.anthropic.com. ANTHROPIC_BASE_URL doesn't redirect them. A strace run saw
# ten direct connection attempts, including four successful connections with
# shared networking.
#
# Both variables are required. The first disables traffic; the second retains
# disk-cached feature flags instead of using compiled defaults. Debug logs with
# a disabled flag showed:
#
#   both set   -- "off, from GrowthBook (the disk cache of an earlier session)"
#   first only -- "off, from the default (GrowthBook is off for this session)"
_FLAGS_ENV = (
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
    ("CLAUDE_CODE_GB_DISK_CACHE_WHEN_TELEMETRY_OFF", "1"),
)


def claude_code_binary() -> Path | None:
    """Return the host ``claude`` entry point, if installed."""
    found = shutil.which("claude")
    return Path(found) if found else None


def claude_code_argv(
    prompt: str, *, settings: Path | None = None, extra: tuple[str, ...] = ()
) -> list[str]:
    """Build a headless command with an optional read-only settings file."""
    argv = ["claude"]
    if settings is not None:
        argv += ["--settings", str(settings)]
    return [*argv, *extra, "-p", prompt]


def claude_code(
    worktree: Path,
    *,
    state: Path,
    egress: tuple[Backend, ...] = (),
    config: Path | None = None,
    extra_ro: tuple[Path, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 2048,
    home_size_mb: int = 1024,
    memory_max: str = "32G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """Build a Claude Code confinement spec for ``worktree``."""
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
        # Keep operator settings read-only above the writable home tmpfs.
        *([Bind(config, RO)] if config is not None else []),
        # The writable root already exposes an embedded state directory and
        # allows Claude Code to create it after assembly.
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        # Private tmpfs mounts provide writable home and /tmp scratch space.
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            *DEFANG_ENV.items(),
            *NESTING_ENV.items(),
            *GC_ENV,
            ("HOME", str(home)),
            ("PATH", "/usr/bin"),
            # Claude Code stores sessions and reads user instructions here.
            ("CLAUDE_CONFIG_DIR", str(state)),
            *_RETRY_ENV,
            *_UPDATER_ENV,
            *_FLAGS_ENV,
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


def claude_code_default(worktree: Path) -> BoxSpec:
    """Build the deployment-independent profile used by ``explain``."""
    return claude_code(worktree, state=worktree / ".aisan-claude-state")
