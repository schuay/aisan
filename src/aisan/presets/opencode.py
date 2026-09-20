# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Confine OpenCode with isolated state and host-side credentials.

The box doesn't mount ``~/.local/share/opencode/auth.json``. OpenCode reads a
placeholder from ``OPENCODE_AUTH_CONTENT``, and the host backend replaces it on
outbound model requests.

Only ``XDG_DATA_HOME`` persists; it contains sessions, logs, and the database.
Instance locks, caches, and config remain in the private home tmpfs. Leaving
``XDG_CONFIG_HOME`` unset also keeps Git reading the read-only global config at
``~/.config/git/config``.

The optional host model catalog replaces OpenCode's older embedded catalog.
Testing found ``glm-5.3`` in the fetched catalog but absent from the embedded
catalog in OpenCode 1.18.18. Backend inline config has higher precedence than
project config, so a repository can't change the provider route.

Interactive use with bubblewrap's ``--new-session`` remains untested. The
measurements for this profile used headless ``opencode run``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import RO, RW, Bind, BindOver, BindSpec
from ..spec import DEFANG_ENV, NESTING_ENV, BoxSpec, Limits

# Disable network-dependent update and download retries. The network namespace
# enforces isolation; these settings make offline failures return promptly.
_DISABLE_ENV = (
    ("OPENCODE_DISABLE_AUTOUPDATE", "1"),
    ("OPENCODE_DISABLE_MODELS_FETCH", "1"),
    ("OPENCODE_DISABLE_LSP_DOWNLOAD", "1"),
)


# Honor the host's XDG cache location for the fetched provider catalog.
def _default_catalog() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "opencode" / "models.json"


def _catalog_bind(catalog: Path) -> list[BindSpec]:
    """Bind an existing host catalog at the box's default cache path."""
    if not catalog.is_file():
        return []
    box_dst = Path.home() / ".cache" / "opencode" / "models.json"
    if catalog.resolve() == box_dst.resolve():
        return [Bind(catalog, RO)]
    return [BindOver(catalog, box_dst)]


def opencode_binary() -> Path | None:
    """Return the host ``opencode`` entry point, if installed."""
    found = shutil.which("opencode")
    return Path(found) if found else None


def opencode(
    worktree: Path,
    *,
    state: Path,
    egress: tuple[Backend, ...] = (),
    extra_ro: tuple[Path, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    models: Path | None = None,
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 2048,
    home_size_mb: int = 1024,
    memory_max: str = "8G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """Build an OpenCode confinement spec for ``worktree``.

    ``state`` stores job-owned session history. ``models`` overrides the
    optional host catalog selected through ``XDG_CACHE_HOME``.
    """
    home = Path.home()
    catalog = models if models is not None else _default_catalog()
    binds: list[BindSpec] = [
        *(Bind(p, RO, optional=True, guard=False) for p in extra_ro),
        # Keep shared Git objects writable, pin steering files read-only, and
        # hide sibling worktrees. Plain checkouts receive the same steering pins.
        #
        # Hiding sibling refs makes their unique objects appear unreachable.
        # Pin packs read-only in an isolated network to prevent Git GC pruning
        # those objects from the shared store.
        *git_binds(worktree, pin_packs=unshare_net),
        # Mount persistent state after read-only paths so it remains writable.
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
        # Substitute the host's XDG-aware catalog at the box's default cache path.
        *_catalog_bind(catalog),
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
            # Persist sessions, logs, and the database.
            ("XDG_DATA_HOME", str(state)),
            *_DISABLE_ENV,
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


def opencode_default(worktree: Path) -> BoxSpec:
    """Build the deployment-independent profile used by ``explain``."""
    return opencode(worktree, state=worktree / ".aisan-opencode-state")
