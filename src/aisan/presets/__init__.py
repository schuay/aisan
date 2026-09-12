# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Pure functions that build adjustable ``BoxSpec`` values.

Presets select binds, environment, and limits without starting services,
reading configuration, using the network, or retaining state. Registry entries
remain deployment-independent so ``explain --dry-run`` can inspect them on a
fresh host.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..egress.base import EgressProfile
from ..spec import BoxSpec, Grant
from .claude_code import claude_code, claude_code_default
from .codex import codex, codex_default
from .depot_tools_job import (
    depot_tools_grant,
    depot_tools_job,
    depot_tools_job_default,
    v8_rbe,
    v8_rbe_shared_net,
)
from .opencode import opencode, opencode_default

PRESETS: dict[str, Callable[[Path], BoxSpec]] = {
    "claude_code": claude_code_default,
    "codex": codex_default,
    "opencode": opencode_default,
    "depot_tools_job": depot_tools_job_default,
}

# Egress profiles resolve project-specific backends and mounts from the box root.
EGRESS_PROFILES: dict[str, Callable[[Path], EgressProfile]] = {
    "v8-rbe": v8_rbe,
    # This profile exposes the LUCI credential to a shared-network box.
    "v8-rbe-with-net-unsafe": v8_rbe_shared_net,
}

# Grants describe host tools and caches and don't depend on the box root.
GRANTS: dict[str, Callable[[], Grant]] = {
    "depot_tools": depot_tools_grant,
}

__all__ = [
    "EGRESS_PROFILES",
    "GRANTS",
    "PRESETS",
    "claude_code",
    "claude_code_default",
    "codex",
    "codex_default",
    "depot_tools_grant",
    "depot_tools_job",
    "depot_tools_job_default",
    "opencode",
    "opencode_default",
    "v8_rbe",
    "v8_rbe_shared_net",
]
