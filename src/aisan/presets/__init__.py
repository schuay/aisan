# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Presets: pure functions from arguments to a `BoxSpec`. Data, not behaviour.

A preset may not do anything a caller could not have written by hand. It picks
binds, an environment and limits, and returns them; it does not start anything,
read a config file, consult the network, or hold state. That is what makes the
spec it returns adjustable (`with_binds`, `with_egress`) instead of a black box
somebody has to re-implement the moment their case differs by one mount.

The registry below is the `explain --dry-run` surface: `PRESETS[name](root)`
gives a full spec with no config file, no deployment and no credential, which is
what lets a reviewer see the profile a diff produces on a host that has never
run the app.
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

# What `--egress NAME` resolves through. A profile is a function from the box's
# root to the backends that tree wants, so everything project-specific about an
# egress route -- an RBE project, where a checkout keeps its config -- lives in
# the preset module that already owns those facts, and the launchers know only
# that names exist.
EGRESS_PROFILES: dict[str, Callable[[Path], EgressProfile]] = {
    "v8-rbe": v8_rbe,
    # The shared-network route, which trades the proxy's whole property away.
    # Spelled out in the name because the flag is the only place an operator
    # sees the choice.
    "v8-rbe-with-net-unsafe": v8_rbe_shared_net,
}

# What `--grant NAME` resolves through. A grant is a fact about the HOST rather
# than about the checkout -- where depot_tools is installed, where vpython keeps
# its venvs -- so unlike an egress profile it takes no root, and a box rooted
# anywhere gets the same one.
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
