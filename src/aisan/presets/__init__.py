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

from ..egress.base import Backend
from ..spec import BoxSpec
from .claude_code import claude_code, claude_code_default
from .codex import codex, codex_default
from .opencode import opencode, opencode_default
from .v8_job import v8_job, v8_job_default, v8_rbe

PRESETS: dict[str, Callable[[Path], BoxSpec]] = {
    "claude_code": claude_code_default,
    "codex": codex_default,
    "opencode": opencode_default,
    "v8_job": v8_job_default,
}

# What `--egress NAME` resolves through. A profile is a function from the box's
# root to the backends that tree wants, so everything project-specific about an
# egress route -- an RBE project, where a checkout keeps its config -- lives in
# the preset module that already owns those facts, and the launchers know only
# that names exist.
EGRESS_PROFILES: dict[str, Callable[[Path], tuple[Backend, ...]]] = {
    "v8-rbe": v8_rbe,
}

__all__ = [
    "EGRESS_PROFILES",
    "PRESETS",
    "claude_code",
    "claude_code_default",
    "codex",
    "codex_default",
    "opencode",
    "opencode_default",
    "v8_job",
    "v8_job_default",
    "v8_rbe",
]
