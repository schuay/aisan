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

from ..spec import BoxSpec
from .claude_code import claude_code, claude_code_default
from .codex import codex, codex_default
from .opencode import opencode, opencode_default
from .v8_job import v8_job, v8_job_default

PRESETS: dict[str, Callable[[Path], BoxSpec]] = {
    "claude_code": claude_code_default,
    "codex": codex_default,
    "opencode": opencode_default,
    "v8_job": v8_job_default,
}

__all__ = [
    "PRESETS",
    "claude_code",
    "claude_code_default",
    "codex",
    "codex_default",
    "opencode",
    "opencode_default",
    "v8_job",
    "v8_job_default",
]
