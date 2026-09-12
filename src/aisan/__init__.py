# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Run commands in a confined box with audited egress.

    async with Box(spec, box_id="whatever-you-call-it") as box:
        box.env                  # environment serialized into bwrap argv
        argv = box.command(["bash", "-lc", "..."])

``BoxSpec`` contains ordered binds, environment, limits, and egress backends.
Callers can construct one directly or adjust the value returned by a preset.

The core modules are workload-neutral. Provider and client knowledge is kept in
the `egress`, `proxy`, and `presets` integration modules; the package boundary
test ensures those integrations remain standalone and declare their external
dependencies.

``explain`` and ``launch`` aren't re-exported because ``python -m`` warns and
re-executes modules already imported by the package initializer.
"""

from __future__ import annotations

from .box import Box
from .egress.base import Backend, PreflightError
from .runtime import runtime_dir
from .spec import BoxSpec, Limits

__all__ = [
    "Backend",
    "Box",
    "BoxSpec",
    "Limits",
    "PreflightError",
    "runtime_dir",
]
