# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""aisan: one API for a confined box with audited egress.

    async with Box(spec, box_id="whatever-you-call-it") as box:
        box.env                  # environment serialized into bwrap argv
        argv = box.command(["bash", "-lc", "..."])

`spec` is a `BoxSpec`: an ordered bind list, an environment, limits, and zero or
more egress backends. Build one by hand, or call a preset -- presets are pure
functions returning the same object, so nothing a preset does is unavailable to
a caller writing it out.

The core modules are workload-neutral. Provider and client knowledge is kept in
the `egress`, `proxy`, and `presets` integration modules; the package boundary
test ensures those integrations remain standalone and declare their external
dependencies.

The renderer is deliberately NOT re-exported here. `explain` and `launch` are
run as `python -m`, and runpy warns and re-executes a module the package
__init__ has already imported -- so both are reached as submodules
(`from .explain import explain`) by the call sites that want them.
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
