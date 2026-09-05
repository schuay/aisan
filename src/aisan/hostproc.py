# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Starting the host-side children that refresh a credential.

These children are the one part of aisan that runs as the operator, outside any
box, holding the real credential. Everything else here is about what the box may
reach; this module is about what those children may read.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def neutral_cwd() -> Iterator[str]:
    """An empty directory to start a host credential child in.

    Without one a child inherits the launcher's cwd, which is normally the very
    repository the boxed agent has been editing. Both `claude` and `codex` read
    configuration out of the directory they start in, so a child started there
    is one agent-written file away from running the agent's code as the operator
    -- the box's whole point, undone by the refresh path. `--safe-mode` closes
    some of that for `claude` and there is no equivalent for `codex`, so the cwd
    is closed here instead, for every such child.

    A fresh directory per child, removed when it exits: nothing to read, and
    nothing left for the next one to find.
    """
    with tempfile.TemporaryDirectory(prefix="aisan-host-child-") as path:
        yield path
