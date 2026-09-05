# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Starting the host-side children that refresh a credential.

These children are the one part of aisan that runs as the operator, outside any
box, holding the real credential. Everything else here is about what the box may
reach; this module is about what those children may read.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .private import host_child_root, prepare_private_dir


@dataclass(frozen=True)
class HostChild:
    """The cwd and complete environment for one host credential child."""

    cwd: Path
    env: dict[str, str]


@contextmanager
def neutral_child(env: Mapping[str, str] | None = None) -> Iterator[HostChild]:
    """An empty, box-hidden cwd and normalized environment for a host child.

    Without one a child inherits the launcher's cwd, which is normally the very
    repository the boxed agent has been editing. Both `claude` and `codex` read
    configuration out of the directory they start in, so a child started there
    is one agent-written file away from running the agent's code as the operator
    -- the box's whole point, undone by the refresh path. `--safe-mode` closes
    some of that for `claude` and there is no equivalent for `codex`, so the cwd
    is closed here instead, for every such child.

    A fresh directory per child, removed when it exits: nothing to read, and
    nothing left for the next one to find. It lives below a fixed private root,
    not the ambient TMPDIR, and every Box seals that root. Otherwise an operator
    whose TMPDIR is inside a writable bind would put this host process back in
    reach of the box.

    The path-valued process metadata is normalized with the cwd. Merely passing
    cwd to subprocess leaves PWD and several temp variables pointing back at the
    launcher's repository or another box-visible directory.
    """
    root = prepare_private_dir(host_child_root())
    with tempfile.TemporaryDirectory(prefix="child-", dir=root) as raw_path:
        path = Path(raw_path)
        child_env = dict(os.environ if env is None else env)
        child_env.update(
            {
                "PWD": str(path),
                "TMPDIR": str(path),
                "TMP": str(path),
                "TEMP": str(path),
            }
        )
        child_env.pop("OLDPWD", None)
        child_env.pop("INIT_CWD", None)
        yield HostChild(path, child_env)
