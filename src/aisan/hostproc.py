# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Start host processes that refresh credentials.

These processes run outside the box as the operator and hold real credentials.
This module limits what they can read from the host.
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
    """Give a host child an empty, box-hidden cwd and normalized environment.

    A child would otherwise inherit the repository that the boxed agent edits.
    Both Claude and Codex read configuration from their starting directory, so
    agent-written configuration could execute as the operator during refresh.

    Each child gets a fresh directory that is removed on exit. It lives below a
    fixed private root hidden from every box; ambient ``TMPDIR`` may point into a
    writable bind.

    Normalize path-valued environment variables as well as the subprocess cwd.
    Otherwise ``PWD`` or a temp variable may still expose a box-visible path.
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
