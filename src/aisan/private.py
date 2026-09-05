# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-only paths shared by runtime control and credential children.

The root is deliberately independent of TMPDIR. A box may be given a TMPDIR
inside its writable worktree, and host-side capabilities placed there would
then cross the boundary by accident. The name is short because runtime UNIX
socket paths live below it and Linux limits those paths to 108 bytes.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_SYSTEM_TEMP_ROOT = Path("/") / "tmp"
_PRIVATE_ROOT = _SYSTEM_TEMP_ROOT / f"aisan-{os.getuid()}"
_HOST_CHILD_DIR = "host-children"


def private_root() -> Path:
    """The fixed per-user root that every box must hide."""
    return _PRIVATE_ROOT


def host_child_root() -> Path:
    """The directory containing empty cwd directories for host children."""
    return private_root() / _HOST_CHILD_DIR


def prepare_private_dir(path: Path) -> Path:
    """Create and validate a private directory under the private root.

    The final component must be a real directory owned by this user and must
    grant no group or other access. Refusing an unsafe pre-existing path is
    important because /tmp is shared and the directory holds unauthenticated
    sockets and host processes with access to durable credentials.
    """
    root = private_root()
    try:
        path.relative_to(root)
    except ValueError as e:
        raise ValueError(f"private path is outside {root}: {path}") from e

    directories = (root,) if path == root else (root, path)
    for directory in directories:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = directory.lstat()
        except OSError as e:
            raise RuntimeError(
                f"cannot inspect private directory {directory}: {e}"
            ) from e
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"private path is not a directory: {directory}")
        if info.st_uid != os.getuid():
            raise PermissionError(
                f"private directory is not owned by this user: {directory}"
            )
        if info.st_mode & 0o077:
            raise PermissionError(f"private directory is too permissive: {directory}")
    return path
