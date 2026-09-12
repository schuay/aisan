# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Manage host-only paths for runtime control and credential processes.

The root ignores ``TMPDIR`` because it may point inside a box-writable worktree.
Placing host capabilities there would expose them to the box.

``AISAN_PRIVATE_ROOT`` is an explicit override for nested aisan processes. Each
box hides the host root, so an aisan process inside the box needs a different
root for its own runtime files. See ``spec.NESTING_ENV``.

Controlling this override already requires control of the process environment,
including ``PYTHONPATH`` and ``PATH``. Box environments use ``--clearenv`` and
receive the variable only through an explicit spec. ``prepare_private_dir``
validates either root.

The default path is short enough to leave room under Linux's 108-byte UNIX
socket path limit. An overriding caller must stay within the same limit.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_SYSTEM_TEMP_ROOT = Path("/") / "tmp"
_DEFAULT_ROOT = _SYSTEM_TEMP_ROOT / f"aisan-{os.getuid()}"
_PRIVATE_ROOT = Path(os.environ.get("AISAN_PRIVATE_ROOT") or _DEFAULT_ROOT)
_HOST_CHILD_DIR = "host-children"

# A private root for aisan processes launched inside a box.
_NESTED_ROOT = _SYSTEM_TEMP_ROOT / f"aisan-nested-{os.getuid()}"


def nested_root() -> Path:
    """Return the private root available to aisan inside a box.

    This path lies on the box's private ``/tmp`` and can't reach the host. It
    differs from the sealed host path so nested processes can't write where the
    host binds runtime directories.
    """
    return _NESTED_ROOT


def private_root() -> Path:
    """Return the per-user host root hidden from every box."""
    return _PRIVATE_ROOT


def host_child_root() -> Path:
    """The directory containing empty cwd directories for host children."""
    return private_root() / _HOST_CHILD_DIR


def prepare_private_dir(path: Path) -> Path:
    """Create and validate a private directory under the private root.

    The final component must be a real directory owned by this user with no
    group or other access. The directory may live below shared ``/tmp`` and hold
    unauthenticated sockets or host processes with durable credentials.
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
