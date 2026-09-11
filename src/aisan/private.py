# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-only paths shared by runtime control and credential children.

The root does not follow TMPDIR. A box may be given a TMPDIR inside its
writable worktree, and host-side capabilities placed there would then cross
the boundary by accident -- the hazard being that TMPDIR is a variable other
tools set for their own reasons, so following it means following decisions
nobody made about this.

``AISAN_PRIVATE_ROOT`` overrides it, which is not the same bargain: nothing
else writes that name, so it moves only when someone means to move it. It
exists because a box SEALS this root, so aisan running inside a box -- its own
test suite, most of all -- has no usable one and every box-staging test fails
on a directory it cannot create. A box gets the override in its environment
(see spec.NESTING_ENV) and nests cleanly; a host sets nothing and keeps the
fixed path.

The override buys an attacker nothing. Setting it requires control of the
environment, which is already control of PYTHONPATH and PATH -- code execution
inside this process, next to the credentials themselves, which is strictly
more than relocating a socket. And it is not a way into a box either: box
environments are built with --clearenv, so the variable reaches one only when
a spec names it. What guards the directory is `prepare_private_dir` below,
which validates whatever root it is handed.

The name is short because runtime UNIX socket paths live below it and Linux
limits those to 108 bytes; an overriding caller inherits that budget, which
`runtime.prepare_runtime_dir` checks rather than assumes.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_SYSTEM_TEMP_ROOT = Path("/") / "tmp"
_DEFAULT_ROOT = _SYSTEM_TEMP_ROOT / f"aisan-{os.getuid()}"
_PRIVATE_ROOT = Path(os.environ.get("AISAN_PRIVATE_ROOT") or _DEFAULT_ROOT)
_HOST_CHILD_DIR = "host-children"

# The root a box names for an aisan nested inside it. Here rather than in spec,
# with the other root: it is the same concept, and the sibling construction
# above is what keeps a literal "/tmp" out of the source.
_NESTED_ROOT = _SYSTEM_TEMP_ROOT / f"aisan-nested-{os.getuid()}"


def nested_root() -> Path:
    """The root a box offers an aisan running inside it.

    Under the box's own /tmp, which is a tmpfs it owns: nothing written there
    reaches the host, and the nested aisan creates it 0700 like any other root.
    Deliberately not the sealed name -- a box able to write to THAT path would
    be writing where the host binds its runtime dir.
    """
    return _NESTED_ROOT


def private_root() -> Path:
    """The per-user root that every box must hide."""
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
