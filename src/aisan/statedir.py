# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Safely manage persistent, box-writable client state and seed files.

The agent controls this directory between sessions. It could plant a symlink at
a fixed seed path before the host-side launcher writes there as the operator.
Following that link would let the box truncate or create an arbitrary host file.

State directories receive the same ownership and permission checks as runtime
directories, and seed I/O never follows symlinks. This module has no aisan
imports so all seed writers can use it without creating an import cycle.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def prepare_state_dir(state: Path) -> Path:
    """Create a private state directory and tighten its permissions if needed.

    State contains client transcripts, seeded configuration, and MCP declarations
    that may carry tokens. ``mkdir`` doesn't change the mode of an existing
    directory, so tighten directories left by older versions. Refusing them would
    prevent every future session for the same client and repository.
    """
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = state.lstat()
    except OSError as e:
        raise RuntimeError(f"cannot inspect state directory {state}: {e}") from e
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"state path is not a directory: {state}")
    if info.st_uid != os.getuid():
        raise PermissionError(f"state directory is not owned by this user: {state}")
    if info.st_mode & 0o077:
        state.chmod(0o700)
    return state


# Credential paths relative to each client's state directory. Redirecting
# CLAUDE_CONFIG_DIR, CODEX_HOME, or XDG_DATA_HOME makes an in-box login write to
# these paths instead of the host configuration. Ordinary boxed sessions pass
# credentials through the environment and do not write these files.
BOXED_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "claude": (".credentials.json",),
    "codex": ("auth.json",),
    "opencode": ("opencode/auth.json",),
}


def planted_credentials(state: Path, client: str) -> list[Path]:
    """Find credentials left in client state by an earlier boxed session.

    The state directory is writable and persists across sessions for a
    repository. A login during a networked session could therefore expose its
    token to later sessions without adding a credential mount.

    ``lstat`` counts symlinks because an occupied credential path is sufficient.
    """
    found: list[Path] = []
    for name in BOXED_CREDENTIALS.get(client, ()):
        path = state / name
        try:
            os.lstat(path)
        except OSError:
            continue
        found.append(path)
    return found


def write_sealed(path: Path, data: str | bytes, *, mode: int = 0o600) -> None:
    """Write data to a regular file without following a final symlink.

    ``O_NOFOLLOW`` prevents a planted link such as
    ``state/CLAUDE.md -> ~/.config/git/config`` from truncating its host target.
    Remove a non-regular entry first so it can't block future launches.
    """
    _unlink_if_not_regular(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    try:
        os.fchmod(fd, mode)  # Set exact permissions regardless of the process umask.
        os.write(fd, data.encode() if isinstance(data, str) else data)
    finally:
        os.close(fd)


def read_sealed_text(path: Path) -> str | None:
    """Read a regular file without following a final symlink.

    Return ``None`` for an absent, non-regular, or symlinked path. The caller can
    then rebuild the file without reading or overwriting a symlink target.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError:
        return None  # A symlink or other unreadable path is treated as absent.
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "r") as handle:
            return handle.read()
    except OSError:
        return None


def read_sealed_object(path: Path) -> dict:
    """Read a JSON object safely, returning an empty object for invalid input.

    Seed writers merge their keys into files that boxed clients can also change.
    Rebuild missing, replaced, malformed, and non-object files so later launches
    can continue.
    """
    text = read_sealed_text(path)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _unlink_if_not_regular(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(info.st_mode):
        return
    if stat.S_ISDIR(info.st_mode):
        raise IsADirectoryError(f"seed path is a directory: {path}")
    path.unlink()
