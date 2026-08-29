# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The host<->box state seam: the per-session state dir and the seeds in it.

The state dir is bound rw into the box, so between sessions the agent controls
its contents -- most dangerously by planting a symlink where the launcher next
writes a fixed-name seed. The launcher runs as the operator, OUTSIDE the box, so
a seed write that follows a planted link truncates or creates whatever the link
points at, with the operator's authority: an escape straight out of the sandbox.

`runtime.py` already treats its own dir as hostile (0o700, lstat, uid and
permission checks); this gives the state dir the same rigor and makes every seed
write refuse to follow a link. It imports nothing from the rest of aisan so the
three seed call sites (`session`, `session_mcp`, `cli/claude`) can all use it
without an import cycle.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def prepare_state_dir(state: Path) -> Path:
    """Create the state dir private to this user, tighten a stale one, return it.

    0o700 because it holds the client's transcripts, its seeded config, and the
    imported MCP declarations (which exist to carry tokens). `mkdir` does not
    change an existing directory's mode, so a dir left 0o755 by an older version
    is tightened here rather than trusted: this dir is stable per (client, repo),
    so refusing a pre-existing loose one would wedge every future session.
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


def write_sealed(path: Path, data: str | bytes, *, mode: int = 0o600) -> None:
    """Write `data` to `path` as a fresh regular file, never through a symlink.

    O_NOFOLLOW refuses to open a symlink at the final component, so a planted
    `state/CLAUDE.md -> ~/.config/git/config` cannot make this truncate the host
    target. A non-regular entry (the planted link most of all) is unlinked first
    so the seed still succeeds with a real file rather than failing ELOOP -- the
    goal is to defeat the escape without handing the agent a launch-time DoS.
    """
    _unlink_if_not_regular(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    try:
        os.fchmod(fd, mode)  # exact bits regardless of umask, as the old chmod did
        os.write(fd, data.encode() if isinstance(data, str) else data)
    finally:
        os.close(fd)


def read_sealed_text(path: Path) -> str | None:
    """Read `path` without following a symlink; None if absent or not regular.

    A planted symlink reads as absent so the caller rebuilds the file rather than
    reading through the link (into a host file it does not own) and writing the
    merged result back through it.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError:
        return None  # ELOOP: a symlink is planted; treat as absent
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "r") as handle:
            return handle.read()
    except OSError:
        return None


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
