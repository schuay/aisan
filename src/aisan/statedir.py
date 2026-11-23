# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Safely manage persistent, box-writable client state and seed files.

The agent controls this directory between sessions. It could plant a symlink at
a fixed seed path before the host-side launcher writes there as the operator.
Following that link would let the box truncate or create an arbitrary host file.

State directories receive the same ownership and permission checks as runtime
directories, and seed I/O never follows symlinks. This module imports only
`policy_tree` and `private` so all seed writers can use it without creating an
import cycle.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .policy_tree import resolved_path
from .private import prepare_private_dir, private_root


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
    "antigravity": ("antigravity-cli/antigravity-oauth-token",),
    "claude": (".credentials.json",),
    "codex": ("auth.json",),
    "opencode": ("opencode/auth.json",),
}

# Bubblewrap creates this bind-over destination as an empty regular file in the
# persistent state directory. The generated placeholder credential is mounted
# over it from the protected runtime directory and never persists here.
EMPTY_CREDENTIAL_MOUNTPOINTS: dict[str, tuple[str, ...]] = {
    "antigravity": ("antigravity-cli/antigravity-oauth-token",),
    "codex": ("auth.json",),
}


class StateBusyError(RuntimeError):
    """A session of an aisan that cannot share `state` is running in it."""


class StateLease:
    """One session's membership in a state directory shared on Darwin.

    Seatbelt has no private mount namespace, so `PinnedSymlink` entries are
    host symlinks in the state directory, common to every concurrent session.
    The sessions therefore stage identical links. A source that would be
    per-session lives in `links` instead, a directory below `private_root()`
    that the sessions share, and the last session to leave removes it.

    `exclusive` serializes staging and teardown between the sessions.
    """

    def __init__(self, links: Path, member_fd: int, mutex: Path) -> None:
        self.links = links
        self._member_fd = member_fd
        self._mutex = mutex

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Block until this session alone may change the shared links."""
        fd = os.open(self._mutex, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def alone(self) -> bool:
        """Return whether no other session is live. Call inside `exclusive`."""
        try:
            fcntl.flock(self._member_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            alone = False
        else:
            alone = True
        # A lock conversion may drop the held lock, even when it fails.
        fcntl.flock(self._member_fd, fcntl.LOCK_SH)
        return alone


@contextmanager
def share_state_dir(state: Path) -> Iterator[StateLease]:
    """Join the sessions using `state` for the duration of the context.

    The lock files and `StateLease.links` live inside `private_root()`, so a
    boxed process can neither fake nor end a membership.
    """
    root = prepare_private_dir(private_root())
    digest = hashlib.sha256(str(resolved_path(state)).encode()).hexdigest()[:16]
    links = root / f"links-{digest}"
    fd = os.open(
        root / f"state-{digest}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    lease = StateLease(links, fd, root / f"state-{digest}.mutex")
    try:
        with lease.exclusive():
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as e:
                # Sessions hold the lock shared; older releases held it
                # exclusively and repoint the links of any other session.
                raise StateBusyError(
                    f"a session of an older aisan is running with state"
                    f" directory {state}; exit it first"
                ) from e
            if lease.alone():
                # A killed session leaves its links behind.
                shutil.rmtree(links, ignore_errors=True)
            prepare_private_dir(links)
        try:
            yield lease
        finally:
            with lease.exclusive():
                if lease.alone():
                    shutil.rmtree(links, ignore_errors=True)
    finally:
        os.close(fd)


def is_runtime_symlink(path: Path, allowed_root: Path | None) -> bool:
    """Return whether `path` is a symlink targeting aisan's private runtime tree."""
    if allowed_root is None:
        return False
    try:
        target = Path(os.readlink(path))
    except OSError:
        return False
    if not target.is_absolute() or ".." in target.parts:
        return False
    if target.is_relative_to(allowed_root):
        return True
    try:
        return target.resolve().is_relative_to(allowed_root.resolve())
    except OSError:
        return False


def planted_credentials(
    state: Path,
    client: str,
    *,
    allowed_symlink_root: Path | None = None,
) -> list[Path]:
    """Find credentials left in client state by an earlier boxed session.

    The state directory is writable and persists across sessions for a
    repository. A login during a networked session could therefore expose its
    token to later sessions without adding a credential mount.

    ``lstat`` counts symlinks because an occupied credential path is sufficient.
    An empty regular bind-over mountpoint (or a stale Seatbelt symlink into
    aisan's private root) is the exception; it contains no credential and the
    runtime placeholder replaces it inside the box.
    """
    found: list[Path] = []
    for name in BOXED_CREDENTIALS.get(client, ()):
        path = state / name
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if name in EMPTY_CREDENTIAL_MOUNTPOINTS.get(client, ()):
            if stat.S_ISREG(info.st_mode) and info.st_size == 0:
                continue
            if stat.S_ISLNK(info.st_mode) and is_runtime_symlink(
                path, allowed_symlink_root
            ):
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


def ensure_sealed_dir(path: Path, *, mode: int = 0o700) -> None:
    """Make ``path`` a real directory, replacing a link or file planted there.

    ``write_sealed`` guards only the final component; a seed file below a
    subdirectory also needs that directory not to be a link out of the state.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=mode)
        return
    if not stat.S_ISDIR(info.st_mode):
        path.unlink()
        path.mkdir(mode=mode)


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
