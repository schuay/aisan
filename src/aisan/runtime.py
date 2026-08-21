# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The per-box runtime directory: sockets, backend files, and launcher control.

One directory per box, holding one socket per backend plus the small files the
box needs in order to dial them, plus `relays.json` -- the manifest the in-box
launcher reads to know what to listen on. The box binds the DIRECTORY, so every
file in it is reachable inside at the same absolute path the host wrote it at.
Isolated mode carries `relays.json`; shared-network mode carries a protected
`client-env.json` whose live port and token the launcher injects before exec.

The directory is keyed by an opaque `box_id`. Opaque is the point: aisan never
parses it, never derives a job id or a worktree out of it, and two boxes with
different ids are simply different boxes. A consumer that wants its own notion
of identity in there passes its own string.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .sandbox import RO, Bind

# Names the directory as ours, so an operator seeing it in the system temp dir
# knows what left it there and a stale one is safe to remove.
_DIR_PREFIX = "aisan-proxy-"

# The manifest file name. Read by the in-box launcher, written by the Box.
MANIFEST_NAME = "relays.json"
CLIENT_ENV_NAME = "client-env.json"


def runtime_dir(box_id: str) -> Path:
    """This box's directory, under the system temp dir, named by a digest.

    The digest is not obfuscation, it is a length bound. A UNIX socket path is
    capped at 108 bytes (sizeof sun_path), and the natural path -- an operator's
    control root plus a job id plus "vertex.sock" -- once reached 129 bytes, so
    bind() died "AF_UNIX path too long" and took the job with
    it. Both terms were unbounded and neither was ours to shorten. Hashing makes
    the length independent of the caller's naming entirely: /tmp/aisan-proxy-XXXX
    is 26 bytes and leaves ~80 for the file name on any sane system.

    Truncated to 16 hex chars (64 bits). A collision means two live boxes share
    a socket directory, which needs a birthday pair among boxes alive at the
    same moment -- a population in the tens, not the billions.
    """
    digest = hashlib.sha256(box_id.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"{_DIR_PREFIX}{digest}"


def prepare_runtime_dir(box_id: str) -> Path:
    """Create the directory, private to this user, and return it.

    0o700 because on a shared host the system temp dir is world-writable and
    these sockets are unauthenticated capabilities: the host halves behind them
    hold the credentials, so anyone who can connect gets model calls and RBE
    calls on somebody else's identity.
    """
    d = runtime_dir(box_id)
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = d.lstat()
    except OSError as e:
        raise RuntimeError(f"cannot inspect runtime directory {d}: {e}") from e
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"runtime path is not a directory: {d}")
    if info.st_uid != os.getuid():
        raise PermissionError(f"runtime directory is not owned by this user: {d}")
    if info.st_mode & 0o077:
        raise PermissionError(f"runtime directory is too permissive: {d}")
    return d


def cleanup_runtime_dir(box_id: str) -> None:
    """Remove the directory once its last file is gone.

    Best-effort and only when empty: the directory is ours alone (its name is
    this box's digest), but a non-empty one means something we do not own is in
    there, and deleting a stranger's file is worse than leaving litter.
    """
    with contextlib.suppress(OSError):
        runtime_dir(box_id).rmdir()


def runtime_bind(box_id: str) -> Bind:
    """The ro bind that makes this box's sockets reachable from inside it.

    A preset puts this in its spec, so the spec stays the complete mount policy
    and `Box` augments nothing a reader of `explain` would not see. ro is
    enough: connect() needs no write bit on a socket file, and the box has no
    business unlinking the host's socket. Optional because a box that is only
    being explained has no live runtime dir, and refusing to describe a profile
    for want of a directory nobody has created yet helps no one.
    """
    return Bind(runtime_dir(box_id), RO, optional=True)


def write_manifest(directory: Path, entries: list[dict[str, object]]) -> Path:
    """Write `relays.json`, the in-box launcher's whole input.

    A file rather than environment variables or argv, for the reason the whole
    design puts the wiring here: N backends means N sockets and N ports, and
    every alternative encoding either fixes N at the shape of the command line
    or reinvents a list format in a string. The launcher reads one file and
    knows everything; adding a backend touches no launcher code and no argv.

    Written into the runtime dir, which is already bound into the box and
    already 0o700, so it needs no mount of its own and grants no reader who
    could not already open the sockets it names.
    """
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(entries, indent=2) + "\n")
    return path


def read_manifest(directory: Path) -> list[dict[str, object]]:
    """Read `relays.json` from inside the box."""
    return json.loads((directory / MANIFEST_NAME).read_text())


def write_client_env(directory: Path, env: dict[str, str]) -> Path:
    """Write shared-mode client environment inside the private runtime dir."""
    path = directory / CLIENT_ENV_NAME
    path.write_text(json.dumps(env, sort_keys=True) + "\n")
    path.chmod(0o600)
    return path


def read_client_env(directory: Path) -> dict[str, str]:
    """Read the shared-mode environment immediately before payload exec."""
    data = json.loads((directory / CLIENT_ENV_NAME).read_text())
    if not isinstance(data, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError("client environment must be a string mapping")
    return data
