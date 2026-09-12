# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Manage per-box sockets, backend files, and launcher control files.

Each box gets a directory with one socket per backend and the files needed to
connect to them. The box binds the entire directory at the same absolute path.
Isolated mode uses ``relays.json``; shared-network mode uses a protected
``client-env.json`` containing the port and token to inject before execution.

An opaque ``box_id`` identifies the directory. Aisan doesn't parse the ID or
derive other identifiers from it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from .private import prepare_private_dir, private_root
from .sandbox import RO, Bind

# Identify aisan directories within the private root.
_DIR_PREFIX = "proxy-"

# The Box writes this manifest for the in-box launcher.
MANIFEST_NAME = "relays.json"
CLIENT_ENV_NAME = "client-env.json"


def runtime_dir(box_id: str) -> Path:
    """Return the box's runtime directory, named with a digest of its ID.

    The digest bounds path length; it doesn't conceal the ID. Linux limits Unix
    socket paths to 108 bytes, while caller-provided IDs are unbounded. A path
    such as ``/tmp/aisan-UID/proxy-XXXX`` leaves room for socket file names.

    Sixteen hexadecimal characters provide 64 bits. A harmful collision
    requires two boxes with the same digest to be live at once.
    """
    digest = hashlib.sha256(box_id.encode()).hexdigest()[:16]
    return private_root() / f"{_DIR_PREFIX}{digest}"


def prepare_runtime_dir(box_id: str) -> Path:
    """Create the directory, private to this user, and return it.

    Both the directory and its parent use mode 0o700. They live below the shared
    system temp directory, and their unauthenticated sockets provide access to
    host processes that hold credentials.
    """
    d = prepare_private_dir(runtime_dir(box_id))
    # A killed shared-network run may leave client-env.json behind. Because the
    # launcher checks it before the manifest, the next isolated run would use a
    # dead port instead of starting relays. Staging recreates both control files.
    for name in (MANIFEST_NAME, CLIENT_ENV_NAME):
        (d / name).unlink(missing_ok=True)
    return d


def cleanup_runtime_dir(box_id: str) -> None:
    """Remove the directory once its last file is gone.

    Removal is best-effort and succeeds only when the directory is empty. An
    unexpected file prevents removal and remains intact.
    """
    with contextlib.suppress(OSError):
        runtime_dir(box_id).rmdir()


def runtime_bind(box_id: str) -> Bind:
    """Return the read-only bind that exposes runtime sockets to the box.

    Presets include this bind so the spec describes the complete mount policy.
    Connecting to a socket doesn't require write access to its file. The bind is
    optional because explaining a box doesn't create its runtime directory.
    """
    return Bind(runtime_dir(box_id), RO, optional=True)


def write_manifest(directory: Path, entries: list[dict[str, object]]) -> Path:
    """Write the complete relay configuration for the in-box launcher.

    A file represents any number of socket and port pairs without encoding a
    list in environment variables or command-line arguments.

    The box already mounts the private runtime directory, so the manifest
    doesn't need an additional mount. Access to the directory also grants access
    to its relay sockets.
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
