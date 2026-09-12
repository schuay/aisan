# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Define the interface between boxes and credential-aware backends.

A backend provides one route out of the box. Isolated mode serves a host UNIX
socket through an in-box relay. Supported shared-network backends instead serve
authenticated TCP on host loopback and issue a per-box port and token. Upstream
credentials remain in the host process.

Each backend defines its box-facing port and environment, preflight credential
check, host service, and credential paths that mount policy must hide. Backends
may also report a mint failure for structured fallback decisions. This avoids
matching third-party error text such as "code = Unauthenticated."
"""

from __future__ import annotations

import abc
import re
import secrets
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ..sandbox import BindSpec, paths_overlap


class PreflightError(RuntimeError):
    """A credential failure with the command needed to fix it."""

    def __init__(self, backend: str, reason: str, fix: str) -> None:
        super().__init__(f"{backend}: {reason}\n  fix: {fix}")
        self.backend = backend
        self.reason = reason
        self.fix = fix


# A self-describing placeholder that stops clients from searching for real
# credentials. Each backend chooses the appropriate environment variable.
#
# Shared mode keeps the last 20 characters as a Claude-compatible suffix on a
# random token. The proxy compares the full value.
PLACEHOLDER_KEY = "aisan-placeholder-not-a-credential"
SHARED_PORT_MARKER = "(assigned at launch)"
SHARED_TOKEN_MARKER = "<per-box proxy token>"  # noqa: S105 - descriptive marker
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

# Linux allows 107 path bytes plus the terminating NUL in ``sun_path``.
_SUN_PATH_BYTES = 107


@dataclass(frozen=True)
class BackendActivation:
    """A backend endpoint and client environment for one box activation."""

    port: int
    client_env: dict[str, str]


def shared_proxy_token() -> str:
    """Create a proxy token with Claude Code's accepted key suffix."""
    return secrets.token_urlsafe(32) + PLACEHOLDER_KEY[-20:]


class Backend(abc.ABC):
    """One host-side egress service and its box-facing configuration."""

    #: Stable identifier used in the socket filename and manifest.
    name: str
    #: The in-box loopback port. Fixed per backend; see the module docstring.
    port: int
    #: Host paths containing this backend's credentials. Box mount checks keep
    #: every path unreadable inside the sandbox. Empty for in-memory credentials.
    #:
    #: A store may contain credentials for other routes, as OpenCode auth.json
    #: does, so checks protect the complete path.
    credentials: tuple[Path, ...] = ()
    #: Whether this backend implements the authenticated host-loopback path.
    supports_shared_net: bool = False

    def socket_path(self, runtime_dir: Path) -> Path:
        """Return this backend's socket path in the runtime directory.

        Deriving the filename from the validated backend name keeps the service
        and manifest aligned. Check the full encoded path against Linux's UNIX
        socket limit after combining the configurable root and backend name.
        """
        if not isinstance(self.name, str) or _NAME_RE.fullmatch(self.name) is None:
            raise ValueError(
                f"backend name {self.name!r} is not a safe socket-file component"
            )
        path = runtime_dir / f"{self.name}.sock"
        # The configurable private root and backend name are bounded separately,
        # so check their combined byte length here.
        if len(str(path).encode()) > _SUN_PATH_BYTES:
            raise ValueError(
                f"socket path is too long for AF_UNIX "
                f"({len(str(path).encode())} bytes, limit {_SUN_PATH_BYTES}): "
                f"{path}. Shorten the private root or the backend name."
            )
        return path

    def client_env(self) -> dict[str, str]:
        """Return client environment variables for the isolated relay.

        The launcher configures relays from the manifest. These variables use the
        client application's vocabulary to point it at the fixed loopback port.
        """
        return {}

    def shared_client_env_description(self) -> dict[str, str]:
        """Return stable markers for shared-network ``explain`` output."""
        raise NotImplementedError(f"{self.name} does not support shared networking")

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        """Return extra mounts needed by the boxed client."""
        return []

    def prepare(self, runtime_dir: Path) -> None:
        """Create bind sources before bubblewrap resolves the mount list."""

    async def preflight(self) -> None:
        """Raise ``PreflightError`` if credentials aren't available now.

        Backends that mint credentials override this so interactive fixes remain
        possible before the box takes over the terminal.
        """

    @property
    def refused(self) -> Exception | None:
        """Return a mint failure recorded during this box, if any."""
        return None

    @abc.abstractmethod
    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        """Run the isolated host service for the duration of the context.

        Fail startup if the backend can't listen, since every dependent request
        would otherwise fail later.
        """
        raise NotImplementedError
        yield  # pragma: no cover - keep the abstract method an async generator

    @asynccontextmanager
    async def serve_shared(self, runtime_dir: Path) -> AsyncIterator[BackendActivation]:
        """Run an authenticated host-loopback service for one activation."""
        raise NotImplementedError(f"{self.name} does not support shared networking")
        yield  # pragma: no cover - keep the method an async generator


@dataclass(frozen=True)
class EgressProfile:
    """Describe the resources contributed by one named egress profile.

    Backends provide credential-aware host services. ``binds`` supports direct
    routes that expose a credential mount instead. ``notice`` explains boundary
    changes when the launcher applies the profile.
    """

    backends: tuple[Backend, ...] = ()
    binds: tuple[BindSpec, ...] = ()
    notice: str = ""

    def __bool__(self) -> bool:
        return bool(self.backends or self.binds)


def credential_exposure(
    binds: tuple[BindSpec, ...], backends: tuple[Backend, ...]
) -> tuple[Path, Backend, Path] | None:
    """Find a bind source that overlaps a backend credential path.

    Ancestor and descendant mounts both count as exposure. This check rejects
    explicit user bind entries even when a later mount would hide them; the
    resolved sandbox performs a separate final visibility check.
    """
    for spec in binds:
        src = getattr(spec, "path", None) or getattr(spec, "src", None)
        if src is None:
            continue
        for backend in backends:
            hit = credential_overlap((src,), backend.credentials)
            if hit is not None:
                return src, backend, hit[1]
    return None


def credential_overlap(
    sources: Iterable[Path], credentials: Iterable[Path]
) -> tuple[Path, Path] | None:
    """Find the first overlapping source and credential paths."""
    for src in sources:
        for cred in credentials:
            if paths_overlap(src, cred):
                return src, cred
    return None
