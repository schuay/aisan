# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Configure Vertex model access without giving the box a credential.

``aisan.proxy.vertex`` filters and forwards requests. This backend selects the
project, models, impersonated principal, and box-facing endpoint.

``mint.vertex_fetch`` creates short-lived tokens on the host. The proxy adds them
after receiving a request from the box.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..mint import vertex_fetch
from ..proxy.http import RateLimit
from ..proxy.http import serve as serve_proxy
from ..proxy.vertex import Allowlist, make_app
from .base import Backend, PreflightError

log = logging.getLogger(__name__)

# Refresh one-hour tokens five minutes before expiry.
_REFRESH_MARGIN_S = 300

PORT = 8711


class _Token:
    """Mint on demand and cache the token until close to expiry.

    Jobs can outlive one-hour tokens, so the proxy reads this provider for every
    request. Lazy refresh doesn't require a background task.
    """

    def __init__(self, fetch) -> None:
        self._fetch = fetch
        self._token = ""
        self._expiry = datetime.fromtimestamp(0, UTC)
        # Serialize refreshes and let waiters reuse the new token.
        self._lock = asyncio.Lock()

    def _fresh(self, now: datetime) -> bool:
        return bool(self._token) and (self._expiry - now).total_seconds() >= (
            _REFRESH_MARGIN_S
        )

    async def __call__(self) -> str:
        if self._fresh(datetime.now(UTC)):
            return self._token
        async with self._lock:
            if self._fresh(datetime.now(UTC)):
                return self._token
            self._token, self._expiry = await self._fetch()
        return self._token


def default_credentials() -> tuple[Path, ...]:
    """Return every file Application Default Credentials may read.

    ``google.auth.default()`` uses ``GOOGLE_APPLICATION_CREDENTIALS`` when set
    and otherwise checks the gcloud file under ``CLOUDSDK_CONFIG``. Protect both
    because a process in the box could redirect ADC to either path.
    """
    paths = []
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        paths.append(Path(explicit))
    config = os.environ.get("CLOUDSDK_CONFIG")
    root = Path(config) if config else Path.home() / ".config" / "gcloud"
    paths.append(root / "application_default_credentials.json")
    return tuple(dict.fromkeys(paths))


class VertexBackend(Backend):
    """Model calls through a host-side proxy on loopback `PORT`."""

    name = "vertex"
    port = PORT

    def __init__(
        self,
        *,
        rpm: int,
        project: str,
        location: str,
        models: tuple[str, ...],
        anthropic_models: tuple[str, ...] = (),
        session_header: str | None = None,
        impersonate: str = "",
        fetch=None,
    ) -> None:
        self._rpm = rpm
        self._project = project
        self._location = location
        self._models = models
        # Keep Claude model names on their separate Anthropic route.
        self._anthropic_models = anthropic_models
        # The deployment supplies the session-affinity header name; the proxy
        # sends it on every route.
        self._session_header = session_header
        self._impersonate = impersonate
        # A box that reads the ADC source file could mint its own tokens.
        self.credentials = default_credentials()
        # Tests can replace the real cloud credential provider.
        self._injected = fetch
        self._cached: _Token | None = None

    @property
    def _token(self) -> _Token:
        """Build the token provider on first use and then cache it.

        Lazy construction lets ``explain`` describe invalid configuration.
        ``preflight`` still validates the target before bubblewrap starts.

        Reusing the provider also avoids an extra IAM request after preflight.
        """
        if self._cached is None:
            self._cached = _Token(self._injected or vertex_fetch(self._impersonate))
        return self._cached

    def client_env(self) -> dict[str, str]:
        return {
            "AISAN_VERTEX_PROXY_ENDPOINT": f"http://127.0.0.1:{self.port}",
            # The Anthropic SDK takes ``/v1`` from its base URL. Always point it
            # at the proxy so unsupported routes receive a clear refusal.
            "ANTHROPIC_VERTEX_BASE_URL": f"http://127.0.0.1:{self.port}/v1",
        }

    async def preflight(self) -> None:
        try:
            await self._token()
        except Exception as e:
            raise PreflightError(
                self.name,
                f"cannot mint a Vertex token as {self._impersonate or '(no target)'}:"
                f" {e}",
                "gcloud auth application-default login",
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        # Remove a socket left by an earlier crashed run of this box.
        sock.unlink(missing_ok=True)
        app = make_app(
            allowlist=Allowlist(
                project=self._project,
                location=self._location,
                models=self._models,
                anthropic_models=self._anthropic_models,
            ),
            token=self._token,
            location=self._location,
            rate=RateLimit(per_minute=self._rpm),
            session_header=self._session_header,
        )
        runner = await serve_proxy(sock, app)
        log.info("vertex proxy: listening on %s for 127.0.0.1:%d", sock, self.port)
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)
