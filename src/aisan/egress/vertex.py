# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Vertex backend: the box's only model route, and no credential with it.

`aisan.proxy` owns the mechanism (allowlist, auth injection, forwarding);
this owns the backend -- which project and models are allowed, which principal
the token is minted as, and what the box has to read to find the endpoint.

The credential never enters the box: it is minted here, host-side, by
`mint.vertex_fetch`, and attached to a request the box never sees.
"""

from __future__ import annotations

import logging
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

# Re-mint this long before expiry. Tokens live ~1h and a single model call is
# minutes at worst, so this is generous; the point is that a long job never
# reaches for one that has just expired.
_REFRESH_MARGIN_S = 300

PORT = 8711


class _Token:
    """Mints on demand and caches until nearly expired.

    Read per request by the proxy rather than captured, because jobs run for
    hours and tokens live ~1h -- a value fetched once would strand a long job
    mid-flight. Refreshing lazily (rather than on a timer) keeps this a plain
    object with no task to leak.
    """

    def __init__(self, fetch) -> None:
        self._fetch = fetch
        self._token = ""
        self._expiry = datetime.fromtimestamp(0, UTC)

    async def __call__(self) -> str:
        now = datetime.now(UTC)
        if not self._token or (self._expiry - now).total_seconds() < _REFRESH_MARGIN_S:
            self._token, self._expiry = await self._fetch()
        return self._token


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
        impersonate: str = "",
        fetch=None,
    ) -> None:
        self._rpm = rpm
        self._project = project
        self._location = location
        self._models = models
        self._impersonate = impersonate
        # A test may inject `fetch` in place of the real mint -- otherwise
        # exercising this backend would require a real cloud credential, which is
        # exactly the dependency the proxy exists to keep out of the loop.
        self._injected = fetch
        self._cached: _Token | None = None

    @property
    def _token(self) -> _Token:
        """The mint, built on first use and then cached.

        Lazy for one reason: constructing the real provider validates the
        impersonation target, and a backend that raises in its constructor
        cannot be DESCRIBED -- `explain` would have to build a different object
        than the configured one, which is the exact fidelity the inspector
        exists to preserve. Nothing is lost: `preflight` runs before bwrap, so a
        missing target still fails early, and now with a fix command attached.

        Cached rather than rebuilt so preflight's mint is the same one the first
        request reads, instead of costing an extra IAM round trip per box.
        """
        if self._cached is None:
            self._cached = _Token(self._injected or vertex_fetch(self._impersonate))
        return self._cached

    def client_env(self) -> dict[str, str]:
        return {"AISAN_VERTEX_PROXY_ENDPOINT": f"http://127.0.0.1:{self.port}"}

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
        # A leftover socket from a killed box makes bind() fail EADDRINUSE
        # forever; the dir name is derived from this box, so nothing else owns
        # this path.
        sock.unlink(missing_ok=True)
        app = make_app(
            allowlist=Allowlist(
                project=self._project, location=self._location, models=self._models
            ),
            token=self._token,
            location=self._location,
            rate=RateLimit(per_minute=self._rpm),
        )
        runner = await serve_proxy(sock, app)
        log.info("vertex proxy: listening on %s for 127.0.0.1:%d", sock, self.port)
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)
