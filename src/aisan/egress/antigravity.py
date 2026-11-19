# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Connect a boxed Antigravity CLI to Cloud Code through a host proxy.

The boxed client reads its login from a token file when no D-Bus session bus
is reachable, which is always the case in a box. That file holds a
placeholder, or the per-activation proxy token when the box shares the host
network; the client presents its ``access_token`` as a bearer and never tries
to refresh a token that is far from expiry. ``CLOUD_CODE_URL`` and
``USER_INFO_OVERRIDE_URL`` point the client at the loopback proxy.

``aisan.proxy.antigravity`` filters and forwards the requests. The host
credential is injected, so another build of the client can supply its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from ..proxy.antigravity import Profile, make_app
from ..proxy.gemini import BodyPolicy
from ..proxy.http import RateLimit, serve_tcp
from ..proxy.http import serve as serve_proxy
from ..sandbox import BindOver, BindSpec
from ..statedir import write_sealed
from ..upstream import DIRECT, Transport
from .base import (
    PLACEHOLDER_KEY,
    SHARED_PORT_MARKER,
    Backend,
    BackendActivation,
    PreflightError,
    shared_proxy_token,
)

log = logging.getLogger(__name__)

__all__ = [
    "CLOUD_CODE_URL_ENV",
    "PORT",
    "USER_INFO_URL_ENV",
    "AntigravityBackend",
    "Credential",
    "box_token_file",
]

PORT = 8716
CLOUD_CODE_URL_ENV = "CLOUD_CODE_URL"
USER_INFO_URL_ENV = "USER_INFO_OVERRIDE_URL"
_BOX_TOKEN_NAME = "antigravity-oauth-token"  # noqa: S105 -- a file name


class Credential(Protocol):
    """A host login the proxy presents upstream."""

    #: Host paths holding the login or its fallback store, for mount guards.
    files: tuple[Path, ...]

    async def token(self) -> str:
        """Return a currently valid access token."""
        ...

    async def check(self, backend: str) -> None:
        """Raise ``PreflightError`` if no valid token is available."""
        ...


def box_token_file(access_token: str) -> str:
    """Return the client's token file for a box.

    Antigravity CLI 1.5.11 accepts this shape as a consumer login. The expiry
    is far enough away that the client never redeems the refresh token.
    """
    return json.dumps(
        {
            "token": {
                "access_token": access_token,
                "token_type": "Bearer",
                "refresh_token": PLACEHOLDER_KEY,
                "expiry": "2099-01-01T00:00:00.000000000+00:00",
                "expires_in": 3599,
            },
            "auth_method": "consumer",
        }
    )


class AntigravityBackend(Backend):
    """Cloud Code through a host proxy on loopback ``PORT``."""

    name = "antigravity"
    port = PORT
    supports_shared_net = True

    def __init__(
        self,
        *,
        credential: Credential,
        profile: Profile | None = None,
        token_file: Path,
        rpm: int = 120,
        port: int = PORT,
        transport: Transport = DIRECT,
    ) -> None:
        """``token_file`` is the path at which the boxed client reads its login."""
        self._credential = credential
        self._profile = profile or Profile()
        self._token_file = token_file
        self.credentials = credential.files
        self.transport = transport
        self._rpm = rpm
        self.port = port

    def client_env(self) -> dict[str, str]:
        return self._client_env(self.port)

    def _client_env(self, port: int | str) -> dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {CLOUD_CODE_URL_ENV: base, USER_INFO_URL_ENV: f"{base}/userinfo"}

    def shared_client_env_description(self) -> dict[str, str]:
        return self._client_env(SHARED_PORT_MARKER)

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        return [BindOver(runtime_dir / _BOX_TOKEN_NAME, self._token_file)]

    def prepare(self, runtime_dir: Path) -> None:
        self._write_box_token(runtime_dir, PLACEHOLDER_KEY)

    def _write_box_token(self, runtime_dir: Path, access_token: str) -> None:
        write_sealed(runtime_dir / _BOX_TOKEN_NAME, box_token_file(access_token))

    async def preflight(self) -> None:
        """Check the upstream and the credential before starting the box.

        Check connectivity first because refreshing a login needs the network.
        """
        await self._check_upstream()
        await self._credential.check(self.name)

    async def _check_upstream(self) -> None:
        # Any HTTP response proves connectivity. Don't follow redirects.
        from aiohttp import ClientError, ClientTimeout

        upstream = self._profile.upstream
        fix = "check the network"
        try:
            async with (
                self.transport.session(ClientTimeout(total=10)) as session,
                session.get(upstream, allow_redirects=False),
            ):
                pass
        except ClientError as e:
            raise PreflightError(
                self.name, f"upstream {upstream} does not answer: {e}", fix
            ) from e
        except TimeoutError as e:
            raise PreflightError(
                self.name, f"upstream {upstream} timed out", fix
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        app = make_app(
            token=self._credential.token,
            client_token=PLACEHOLDER_KEY,
            profile=self._profile,
            rate=RateLimit(per_minute=self._rpm),
            transport=self.transport,
        )
        runner = await serve_proxy(sock, app)
        log.info("antigravity proxy: listening on %s for 127.0.0.1:%d", sock, self.port)
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)

    @asynccontextmanager
    async def serve_shared(
        self, runtime_dir: Path, *, isolated: bool = False
    ) -> AsyncIterator[BackendActivation]:
        client_token = shared_proxy_token()
        self._write_box_token(runtime_dir, client_token)
        body = BodyPolicy() if isolated else BodyPolicy.for_shared_network()
        app = make_app(
            token=self._credential.token,
            client_token=client_token,
            profile=self._profile,
            body=body,
            rate=RateLimit(per_minute=self._rpm),
            transport=self.transport,
        )
        runner, port = await serve_tcp(app)
        log.info("antigravity proxy: listening on 127.0.0.1:%d", port)
        try:
            yield BackendActivation(port, self._client_env(port))
        finally:
            await runner.cleanup()
