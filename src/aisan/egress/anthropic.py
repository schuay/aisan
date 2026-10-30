# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Connect a boxed Anthropic Messages API client without exposing the host
credential.

``aisan.proxy.anthropic`` filters and forwards requests. This module selects
the upstream and the credential, and configures the boxed client through
``ANTHROPIC_BASE_URL`` and a placeholder in the credential's own variable.

Two credential sources, kept on one object so upstream authentication and the
box's configuration can't diverge: the Claude Code subscription login
(``claude_plan``), which refreshes through host Claude and dresses the box with
``CLAUDE_CODE_OAUTH_TOKEN``; and a static API key held in memory, which sends
``x-api-key`` upstream and dresses the box with ``ANTHROPIC_API_KEY``. The
variable must match the credential type because Claude Code sends different
``anthropic-beta`` values for subscription and API-key authentication. An SDK
client (the ``anthropic`` package and what is built on it) needs only the two
variables and is served by either source.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ..proxy.anthropic import BodyPolicy, make_app
from ..proxy.http import RateLimit, serve_tcp
from ..proxy.http import serve as serve_proxy
from .base import (
    PLACEHOLDER_KEY,
    SHARED_PORT_MARKER,
    SHARED_TOKEN_MARKER,
    Backend,
    BackendActivation,
    PreflightError,
    shared_proxy_token,
)
from .claude_plan import (
    OAUTH_TOKEN_ENV,
    SUBSCRIPTION_ENV,
    PlanCredential,
    claude_config_dir,
    default_credentials,
)

log = logging.getLogger(__name__)

PORT = 8713

# The plan credential's names stay importable from here: the Claude launcher,
# the session MCP importer and the tests reached them through this module
# before the credential had a module of its own.
__all__ = [
    "API_KEY_ENV",
    "DEFAULT_UPSTREAM",
    "OAUTH_TOKEN_ENV",
    "PLACEHOLDER_KEY",
    "PORT",
    "SUBSCRIPTION_ENV",
    "AnthropicBackend",
    "claude_config_dir",
    "default_credentials",
]


# Callers can replace the standard endpoint with a compatible upstream.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

_FIX_NO_KEY = "pass a non-empty key, or drop --api-key to use the plan login"

API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class _ApiKeyCredential:
    """Hold a static Anthropic API key on the host.

    No credential path needs protection because the caller supplies the key in
    memory. The box-facing behavior is covered by client tests, and the proxy is
    tested against a fake upstream. Sending a real API key upstream hasn't been
    tested because none was available.
    """

    key: str

    files: tuple[Path, ...] = ()

    async def upstream(self) -> dict[str, str]:
        return {"x-api-key": self.key}

    def dress(self, token: str) -> dict[str, str]:
        return {API_KEY_ENV: token}

    async def check(self, backend: str) -> None:
        if not self.key:
            raise PreflightError(
                backend, "the API key given to this backend is empty", _FIX_NO_KEY
            )


# Keep upstream authentication and box configuration on the same credential
# object so their types can't diverge.
#
# Claude Code 2.1.246 sends ``authorization: Bearer`` and two extra beta flags
# for subscriptions, but sends ``x-api-key`` without those flags for API keys.
# The proxy forwards the beta flags, so its credential type must match the box.
_Credential = PlanCredential | _ApiKeyCredential


class AnthropicBackend(Backend):
    """Route model calls through a host proxy on loopback ``PORT``."""

    name = "anthropic"
    port = PORT
    supports_shared_net = True

    def __init__(
        self,
        *,
        credentials: Path | None = None,
        api_key: str | None = None,
        upstream: str = DEFAULT_UPSTREAM,
        claude_command: tuple[str, ...] = ("claude",),
        rpm: int = 120,
        models: tuple[str, ...] = (),
    ) -> None:
        """Use the subscription credential unless an API key is supplied.

        The box must be configured for the selected credential type because the
        client sends different beta flags for subscriptions and API keys. Reject
        both arguments together instead of giving one implicit precedence.

        ``models`` restricts requests to the named model ids; empty leaves the
        choice to the box, which is what an interactive session wants.
        """
        if api_key is not None and credentials is not None:
            raise ValueError("provide at most one of credentials or api_key")
        self._credential: _Credential = (
            _ApiKeyCredential(api_key)
            if api_key is not None
            else PlanCredential(
                credentials or default_credentials(), claude_command=claude_command
            )
        )
        # Box mount checks must keep these host credential files hidden.
        self.credentials = self._credential.files
        self._upstream = upstream
        self._rpm = rpm
        self._models = frozenset(models) or None

    def client_env(self) -> dict[str, str]:
        """Return the relay location and placeholder credential for the box.

        The base URL routes requests to the relay. The placeholder stops Claude
        Code from searching for a real credential and trying an interactive
        login. This method stays usable by ``explain`` when the host isn't logged
        in.
        """
        return self._client_env(self.port, PLACEHOLDER_KEY)

    def _client_env(self, port: int | str, token: str) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            **self._credential.dress(token),
        }

    def shared_client_env_description(self) -> dict[str, str]:
        return self._client_env(SHARED_PORT_MARKER, SHARED_TOKEN_MARKER)

    async def _upstream_auth(self) -> dict[str, str]:
        """Read authentication headers for one upstream request.

        Host Claude may replace the credential during a long session, so reread
        it instead of caching the preflight value.
        """
        return await self._credential.upstream()

    async def preflight(self) -> None:
        """Check the upstream and credential before starting the box.

        Check connectivity first because refreshing an expiring OAuth token also
        needs the network. The credential implementation then reports missing,
        invalid, or stale credentials with the appropriate login command.
        """
        await self._check_upstream()
        await self._credential.check(self.name)

    async def _check_upstream(self) -> None:
        """Check whether the configured upstream accepts a connection.

        Any HTTP response proves connectivity, including a 4xx or redirect. Don't
        follow redirects or spend model quota during this check.
        """
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(
                    f"{self._upstream.rstrip('/')}/api/hello",
                    allow_redirects=False,
                ),
            ):
                pass
        except ClientError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} does not answer: {e}",
                f"start the service at {self._upstream}, or point the backend"
                " at a different upstream",
            ) from e
        except TimeoutError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} timed out",
                f"start the service at {self._upstream}, or point the backend"
                " at a different upstream",
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        # ``serve_proxy`` removes stale sockets before binding.
        app = make_app(
            authorization=self._upstream_auth,
            upstream=self._upstream,
            body=BodyPolicy(models=self._models),
            rate=RateLimit(per_minute=self._rpm),
        )
        runner = await serve_proxy(sock, app)
        log.info("anthropic proxy: listening on %s for 127.0.0.1:%d", sock, self.port)
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)

    @asynccontextmanager
    async def serve_shared(self, runtime_dir: Path) -> AsyncIterator[BackendActivation]:
        client_token = shared_proxy_token()
        # ``Box`` calls this method only when the spec shares the host network.
        # Isolated serving above always uses the strict body policy.
        app = make_app(
            authorization=self._upstream_auth,
            upstream=self._upstream,
            body=BodyPolicy.for_shared_network(models=self._models),
            rate=RateLimit(per_minute=self._rpm),
            client_token=client_token,
        )
        runner, port = await serve_tcp(app)
        log.info("anthropic proxy: listening on 127.0.0.1:%d", port)
        try:
            yield BackendActivation(port, self._client_env(port, client_token))
        finally:
            await runner.cleanup()
