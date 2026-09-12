# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Configure OpenAI-compatible providers for boxed OpenCode sessions.

``aisan.proxy.openai_compat`` filters and forwards the common wire protocol.
This backend supplies provider-specific data: the upstream URL and static API
key. OpenCode's models.dev catalog currently describes about 147 compatible
providers, so supporting another provider usually requires only its catalog ID.

The backend reads host OpenCode's ``auth.json`` but never writes it. It accepts
static API keys only. OAuth and well-known entries require refresh behavior that
could race host OpenCode while both processes update an unlocked file. The key
is reread for every request so a reconnect takes effect during a session.

The box receives an in-memory placeholder through ``OPENCODE_AUTH_CONTENT``.
Measurements showed that OpenCode doesn't read ``auth.json`` when this variable
is present. The proxy discards the placeholder bearer and adds the real key.

``OPENCODE_CONFIG_CONTENT`` points the provider at the relay and disables session
sharing. Inline configuration outranks repository configuration, preventing the
agent from changing the route or enabling transcript uploads. Network isolation
still enforces containment if the client tries another route.

The preset mounts the host's fetched model catalog because OpenCode's embedded
catalog can lag behind it. The preset owns that filesystem policy; this backend
only reads provider data.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ..proxy.http import RateLimit, serve_tcp
from ..proxy.http import serve as serve_proxy
from ..proxy.openai_compat import make_app
from .base import (
    PLACEHOLDER_KEY,
    SHARED_PORT_MARKER,
    SHARED_TOKEN_MARKER,
    Backend,
    BackendActivation,
    PreflightError,
    shared_proxy_token,
)

log = logging.getLogger(__name__)

PORT = 8714


# Honor the host's XDG paths for both OpenCode files.
def _xdg(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


def default_credentials() -> Path:
    return (
        _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share")
        / "opencode"
        / "auth.json"
    )


DEFAULT_CREDENTIALS = default_credentials()
DEFAULT_CATALOG = (
    _xdg("XDG_CACHE_HOME", Path.home() / ".cache") / "opencode" / "models.json"
)

_FIX_LOGIN = "opencode providers login   (on the HOST -- aisan never writes this file)"
# A subscription login can't supply the static key this backend requires.
_FIX_SHAPE = (
    "connect the provider with an API key (opencode /connect on the HOST), or"
    ' point the backend at credentials whose entry has "type": "api"'
)
_FIX_CATALOG = (
    "run opencode once on the HOST (it fetches the provider catalog), or pass"
    " upstream= explicitly"
)


class OpenAICompatBackend(Backend):
    """Route one OpenAI-compatible provider through the host proxy."""

    #: Default in-box loopback port. Override it when composing two providers.
    port = PORT
    supports_shared_net = True

    def __init__(
        self,
        *,
        provider: str,
        model: str = "",
        upstream: str | None = None,
        credentials: Path | None = None,
        catalog: Path | None = None,
        rpm: int = 120,
        port: int = PORT,
    ) -> None:
        self._provider = provider
        self._model = model
        # The complete API base includes any provider path prefix. Resolve a
        # missing value from OpenCode's catalog.
        self._upstream = upstream
        self._credentials = credentials or default_credentials()
        # The credential file contains keys for every configured provider.
        self.credentials = (self._credentials,)
        self._catalog = catalog or DEFAULT_CATALOG
        self._rpm = rpm
        # Give each provider a distinct socket name.
        self.name = f"openai-{provider}"
        self.port = port
        self._api: str | None = None

    # Host-side route configuration.

    def _resolve_api(self) -> str:
        """Return the provider's API base from the constructor or catalog.

        Cache the public catalog result so a file change can't reroute an active
        session. Credentials follow a separate per-request path.
        """
        if self._api is None:
            self._api = self._upstream or _catalog_api(self._catalog, self._provider)
        return self._api

    async def _bearer(self) -> str:
        """Read the provider's API key for one request.

        Host OpenCode can replace ``auth.json`` during /connect, so don't retain
        the preflight value.
        """
        return _read_api_key(self._credentials, self._provider)

    async def preflight(self) -> None:
        """Validate the route, credential, and upstream before launch.

        Route failures need a catalog or explicit URL. Missing keys need a host
        login, while unsupported credential shapes need a static API key.
        Connectivity failures require a working upstream.
        """
        try:
            api = self._resolve_api()
        except (OSError, ValueError, KeyError) as e:
            raise PreflightError(
                self.name, f"cannot resolve the route: {e}", _FIX_CATALOG
            ) from e

        try:
            _read_api_key(self._credentials, self._provider)
        except CredentialShapeError as e:
            raise PreflightError(self.name, str(e), _FIX_SHAPE) from e
        except (OSError, ValueError, KeyError) as e:
            raise PreflightError(
                self.name,
                f"no usable API key for {self._provider} in {self._credentials}: {e}",
                _FIX_LOGIN,
            ) from e

        await self._check_upstream(api)

    async def _check_upstream(self, api: str) -> None:
        """Check upstream connectivity without making a model call.

        Query ``/models`` without a key. Any HTTP response proves the connection,
        including a 4xx or redirect. Redirects aren't followed.
        """
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(f"{api.rstrip('/')}/models", allow_redirects=False),
            ):
                pass
        except ClientError as e:
            raise PreflightError(
                self.name,
                f"upstream {api} does not answer: {e}",
                "check the network, or point the backend at a different upstream",
            ) from e
        except TimeoutError as e:
            raise PreflightError(
                self.name,
                f"upstream {api} timed out",
                "check the network, or point the backend at a different upstream",
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        # The upstream API base restores the provider path prefix omitted from
        # the box-facing loopback URL. ``serve_proxy`` removes stale sockets.
        api = self._resolve_api()
        app = make_app(
            token=self._bearer,
            upstream=api,
            rate=RateLimit(per_minute=self._rpm),
        )
        runner = await serve_proxy(sock, app)
        log.info(
            "openai-compat proxy (%s): listening on %s for 127.0.0.1:%d",
            self._provider,
            sock,
            self.port,
        )
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)

    @asynccontextmanager
    async def serve_shared(self, runtime_dir: Path) -> AsyncIterator[BackendActivation]:
        client_token = shared_proxy_token()
        app = make_app(
            token=self._bearer,
            upstream=self._resolve_api(),
            rate=RateLimit(per_minute=self._rpm),
            client_token=client_token,
        )
        runner, port = await serve_tcp(app)
        log.info(
            "openai-compat proxy (%s): listening on 127.0.0.1:%d",
            self._provider,
            port,
        )
        try:
            yield BackendActivation(port, self._client_env(port, client_token))
        finally:
            await runner.cleanup()

    # Box-side client configuration.

    def client_env(self) -> dict[str, str]:
        """Return in-memory OpenCode configuration for the relay.

        This method doesn't read host files, so specs and dry runs work without a
        catalog. The proxy restores the provider's path prefix on the host side.
        If provided, the bare model ID is combined with the provider here because
        the OpenCode 1.18.18 TUI had no ``--model`` flag.
        """
        return self._client_env(self.port, PLACEHOLDER_KEY)

    def _client_env(self, port: int | str, key: str) -> dict[str, str]:
        auth = {self._provider: {"type": "api", "key": key}}
        config: dict[str, object] = {
            # Prevent project config from enabling transcript uploads.
            "share": "disabled",
            "provider": {
                self._provider: {"options": {"baseURL": f"http://127.0.0.1:{port}"}}
            },
        }
        if self._model:
            config["model"] = f"{self._provider}/{self._model}"
        return {
            "OPENCODE_AUTH_CONTENT": json.dumps(auth),
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        }

    def shared_client_env_description(self) -> dict[str, str]:
        return self._client_env(SHARED_PORT_MARKER, SHARED_TOKEN_MARKER)


class CredentialShapeError(Exception):
    """An auth entry with a type this backend can't use."""


def _read_api_key(path: Path, provider: str) -> str:
    """Read a provider's static API key from OpenCode ``auth.json``.

    Errors identify the path and provider without quoting file contents. Use
    ``CredentialShapeError`` when a different login type requires a different
    fix.
    """
    import json

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON") from e
    entry = data.get(provider) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        raise KeyError(f"{path} has no entry for {provider}")
    kind = entry.get("type")
    if kind != "api":
        raise CredentialShapeError(
            f"{path} has a {kind!r} credential for {provider}; this backend"
            " carries static API keys only -- subscription logins refresh, and"
            " a refresh from here would race the host's opencode over a file"
            " neither side locks"
        )
    key = entry.get("key")
    if not isinstance(key, str) or not key:
        raise KeyError(f"{path} has no key for {provider}")
    return key


def _catalog_api(catalog: Path, provider: str) -> str:
    """Read a provider's API base URL from OpenCode's public catalog."""
    try:
        data = json.loads(catalog.read_text())
    except OSError as e:
        raise OSError(f"no provider catalog at {catalog}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{catalog} is not valid JSON") from e
    entry = data.get(provider) if isinstance(data, dict) else None
    api = entry.get("api") if isinstance(entry, dict) else None
    if not isinstance(api, str) or not api:
        raise KeyError(f"{catalog} has no api URL for {provider}")
    return api
