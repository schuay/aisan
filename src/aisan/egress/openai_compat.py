# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The OpenAI-compatible backend: one parameterised route for a family of them.

`aisan.proxy.openai_compat` owns the mechanism (path allowlist, body policy,
credential injection, forwarding); this owns the backend -- and the backend is
the answer to "how do you support 147 providers without 147 backends": all of
them speak the same wire, and what varies is DATA, recorded in opencode's own
provider catalog (`~/.cache/opencode/models.json`, fetched from models.dev):
the `api` base URL per provider id. So a provider is supported by NAMING it,
and its wire facts come from the same catalog the host's opencode already
maintains -- the first consumer is `zai-coding-plan`, and the second is a
constructor argument, not a code change.

Two credential files, two rules, both the settled ones from the Anthropic
backend:

- **The host's `auth.json` is read and never written.** It is the HOST's own
  opencode credential store (`~/.local/share/opencode/auth.json`, 0600), which
  that process rewrites whenever the user re-runs `/connect`. This backend
  carries STATIC API keys only: an entry of another shape (`oauth`, from a
  subscription login; `wellknown`) is a refusal naming the shape, because
  those refresh -- and a refresh from here would race the host's opencode over
  a file neither side locks, which is the failure the never-write rule exists
  to make impossible. Static keys do not expire, so there is no expiry margin
  and no refresh story at all: preflight checks that the key is THERE, and a
  mid-session re-read picks up a re-entered one for free.
- **The box gets a placeholder, not the key.** `OPENCODE_AUTH_CONTENT` carries
  `{"<provider>": {"type": "api", "key": <placeholder>}}` -- an in-memory
  auth store, measured: when set, opencode does not read `auth.json` AT ALL,
  which is a stronger property than an env key (a future bind that exposed the
  file would still expose nothing the client consults). The proxy drops the
  `Authorization` the box presents and attaches the real bearer host-side.

Routing is inline config, and inline config WINS. `OPENCODE_CONFIG_CONTENT`
points the provider's `baseURL` at the in-box relay and disables session
sharing; it sits above the PROJECT config in opencode's precedence, which
matters here: the agent can edit the repo's `opencode.json`, and a project
config that re-pointed the base URL would otherwise be a routing the box chose
for itself. (Containment never rested on that -- `unshare_net` is the boundary,
and a re-pointed client just fails to connect -- but the route holding without
cooperation from inside the repo is the property worth stating.) `share` is
disabled here rather than in the preset because uploading transcripts IS
egress, and egress decisions are this layer's to make.

The catalog also answers a trap the embedded model list sets: the binary ships
a model catalog that LAGS the fetched one (measured: `glm-5.3` refused as
unknown with the fetched catalog listing it), so the preset binds the host's
catalog ro at the path opencode reads by default -- see `presets/opencode.py`,
which owns that bind because it is a fact about the BOX's filesystem, not
about this route.
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


# XDG-aware rather than hardcoded, on both files, because these are the HOST's
# opencode paths and the host's opencode writes wherever the HOST's XDG says --
# a backend that looked only under ~/.local while the host set XDG_DATA_HOME
# would describe a credential that is not where the user put it.
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
# Not a login command: no login turns a subscription credential into a static
# API key, so the fix is to connect the provider WITH one, or to point the
# backend at a file that has one.
_FIX_SHAPE = (
    "connect the provider with an API key (opencode /connect on the HOST), or"
    ' point the backend at credentials whose entry has "type": "api"'
)
_FIX_CATALOG = (
    "run opencode once on the HOST (it fetches the provider catalog), or pass"
    " upstream= explicitly"
)


class OpenAICompatBackend(Backend):
    """One provider of the OpenAI-compatible family, through a host-side proxy
    on loopback `PORT`."""

    #: The in-box loopback port. Fixed per backend; see `egress.base`. A box
    #: composing TWO providers of this family overrides `port` on the second
    #: (BoxSpec refuses the collision otherwise) -- one route per box is the
    #: common shape, not the only expressible one.
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
        # The provider's full `api` base (origin + path prefix, e.g.
        # "https://api.z.ai/api/coding/paas/v4"), when the caller knows it.
        # None means resolve it from the catalog -- the data-driven path that
        # makes another provider a name rather than a code change.
        self._upstream = upstream
        self._credentials = credentials or default_credentials()
        # For the Box's credential-exposure refusal. The one file, and note it
        # carries EVERY provider's key -- see Backend.credentials.
        self.credentials = (self._credentials,)
        self._catalog = catalog or DEFAULT_CATALOG
        self._rpm = rpm
        # Distinct per provider so a manifest with two of this family names
        # two sockets, not the same one twice.
        self.name = f"openai-{provider}"
        self.port = port
        self._api: str | None = None

    # --- host side: where the route goes ------------------------------------

    def _resolve_api(self) -> str:
        """The provider's `api` base URL, from the ctor or the catalog.

        Cached on first resolution: the catalog is not credential-shaped (it
        is a public model listing), so there is no refresh race to lose by
        holding it, and re-reading per request would only allow a mid-session
        catalog swap to silently re-point an established route.
        """
        if self._api is None:
            self._api = self._upstream or _catalog_api(self._catalog, self._provider)
        return self._api

    async def _bearer(self) -> str:
        """The provider's API key. Read per request, never cached.

        Not cached for the same reason the Anthropic backend re-reads, even
        though a static key cannot expire: the HOST's opencode rewrites
        `auth.json` on /connect, and a value captured at preflight would
        survive a key the user has since rotated. The file is small and local.
        """
        return _read_api_key(self._credentials, self._provider)

    async def preflight(self) -> None:
        """Fail before bwrap owns the terminal, with the right fix for each
        case. Three things can be wrong and they need different commands:

        - the route cannot be resolved (no catalog on this host, provider
          unknown to it) -> fetch the catalog, or name the upstream;
        - no usable key for the provider -> the user has not connected it
          HERE, so the login command -- or the entry is a shape this backend
          does not carry, which no login fixes and must say so;
        - a dead upstream -> nothing to log into.
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
        """Does the upstream answer at all. A connect, not a model call (the
        same rule as anthropic's): any HTTP answer counts, including a 4xx --
        an upstream refusing THIS request is still an upstream, and only a
        transport failure means the hop is missing.

        `/models` because every member of the family serves it and no member
        requires a body to answer SOMETHING; the key is deliberately not
        attached, so the check spends nothing and proves only reachability.
        """
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(f"{api.rstrip('/')}/models"),
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
        # `serve_proxy` removes a leftover socket from a killed box before it
        # binds -- without that, bind() fails EADDRINUSE forever on a path
        # this box derived and nothing else owns.
        #
        # The proxy dials the FULL `api` base, prefix included: the box's
        # requests arrive with the family-canonical path (no prefix -- the
        # in-box base URL is the relay's bare origin), so the prefix has to
        # live in the upstream or it is lost between the two halves.
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

    # --- box side: what the client is told -----------------------------------

    def client_env(self) -> dict[str, str]:
        """What the box's opencode reads to find this route, and nothing real.

        Pure -- no file reads -- because nothing in-box depends on the wire
        data: the in-box base URL is the relay's bare origin (the provider's
        path prefix is host-side knowledge; the proxy re-attaches it by
        dialing `api` as configured), so the catalog is not consulted here and
        the spec, the snapshot and the dry run all build without one.

        `model`, when given, is the BARE model id (`glm-5.3`); the provider
        prefix is composed here so a caller cannot desynchronise the two
        halves of `provider/model`. The TUI has no `--model` flag (measured,
        1.18.18: only `opencode run` does), so the default model rides this
        config content -- that is why it is here at all.
        """
        return self._client_env(self.port, PLACEHOLDER_KEY)

    def _client_env(self, port: int | str, key: str) -> dict[str, str]:
        auth = {self._provider: {"type": "api", "key": key}}
        config: dict[str, object] = {
            # Session sharing uploads the transcript to opencode's servers:
            # egress this backend does not allowlist, so it is off here where
            # project config cannot override it.
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
    """The auth entry exists but is not one this backend carries.

    Separate from a missing/unreadable file because the FIX differs: no login
    creates a static API key out of an OAuth subscription, and telling the
    operator to re-login would be the wrong command confidently stated.
    """


def _read_api_key(path: Path, provider: str) -> str:
    """The provider's static API key out of an opencode `auth.json`.

    Raises OSError/ValueError/KeyError naming the PATH and provider, never the
    contents: a message that quoted the document to explain a missing key
    would put the key wherever that message is logged. `CredentialShapeError`
    is raised (not leaked as a bare string) when the entry's shape is one this
    backend does not carry, so preflight can attach the fix that actually
    helps.
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
    """The provider's `api` base URL out of opencode's provider catalog.

    The catalog is the file the HOST's opencode fetched from models.dev --
    public data, no secrets, which is why a parse failure names the path and
    provider and moves on rather than being treated as a security event.
    """
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
