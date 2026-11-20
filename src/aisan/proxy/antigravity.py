# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Proxy Antigravity CLI traffic to Cloud Code without exposing a credential.

Antigravity CLI 1.5.11 sends every account and model call to one Cloud Code
host as ``POST /v1internal:<method>``. Model calls wrap a Gemini
``GenerateContent`` request in an envelope that names the model, whichever
vendor serves it, so one body policy covers them all.

The route allowlist is the measured method set, each with its measured query
string. The proxy refuses the calls that write account settings or upload
analytics, and answers the profile lookup itself: the client asks for it
through ``USER_INFO_OVERRIDE_URL`` only to show an identity.

The box presents a placeholder bearer, or a per-activation token when it
shares the host network. The proxy checks it, then replaces it with the host
token. It never falls back to what the box sent. Cloud Code authorises partly
on the client identity in ``User-Agent``, so that header is forwarded; every
other upstream header is built here or named by the profile.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import MappingProxyType

from aiohttp import ClientSession, ClientTimeout, web

from ..upstream import DIRECT, Transport, upstream_origin, validate_upstream
from .gemini import BodyPolicy
from .http import (
    MAX_BODY_BYTES,
    AmbiguousBody,
    LogGate,
    RateLimit,
    bearer_token,
    forward,
    load_json_unambiguous,
    quoted_names,
    read_capped_body,
    request_path,
    serve,
    token_matches,
)
from .policy import permits as policy_permits
from .policy import refusal as policy_refusal

__all__ = [
    "ALLOWED_ROUTES",
    "DEFAULT_UPSTREAM",
    "ENVELOPE_KEYS",
    "GENERATE_ROUTES",
    "QUIET_REFUSALS",
    "USERINFO",
    "LocalAnswer",
    "Profile",
    "RouteAllowlist",
    "make_app",
    "serve",
]

log = logging.getLogger(__name__)

_SESSION: web.AppKey[ClientSession] = web.AppKey("session")

_TIMEOUT = ClientTimeout(total=None, sock_read=600, sock_connect=30)

# Turns against ``cloudcode-pa.googleapis.com`` got 429 on a consumer login
# while this host served them, and host agy 1.5.11 uses only this one.
DEFAULT_UPSTREAM = "https://daily-cloudcode-pa.googleapis.com"

Route = tuple[str, str]

# Each route maps to the raw query strings the client sent with it.
_NO_QUERY = frozenset({""})
GENERATE_ROUTES = MappingProxyType(
    {
        ("POST", "/v1internal:streamGenerateContent"): frozenset({"alt=sse"}),
        # Search runs as a unary call; the body policy decides whether the
        # box may make it.
        ("POST", "/v1internal:generateContent"): _NO_QUERY,
    }
)
ALLOWED_ROUTES = MappingProxyType(
    {
        ("POST", "/v1internal:loadCodeAssist"): _NO_QUERY,
        ("POST", "/v1internal:fetchUserInfo"): _NO_QUERY,
        ("POST", "/v1internal:fetchAdminControls"): _NO_QUERY,
        ("POST", "/v1internal:retrieveUserQuotaSummary"): _NO_QUERY,
        ("POST", "/v1internal:fetchAvailableModels"): _NO_QUERY,
        ("POST", "/v1internal:listExperiments"): _NO_QUERY,
        ("POST", "/v1internal:writeTrajectoryAcls"): _NO_QUERY,
        # In-app tips, read by the interactive client.
        ("GET", "/v1internal/cascadeNuxes"): frozenset({"page_size=1000"}),
        **GENERATE_ROUTES,
    }
)

# Calls the client makes in normal use that the box has no business making:
# ``setUserSettings`` writes the account's settings, including the telemetry
# opt-in, and ``recordTrajectoryAnalytics`` uploads the session trajectory.
# The client logs a refused ``setUserSettings`` and carries on.
QUIET_REFUSALS: frozenset[Route] = frozenset(
    {
        ("POST", "/v1internal:setUserSettings"),
        ("POST", "/v1internal:recordTrajectoryAnalytics"),
    }
)

# Envelope fields of a model call. Unknown fields are refused: only ``request``
# is inspected, and another field could carry a capability past it.
ENVELOPE_KEYS = frozenset(
    {"model", "project", "request", "requestId", "requestType", "userAgent"}
)


@dataclass(frozen=True)
class LocalAnswer:
    """A response the proxy gives itself, without contacting the upstream."""

    body: bytes
    content_type: str = "application/json"
    status: int = 200


# Google's userinfo shape. The client shows the address and fetches no avatar
# for an empty ``picture``.
USERINFO = LocalAnswer(
    json.dumps(
        {
            "id": "0",
            "email": "box@aisan.invalid",
            "verified_email": True,
            "name": "aisan box",
            "given_name": "aisan",
            "family_name": "box",
            "picture": "",
        }
    ).encode()
)

_LOCAL_ANSWERS = MappingProxyType({("GET", "/userinfo"): USERINFO})

# Upstream headers the proxy builds itself, lowercased.
_BUILT_HEADERS = frozenset({"authorization", "content-type", "user-agent"})


@dataclass(frozen=True)
class Profile:
    """What differs between builds of the client that speak this protocol.

    ``routes`` and ``local`` extend the shared sets; a build cannot remove a
    shared refusal by listing its route. ``headers`` are sent upstream with
    every forwarded request, for an upstream that needs more than the bearer.
    The box cannot set them, and they cannot replace a header the proxy builds.
    """

    upstream: str = DEFAULT_UPSTREAM
    routes: MappingProxyType[Route, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    local: MappingProxyType[Route, LocalAnswer] = field(
        default_factory=lambda: MappingProxyType({})
    )
    headers: MappingProxyType[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class RouteAllowlist:
    """Match a request against the permitted method, path and query."""

    routes: MappingProxyType[Route, frozenset[str]] = ALLOWED_ROUTES
    quiet_refusals: frozenset[Route] = QUIET_REFUSALS

    def permits(self, method: str, path: str, query: str) -> bool:
        if (method, path) in self.quiet_refusals:
            return False
        return query in self.routes.get((method, path), ())

    def warns_on_refusal(self, method: str, path: str) -> bool:
        return (method, path) not in self.quiet_refusals


_STATUS_NAMES = MappingProxyType(
    {
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        413: "INVALID_ARGUMENT",
        429: "RESOURCE_EXHAUSTED",
    }
)


def _error(status: int, message: str) -> web.Response:
    """Return a refusal in the Google API error envelope the client parses."""
    return web.json_response(
        {
            "error": {
                "code": status,
                "message": message,
                "status": _STATUS_NAMES.get(status, "UNAVAILABLE"),
            }
        },
        status=status,
    )


def _envelope_refusal(
    body: bytes, policy: BodyPolicy, models: frozenset[str] | None = None
) -> str | None:
    """Check a model call's envelope, then the Gemini request inside it."""
    try:
        payload = load_json_unambiguous(body)
    except AmbiguousBody as e:
        return f"the sandbox proxy cannot read this body unambiguously: {e}"
    except ValueError as e:
        return f"request body is not JSON the sandbox proxy can classify: {e}"
    if not isinstance(payload, dict):
        return "request body must be a JSON object"
    if unknown := set(payload) - ENVELOPE_KEYS:
        return f"field(s) not permitted by the sandbox proxy: {quoted_names(unknown)}"
    if "request" not in payload:
        return "a model call must carry a `request`"
    if models is not None:
        model = payload.get("model")
        if not isinstance(model, str) or model not in models:
            return f"model {model!r} is not permitted by the sandbox proxy"
    return policy.refuse_request(payload["request"])


# Resolve the bearer for every request because the host client refreshes it.
TokenSource = Callable[[], Awaitable[str]]


def make_app(
    *,
    token: TokenSource,
    client_token: str,
    profile: Profile | None = None,
    body: BodyPolicy | None = None,
    models: frozenset[str] | None = None,
    rate: RateLimit | None = None,
    transport: Transport = DIRECT,
) -> web.Application:
    """Build the proxy; ``client_token`` is the bearer the box must present.

    ``models`` names the models a call may address; ``None`` leaves the choice
    to the box. A login serves every model its account can reach, so a
    deployment that runs one configured model keeps a box from spending on
    another.
    """
    chosen = profile or Profile()
    allow = RouteAllowlist(routes=MappingProxyType({**chosen.routes, **ALLOWED_ROUTES}))
    local = {**chosen.local, **_LOCAL_ANSWERS}
    extra_headers = {
        name: value
        for name, value in chosen.headers.items()
        if name.lower() not in _BUILT_HEADERS
    }
    body_policy = body or BodyPolicy()
    limiter = rate or RateLimit()
    warn = LogGate()
    base = validate_upstream(chosen.upstream)
    origin = upstream_origin(base)

    async def handle(request: web.Request) -> web.StreamResponse:
        if not token_matches(bearer_token(request), client_token):
            return _error(401, "invalid aisan proxy token")
        path = request_path(request)
        route = (request.method, path)
        query = request.rel_url.raw_query_string

        answer = local.get(route)
        if answer is not None and not query:
            return web.Response(
                body=answer.body,
                status=answer.status,
                content_type=answer.content_type,
            )

        if not policy_permits(
            lambda: allow.permits(request.method, path, query),
            subject=f"{request.method} {path}",
        ):
            if allow.warns_on_refusal(request.method, path):
                warn.warning(
                    log, "antigravity proxy: refused %s %s", request.method, path
                )
            return _error(403, "path not permitted by the sandbox proxy")
        if not limiter.allow():
            return _error(429, "sandbox proxy rate limit exceeded")
        body = await read_capped_body(request)
        if body is None:
            return _error(413, "request body too large")

        if route in GENERATE_ROUTES:
            reason = policy_refusal(
                lambda: _envelope_refusal(body, body_policy, models),
                subject=f"body of {path}",
            )
            if reason is not None:
                warn.warning(log, "antigravity proxy: refused body: %s", reason)
                return _error(403, reason)

        try:
            bearer = await token()
        except Exception as e:
            # Report the failure without its token; never send the box's own
            # bearer upstream instead.
            log.error("antigravity proxy: no usable credential to attach: %s", e)
            return _error(503, f"sandbox proxy has no usable credential: {e}")

        headers = {**extra_headers, "Authorization": f"Bearer {bearer}"}
        if body:
            headers["Content-Type"] = "application/json"
        if agent := request.headers.get("User-Agent"):
            headers["User-Agent"] = agent
        return await forward(
            request,
            request.app[_SESSION],
            url=f"{base}{request.path_qs}",
            body=body,
            headers=headers,
            refuse=_error,
            logger=log,
            gate=warn,
            label="antigravity",
            origin=origin,
            failure_detail=True,
        )

    app = web.Application(client_max_size=MAX_BODY_BYTES + 1)
    app.router.add_route("*", "/{tail:.*}", handle)

    async def _open(a: web.Application) -> None:
        a[_SESSION] = transport.session(_TIMEOUT)

    async def _close(a: web.Application) -> None:
        await a[_SESSION].close()

    app.on_startup.append(_open)
    app.on_cleanup.append(_close)
    return app
