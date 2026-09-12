# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Proxy the OpenAI-compatible chat-completions protocol.

OpenCode 1.18.18 uses the same wire format for most providers in its catalog::

    POST {baseURL}/chat/completions
    Authorization: Bearer <key>
    {"stream": true}

The box receives a bare loopback origin. Provider-specific path prefixes stay in
the backend's upstream URL, so one exact route serves the whole family.

No client headers are forwarded. The proxy rebuilds content and authentication
headers for its upstream connection. Measurements found no other required
headers; session IDs, user agents, and similar fields are telemetry.

The body policy permits only ``function`` tools, the client-side type observed
from OpenCode. Unknown types are refused because this transport fronts many
providers with independently changing server-side capabilities.

Errors use the OpenAI ``{"error": {"message", "type"}}`` envelope. OpenCode
1.18.18 displayed its message verbatim, while an Anthropic error became the
unhelpful "Unexpected server error."

The shared ``http`` module provides size limits, rate limiting, and socket
lifecycle. This module contains protocol and policy rules for chat completions.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from .http import (
    MAX_BODY_BYTES,
    LogGate,
    RateLimit,
    bearer_token,
    is_redirect,
    load_json_unambiguous,
    quoted_names,
    relayed_response_headers,
    request_path,
    serve,
    token_matches,
)
from .policy import permits as policy_permits
from .policy import refusal as policy_refusal

log = logging.getLogger(__name__)

__all__ = ["BodyPolicy", "PathAllowlist", "make_app", "serve"]

_SESSION: web.AppKey[ClientSession] = web.AppKey("session")

_CHUNK = 64 << 10
_TIMEOUT = ClientTimeout(total=None, sock_read=600, sock_connect=30)

# OpenCode 1.18.18 used this route for turns and title generation. Exact matching
# excludes paths such as "/chat/completions_evil". Observed turns didn't use
# other family routes such as ``/models`` or ``/embeddings``.
ALLOWED_PATHS = (("POST", "/chat/completions"),)


@dataclass(frozen=True)
class PathAllowlist:
    """Define the upstream routes available to the box.

    The box sends family-wide paths to the proxy while the backend stores each
    provider's path prefix. This allowlist limits the credential to model calls.
    """

    routes: tuple[tuple[str, str], ...] = ALLOWED_PATHS

    def permits(self, method: str, path: str) -> bool:
        return (method.upper(), path) in self.routes


# ``function`` tools execute in the box. Other tool types may make the upstream
# act for the box, outside the reach of ``unshare_net``.
#
# OpenCode always sent the required ``type`` field. Unlike Anthropic, this family
# has no measured default for an absent type, so missing values are refused.
CLIENT_TOOL_TYPES = frozenset({"function"})

# Some top-level fields activate upstream behavior without declaring a tool.
# Keep known server-side fields explicitly refused even if the allowed field set
# later changes.
REFUSED_KEYS: tuple[str, ...] = ("web_search_options",)

# Fields observed in a full OpenCode 1.18.23 tool round trip, plus reasoning
# fields observed with a thinking model on 1.18.18. Unknown fields are refused.
ALLOWED_KEYS = frozenset(
    {
        "max_tokens",
        "messages",
        "model",
        "reasoning_effort",
        "stream",
        "stream_options",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
    }
)

# Text parts contain no URL for the upstream to retrieve. Observed system, user,
# assistant, and tool messages used plain strings. Rich content types that can
# reference images, files, or audio remain blocked.
ALLOWED_CONTENT_TYPES = frozenset({"text"})


@dataclass(frozen=True)
class BodyPolicy:
    """Limit capabilities declared by a chat-completions body.

    Tool types and content parts can make an upstream fetch or execute work for
    the box. Allow only measured client-side shapes so new server-side features
    fail closed.
    """

    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    container_types: frozenset[str] = frozenset()
    # Only these tool fields may contain nested declarations. Other fields must
    # be scalar so uninspected data can't carry another capability.
    structured_tool_keys: frozenset[str] = frozenset({"function"})
    refused_keys: tuple[str, ...] = REFUSED_KEYS
    allowed_keys: frozenset[str] = ALLOWED_KEYS
    content_types: frozenset[str] = ALLOWED_CONTENT_TYPES

    def refuse(self, body: bytes) -> str | None:
        """Return a refusal reason, or ``None`` if the body is allowed.

        Reject malformed or ambiguous input because the proxy and upstream may
        parse it differently.
        """
        try:
            payload = parse_json_object(body)
        except (TypeError, ValueError) as e:
            return f"request body is not unambiguous JSON: {e}"

        for key in self.refused_keys:
            if key in payload:
                return f"`{key}` is not permitted by the sandbox proxy"
        if unknown := set(payload) - self.allowed_keys:
            return (
                f"field(s) not permitted by the sandbox proxy: {quoted_names(unknown)}"
            )

        reason = self._content_refusal(payload)
        if reason is not None:
            return reason

        if "tools" not in payload:
            return None
        return self.refuse_tools(payload["tools"])

    def refuse_tools(self, tools: object) -> str | None:
        """Check a tool array, including nested containers."""
        if not isinstance(tools, list):
            return "`tools` must be an array"
        pending = list(tools)
        while pending:
            tool = pending.pop()
            if not isinstance(tool, dict):
                return "every tool must be a JSON object"
            # Keep an absent type distinct from an explicit null value.
            if "type" not in tool:
                return (
                    "a tool with no `type` is not a declaration this policy"
                    " can resolve, and is not permitted"
                )
            kind = tool["type"]
            if not isinstance(kind, str):
                return "tool `type` must be a string"
            if kind in self.client_types or kind in self.container_types:
                if reason := self._tool_shape_refusal(tool, kind):
                    return reason
            if kind in self.client_types:
                continue
            if kind in self.container_types:
                nested = tool.get("tools")
                if not isinstance(nested, list):
                    return f"tool container {kind!r} must contain a `tools` array"
                pending.extend(nested)
                continue
            return (
                f"tool type {kind!r} executes on the upstream, not in the"
                " sandbox, and is not permitted"
            )
        return None

    def _tool_shape_refusal(self, tool: dict, kind: str) -> str | None:
        """Reject structured values in uninspected tool fields."""
        for key, value in tool.items():
            if key in self.structured_tool_keys:
                continue
            if isinstance(value, (dict, list)):
                return (
                    f"tool {kind!r} field {key!r} is not a field this policy"
                    " reads, and is not permitted to carry a declaration"
                )
        return None

    def _content_refusal(self, payload: dict[str, object]) -> str | None:
        """Check message content parts for external references.

        Fetchable content lets the upstream retrieve a box-controlled URL. Bodies
        without ``messages`` have no content for this check.
        """
        messages = payload.get("messages")
        if messages is None:
            return None
        if not isinstance(messages, list):
            return "`messages` must be an array"
        for message in messages:
            if not isinstance(message, dict):
                return "every message must be a JSON object"
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list):
                return "message content must be a string or an array of parts"
            for part in content:
                if not isinstance(part, dict):
                    return "every content part must be a JSON object"
                kind = part.get("type")
                if not isinstance(kind, str) or kind not in self.content_types:
                    return (
                        f"content type {kind!r} is not permitted by the sandbox proxy"
                    )
        return None


def parse_json_object(body: bytes) -> dict[str, object]:
    """Parse one unambiguous JSON object for policy inspection.

    Shared parsing gives Anthropic, Vertex, and OpenAI-compatible policies the
    same treatment of duplicate keys and nonstandard constants.
    """
    payload = load_json_unambiguous(body)
    if not isinstance(payload, dict):
        raise TypeError("request body must be a JSON object")
    return payload


def _error(status: int, kind: str, message: str) -> web.Response:
    """Return a refusal in the OpenAI-compatible error envelope.

    OpenCode displays the ``message`` field to the user.
    """
    return web.json_response(
        {"error": {"message": message, "type": kind}}, status=status
    )


# Read the bearer for each request because host OpenCode rewrites it on /connect.
TokenSource = Callable[[], Awaitable[str]]
HeaderSource = Callable[[bytes], dict[str, str]]
AuthorizationSource = Callable[[], Awaitable[dict[str, str]]]


def make_app(
    *,
    token: TokenSource | None = None,
    authorization: AuthorizationSource | None = None,
    upstream: str,
    paths: PathAllowlist | None = None,
    body: BodyPolicy | None = None,
    rate: RateLimit | None = None,
    headers: HeaderSource | None = None,
    client_token: str | None = None,
) -> web.Application:
    if (token is None) == (authorization is None):
        raise ValueError("provide exactly one of token or authorization")
    path_allow = paths or PathAllowlist()
    body_policy = body or BodyPolicy()
    limiter = rate or RateLimit()
    warn = LogGate()
    base = upstream.rstrip("/")

    async def handle(request: web.Request) -> web.StreamResponse:
        if not token_matches(bearer_token(request), client_token):
            return _error(401, "authentication_error", "invalid aisan proxy token")
        # Convert exceptions from caller-supplied policy into denials.
        path = request_path(request)
        if not policy_permits(
            lambda: path_allow.permits(request.method, path),
            subject=f"{request.method} {path}",
        ):
            warn.warning(
                log, "openai-compat proxy: refused %s %s", request.method, path
            )
            return _error(
                403, "invalid_request_error", "path not permitted by the sandbox proxy"
            )
        if not limiter.allow():
            return _error(429, "rate_limit_error", "sandbox proxy rate limit exceeded")
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return _error(413, "invalid_request_error", "request body too large")
        if len(body) > MAX_BODY_BYTES:
            return _error(413, "invalid_request_error", "request body too large")

        # Check policy before reading the host credential.
        reason = policy_refusal(
            lambda: body_policy.refuse(body), subject=f"body of {request.path}"
        )
        if reason is not None:
            warn.warning(log, "openai-compat proxy: refused body: %s", reason)
            return _error(403, "invalid_request_error", reason)

        try:
            if authorization is not None:
                authorization_headers = await authorization()
            else:
                assert token is not None
                authorization_headers = {"Authorization": f"Bearer {await token()}"}
        except Exception as e:
            # Return a readable error if the credential changes or disappears.
            log.error("openai-compat proxy: no credential to attach: %s", e)
            return _error(503, "api_error", f"sandbox proxy has no credential: {e}")

        # Build all upstream headers locally. Responses may reparse the body to
        # reconstruct protocol headers, so treat errors as refusals.
        try:
            upstream_headers = headers(body) if headers is not None else {}
        except Exception as e:
            log.error("openai-compat proxy: header reconstruction failed: %s", e)
            return _error(500, "api_error", "sandbox proxy could not build the request")
        upstream_headers.update(authorization_headers)
        upstream_headers.update(
            {
                "Content-Type": "application/json",
            }
        )

        session = request.app[_SESSION]
        url = f"{base}{request.path_qs}"
        try:
            async with session.request(
                request.method,
                url,
                data=body or None,
                headers=upstream_headers,
                allow_redirects=False,
            ) as up:
                if is_redirect(up.status):
                    log.warning(
                        "openai-compat proxy: upstream %s answered %d, not followed",
                        base,
                        up.status,
                    )
                    return _error(
                        502,
                        "api_error",
                        "sandbox proxy does not follow redirects from its upstream",
                    )
                resp = web.StreamResponse(
                    status=up.status, headers=relayed_response_headers(up.headers)
                )
                try:
                    # ``prepare`` can fail if the box disconnects while the
                    # upstream is still working.
                    await resp.prepare(request)
                    async for chunk in up.content.iter_chunked(_CHUNK):
                        await resp.write(chunk)
                    await resp.write_eof()
                except (ConnectionResetError, BrokenPipeError) as e:
                    # Catch failures from headers and body writes so a downstream
                    # disconnect isn't reported as an upstream failure.
                    log.debug(
                        "openai-compat proxy: downstream closed: %s", type(e).__name__
                    )
                return resp
        except ClientError as e:
            # Return a readable failure instead of a retryable disconnect.
            log.warning("openai-compat proxy: upstream %s unreachable: %s", base, e)
            return _error(
                502, "api_error", f"sandbox proxy cannot reach its upstream: {e}"
            )

    app = web.Application(client_max_size=MAX_BODY_BYTES + 1)
    app.router.add_route("*", "/{tail:.*}", handle)

    async def _open(a: web.Application) -> None:
        a[_SESSION] = ClientSession(timeout=_TIMEOUT)

    async def _close(a: web.Application) -> None:
        await a[_SESSION].close()

    app.on_startup.append(_open)
    app.on_cleanup.append(_close)
    return app
