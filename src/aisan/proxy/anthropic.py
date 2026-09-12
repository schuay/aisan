# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Proxy Anthropic requests without exposing the host credential to the box.

The proxy accepts requests through a UNIX socket, checks the path, headers, and
body, adds a host credential, and streams the response. Anthropic needs a
separate proxy from Vertex because some client headers must pass through.

Tests with Claude Code 2.1.219 showed that ``anthropic-version`` is required and
that ``anthropic-beta`` selects behavior expected by the client. Reusing the
Vertex proxy would weaken its rule that no client headers reach the upstream.

The proxy forwards ``anthropic-version``, ``anthropic-beta``, ``accept``, and
``x-claude-code-session-id``. It drops authentication, framing, hop-by-hop,
telemetry, and unknown headers. The per-box token in ``authorization`` or
``x-api-key`` never reaches the upstream; the proxy adds the only upstream
credential. Dropping ``x-stainless-*``, ``user-agent``, and ``x-app`` was
verified with a real ``claude -p`` request.

``x-claude-code-session-id`` lets custom upstreams distinguish boxed sessions.
The API doesn't require it, but dropping it would combine all boxes in one
session bucket on an upstream that uses the header. The box chooses the value,
so upstreams must namespace or sanitize client-provided IDs.

Unknown headers are dropped. Client upgrades may require an explicit allowlist
update before new protocol behavior works.

The body also needs policy checks. Tools and content blocks can ask Anthropic to
fetch a URL chosen by the box, bypassing network isolation. ``BodyPolicy`` only
allows request shapes measured from the client and refuses unknown additions.

The shared ``http`` module provides size limits, rate limiting, and socket
lifecycle. This module contains Anthropic-specific protocol and policy rules.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from .http import (
    MAX_BODY_BYTES,
    AmbiguousBody,
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

__all__ = ["BodyPolicy", "HeaderAllowlist", "PathAllowlist", "make_app", "serve"]

_SESSION: web.AppKey[ClientSession] = web.AppKey("session")

_CHUNK = 64 << 10
_TIMEOUT = ClientTimeout(total=None, sock_read=600, sock_connect=30)

# Store lowercase names because frozenset membership is case-sensitive.
FORWARD_HEADERS = frozenset(
    {
        "anthropic-version",
        "anthropic-beta",
        "accept",
        # Let session-aware upstreams distinguish boxed sessions.
        "x-claude-code-session-id",
    }
)

# Observed during a complete Claude Code 2.1.219 turn. Exact matching prevents a
# path such as "/v1/messages_evil" from passing as a messages request.
ALLOWED_PATHS = (
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("GET", "/api/hello"),
    ("HEAD", "/api/hello"),
)


@dataclass(frozen=True)
class PathAllowlist:
    """Define the upstream routes available to the box.

    The injected subscription bearer may authorize more than model calls. This
    allowlist limits the box to the routes needed by Claude Code.
    """

    routes: tuple[tuple[str, str], ...] = ALLOWED_PATHS

    def permits(self, method: str, path: str) -> bool:
        return (method.upper(), path) in self.routes


# Tool types executed inside the box. Other types may execute upstream, outside
# the reach of ``unshare_net``.
#
# An absent ``type`` selects the ``custom`` variant. Anthropic's validation error
# for an untyped tool with ``max_uses`` identifies it as ``custom``. An explicit
# null is a different value and remains invalid.
#
# Claude Code 2.1.219 used no typed client tools in 646 observed declarations
# across chat, shell, editing, skills, subagents, plan mode, and a TUI session.
# Refuse new types until their behavior has been measured.
CLIENT_TOOL_TYPES = frozenset({"custom"})

# Top-level fields that make the upstream execute work outside the box.
REFUSED_KEYS = ("mcp_servers", "container")

# Top-level fields observed during a complete tool round trip with Claude Code
# 2.1.246. Unknown fields are refused because server-side behavior doesn't have
# to be declared as a tool.
ALLOWED_KEYS = frozenset(
    {
        "context_management",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tools",
    }
)

# Content block types that don't reference external data. These were observed in
# the same tool round trip; ``redacted_thinking`` is Anthropic's transformation
# of a thinking block.
INERT_CONTENT_TYPES = frozenset(
    {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}
)

# Image and document payloads may arrive by reference. A ``url`` source makes
# Anthropic fetch a box-controlled URL, while ``file`` refers to Files API state.
#
# ``base64`` carries bytes inline and triggers no fetch. Claude Code 2.1.259 used
# this form for PNGs, PDFs, and pasted images. Vertex applies the same distinction
# by allowing ``inlineData`` and refusing ``fileData``.
#
# Refuse unobserved source types, including document ``text`` and ``content``.
# The latter also contains nested blocks that this policy doesn't inspect.
SOURCED_CONTENT_TYPES = frozenset({"image", "document"})
INLINE_SOURCE_TYPES = frozenset({"base64"})

# Require the exact source shape observed from Claude Code 2.1.259. Checking only
# ``type`` could allow a source with both inline data and a URL. The current
# client strips extra fields and Anthropic rejects them, but the box can construct
# requests directly and upstream validation may change.
INLINE_SOURCE_KEYS = frozenset({"type", "media_type", "data"})

# Derive the union so every sourced type also passes through the source check.
ALLOWED_CONTENT_TYPES = INERT_CONTENT_TYPES | SOURCED_CONTENT_TYPES

# Extra fields allowed when the box shares the host network.
#
# ``web_search_20250305`` makes Anthropic issue a box-controlled query. Claude
# Code 2.1.246 sent it in a dedicated request with ``tool_choice`` and returned
# the result as an ordinary ``tool_result``.
#
# Claude Code's WebFetch isn't included because the box fetches the URL itself
# and sends text to the model. Add other server-side tools only after measuring
# their request shapes.
SHARED_NET_TOOL_TYPES = CLIENT_TOOL_TYPES | {"web_search_20250305"}
SHARED_NET_KEYS = ALLOWED_KEYS | {"tool_choice"}


@dataclass(frozen=True)
class BodyPolicy:
    """Limit the capabilities declared by an Anthropic request body.

    Tools and content blocks can make Anthropic fetch a box-controlled URL.
    ``--unshare-net`` doesn't stop this indirect route, and the URL itself can
    carry data out of the box.

    The policy allows only observed client-side operations. Unknown server-side
    tools and fields are refused.

    Checks cover top-level fields, tool types, and message content. Most content
    blocks are classified by type; images and documents also require an inline
    source. The allowlists come from observed client requests.

    This policy closes an egress route for data already inside the box. Mount
    policy keeps the host credential out of the box.

    A box sharing the host network can already send data directly, so
    ``for_shared_network`` also permits the measured web-search request. The
    backend selects that policy only from ``serve_shared``; callers can't relax
    the isolated policy.
    """

    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    refused_keys: tuple[str, ...] = REFUSED_KEYS
    allowed_keys: frozenset[str] = ALLOWED_KEYS
    content_types: frozenset[str] = ALLOWED_CONTENT_TYPES
    sourced_types: frozenset[str] = SOURCED_CONTENT_TYPES
    inline_sources: frozenset[str] = INLINE_SOURCE_TYPES
    inline_source_keys: frozenset[str] = INLINE_SOURCE_KEYS

    @classmethod
    def for_shared_network(cls) -> BodyPolicy:
        """Return the policy for a box that shares the host network.

        This adds only the two fields observed for web search. MCP, containers,
        other server-side tools, and URL-backed content remain blocked.
        """
        return cls(client_types=SHARED_NET_TOOL_TYPES, allowed_keys=SHARED_NET_KEYS)

    def refuse(self, body: bytes) -> str | None:
        """Return a refusal reason, or ``None`` if the body is allowed.

        Hello routes use an empty body. Other bodies must contain one unambiguous
        JSON object. A streaming upstream might accept ``{...}{}`` as separate
        values, so passing malformed input through could bypass inspection.
        """
        if not body:
            return None
        try:
            payload = load_json_unambiguous(body)
        except AmbiguousBody as e:
            return f"the sandbox proxy cannot read this body unambiguously: {e}"
        except ValueError as e:
            return f"request body is not JSON the sandbox proxy can classify: {e}"
        if not isinstance(payload, dict):
            return "request body must be a JSON object"

        for key in self.refused_keys:
            if key in payload:
                return f"`{key}` is not permitted by the sandbox proxy"
        if unknown := set(payload) - self.allowed_keys:
            return (
                f"field(s) not permitted by the sandbox proxy: {quoted_names(unknown)}"
            )

        reason = self._tools_refusal(payload.get("tools"))
        if reason is not None:
            return reason
        return self._content_refusal(payload)

    def _tools_refusal(self, tools: object) -> str | None:
        if tools is None:
            return None
        if not isinstance(tools, list):
            return "`tools` must be an array"
        for tool in tools:
            if not isinstance(tool, dict):
                return "every tool must be a JSON object"
            # Anthropic treats an absent type as custom and rejects an explicit
            # null, so don't collapse both cases with ``get``.
            if "type" not in tool:
                continue
            kind = tool["type"]
            if kind not in self.client_types:
                return (
                    f"tool type {kind!r} executes on the upstream, not in the"
                    " sandbox, and is not permitted"
                )
        return None

    def _content_refusal(self, payload: dict[str, object]) -> str | None:
        """Check message and system content for external references.

        An image or document with a URL source makes Anthropic fetch a
        box-controlled URL. ``system`` accepts the same block shape as messages.
        """
        messages = payload.get("messages")
        if messages is not None:
            if not isinstance(messages, list):
                return "`messages` must be an array"
            for message in messages:
                if not isinstance(message, dict):
                    return "every message must be a JSON object"
                reason = self._blocks_refusal(message.get("content"))
                if reason is not None:
                    return reason
        return self._blocks_refusal(payload.get("system"))

    def _blocks_refusal(self, content: object) -> str | None:
        """One content value: absent, a plain string, or allowlisted blocks.

        A ``tool_result`` nests another content value of the same shape, which
        recurses through the same allowlist.
        """
        if content is None or isinstance(content, str):
            return None
        if not isinstance(content, list):
            return "message content must be a string or an array of blocks"
        for block in content:
            if not isinstance(block, dict):
                return "every content block must be a JSON object"
            kind = block.get("type")
            if not isinstance(kind, str) or kind not in self.content_types:
                return f"content type {kind!r} is not permitted by the sandbox proxy"
            if kind in self.sourced_types:
                reason = self._source_refusal(kind, block.get("source"))
                if reason is not None:
                    return reason
            if kind == "tool_result":
                reason = self._blocks_refusal(block.get("content"))
                if reason is not None:
                    return reason
        return None

    def _source_refusal(self, kind: str, source: object) -> str | None:
        """Require inline bytes for an image or document block."""
        if not isinstance(source, dict):
            return f"{kind} blocks must carry a `source` object"
        origin = source.get("type")
        if not isinstance(origin, str) or origin not in self.inline_sources:
            return (
                f"{kind} source {origin!r} is not permitted by the sandbox"
                " proxy: only an inline base64 source carries no fetch for the"
                " upstream"
            )
        if extra := set(source) - self.inline_source_keys:
            return (
                f"{kind} source field(s) not permitted by the sandbox proxy:"
                f" {quoted_names(extra)}"
            )
        return None


@dataclass(frozen=True)
class HeaderAllowlist:
    """Select box request headers to forward upstream.

    Checking each header separately lets the fail-closed wrapper drop a header
    whose predicate raises while still evaluating the rest.
    """

    allow: frozenset[str] = FORWARD_HEADERS

    def permits(self, name: str) -> bool:
        return name.lower() in self.allow


def _error(status: int, kind: str, message: str) -> web.Response:
    """Return a refusal in Anthropic's error format.

    The SDK can then show the policy message to the client instead of failing to
    parse an unexpected response.
    """
    return web.json_response(
        {"type": "error", "error": {"type": kind, "message": message}}, status=status
    )


# Resolve the bearer for each request because host Claude may refresh it.
TokenSource = Callable[[], Awaitable[str]]
# Resolve authentication headers for each request. Subscriptions use
# ``authorization`` while static keys use ``x-api-key``.
AuthorizationSource = Callable[[], Awaitable[dict[str, str]]]


def make_app(
    *,
    token: TokenSource | None = None,
    authorization: AuthorizationSource | None = None,
    upstream: str,
    paths: PathAllowlist | None = None,
    headers: HeaderAllowlist | None = None,
    body: BodyPolicy | None = None,
    rate: RateLimit | None = None,
    client_token: str | None = None,
) -> web.Application:
    if (token is None) == (authorization is None):
        raise ValueError("provide exactly one of token or authorization")
    path_allow = paths or PathAllowlist()
    header_allow = headers or HeaderAllowlist()
    body_policy = body or BodyPolicy()
    limiter = rate or RateLimit()
    warn = LogGate()
    base = upstream.rstrip("/")

    async def handle(request: web.Request) -> web.StreamResponse:
        # Accept the per-box token in the header used by the selected credential
        # type. Both paths reject duplicates and compare in constant time. The
        # token isn't forwarded in either header.
        presented = request.headers.getall("x-api-key", [])
        api_key = presented[0] if len(presented) == 1 else None
        if not (
            token_matches(api_key, client_token)
            or token_matches(bearer_token(request), client_token)
        ):
            return _error(401, "authentication_error", "invalid aisan proxy token")
        # Convert exceptions from caller-supplied policy into denials.
        path = request_path(request)
        if not policy_permits(
            lambda: path_allow.permits(request.method, path),
            subject=f"{request.method} {path}",
        ):
            warn.warning(log, "anthropic proxy: refused %s %s", request.method, path)
            return _error(
                403, "permission_error", "path not permitted by the sandbox proxy"
            )
        if not limiter.allow():
            return _error(429, "rate_limit_error", "sandbox proxy rate limit exceeded")
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return _error(413, "request_too_large", "request body too large")
        if len(body) > MAX_BODY_BYTES:
            return _error(413, "request_too_large", "request body too large")

        # Check the body before reading the credential. ``count_tokens`` uses the
        # same tool declarations and therefore needs the same check.
        reason = policy_refusal(
            lambda: body_policy.refuse(body), subject=f"body of {request.path}"
        )
        if reason is not None:
            warn.warning(log, "anthropic proxy: refused body: %s", reason)
            return _error(403, "permission_error", reason)

        try:
            injected = (
                await authorization()
                if authorization is not None
                else {"authorization": f"Bearer {await token()}"}
            )
        except Exception as e:
            # Return credential failures to the agent without exposing token
            # values. The credential may have disappeared or failed to refresh.
            log.error("anthropic proxy: no usable credential to attach: %s", e)
            return _error(
                503,
                "authentication_error",
                f"sandbox proxy has no usable credential: {e}",
            )

        # Preserve repeated ``anthropic-beta`` headers with a list of pairs.
        out: list[tuple[str, str]] = [
            *injected.items(),
            ("content-type", "application/json"),
        ]
        for name, value in request.headers.items():
            if policy_permits(
                lambda name=name: header_allow.permits(name), subject=f"header {name}"
            ):
                out.append((name, value))

        session = request.app[_SESSION]
        url = f"{base}{request.path_qs}"
        try:
            async with session.request(
                request.method,
                url,
                data=body or None,
                headers=out,
                allow_redirects=False,
            ) as up:
                if is_redirect(up.status):
                    log.warning(
                        "anthropic proxy: upstream %s answered %d, not followed",
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
                    # ``prepare`` also writes and can fail if the box disconnects
                    # while the upstream is still working.
                    await resp.prepare(request)
                    async for chunk in up.content.iter_chunked(_CHUNK):
                        await resp.write(chunk)
                    await resp.write_eof()
                except (ConnectionResetError, BrokenPipeError) as e:
                    # The agent exited or cancelled its turn. Catch failures from
                    # both ``prepare`` and body writes so a downstream disconnect
                    # isn't reported as an upstream failure.
                    log.debug(
                        "anthropic proxy: downstream closed: %s", type(e).__name__
                    )
                return resp
        except ClientError as e:
            # A local upstream may restart while the box is running. Return a
            # readable error instead of a disconnected transport that gets retried.
            log.warning("anthropic proxy: upstream %s unreachable: %s", base, e)
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
