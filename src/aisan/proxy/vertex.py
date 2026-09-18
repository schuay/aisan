# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Proxy allowlisted Vertex requests without exposing a credential.

The network-isolated box reaches a host UNIX socket through ``relay.py``. This
proxy checks each request, adds a host bearer, and streams the Vertex response.

Configuration fixes the project, location, and allowed models. Exact path
patterns keep requests within those resources. Gemini and Anthropic models use
separate routes and body policies, selected by the pattern that matched.

No request headers pass through. The proxy builds its own headers and can add a
per-box session ID for cache affinity, on both dialects: Claude's explicit
prefix cache and Gemini's implicit one are each served by the replica that
holds the prefix. Dropping the client's session ID prevents the box from
choosing an upstream routing key.

Body policy permits only operations executed inside the box. Server-side search,
URL retrieval, and referenced file data could otherwise bypass ``unshare_net``.
Cached-content requests receive the same checks as direct inference.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

# Reuse the measured Anthropic Messages body policy for Claude on Vertex.
from . import anthropic

# Re-exported for compatibility; new shared-infrastructure imports should use
# `aisan.proxy.http` directly.
from .http import (
    MAX_BODY_BYTES,
    AmbiguousBody,
    LogGate,
    RateLimit,
    is_redirect,
    load_json_unambiguous,
    relayed_response_headers,
    request_path,
    run_forever,
    serve,
)
from .policy import decision as policy_decision
from .policy import refusal as policy_refusal

__all__ = [
    "MAX_BODY_BYTES",
    "VERTEX_ANTHROPIC_KEYS",
    "Allowlist",
    "BodyPolicy",
    "RateLimit",
    "anthropic_body_policy",
    "make_app",
    "run_forever",
    "serve",
]

log = logging.getLogger(__name__)

# aiohttp requires a typed key for application state.
_SESSION: web.AppKey[ClientSession] = web.AppKey("session")


# Vertex is regional; "global" alone has no host prefix.
def _upstream(location: str) -> str:
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return f"https://{host}"


# Forward chunks incrementally so streamed responses aren't buffered in full.
_CHUNK = 64 << 10

_TIMEOUT = ClientTimeout(total=None, sock_read=600, sock_connect=30)


@dataclass(frozen=True)
class Allowlist:
    """Match requests for one configured project, location, and model set.

    Escape all configured values before adding them to regular expressions.
    Gemini and Anthropic use different API versions, publishers, methods, and
    body formats, so each gets its own model list and tagged pattern.
    """

    project: str
    location: str
    models: tuple[str, ...]
    anthropic_models: tuple[str, ...] = ()

    def _patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        """Return allowed path patterns tagged with their body protocol.

        A single match chooses both access and body policy. Omit a model pattern
        when its model list is empty; an empty regex group would match a path
        with no model name.
        """
        base = rf"/v1beta1/projects/{re.escape(self.project)}/locations/{re.escape(self.location)}"
        models = "|".join(re.escape(m) for m in self.models)
        patterns: list[tuple[str, re.Pattern[str]]] = []
        if self.models:
            patterns.append(
                (
                    "google",
                    re.compile(
                        rf"^{base}/publishers/google/models/({models})"
                        r":(generateContent|streamGenerateContent)$"
                    ),
                )
            )
        patterns += [
            ("google", re.compile(rf"^{base}/cachedContents$")),
            ("google", re.compile(rf"^{base}/cachedContents/[0-9]+$")),
        ]
        if self.anthropic_models:
            # `/v1`, not `/v1beta1`: the Anthropic SDK's Vertex client puts
            # that version in its base URL.
            anthropic_base = (
                rf"/v1/projects/{re.escape(self.project)}"
                rf"/locations/{re.escape(self.location)}"
            )
            anthropic_models = "|".join(re.escape(m) for m in self.anthropic_models)
            patterns.append(
                (
                    "anthropic",
                    re.compile(
                        rf"^{anthropic_base}/publishers/anthropic/models/"
                        rf"({anthropic_models})"
                        r":(rawPredict|streamRawPredict)$"
                    ),
                )
            )
        return tuple(patterns)

    def dialect(self, method: str, path: str) -> str | None:
        """Return the matching body protocol, or ``None`` for a denial."""
        if method not in ("POST", "DELETE"):
            return None
        for name, pattern in self._patterns():
            if pattern.match(path):
                return name
        return None

    def permits(self, method: str, path: str) -> bool:
        return self.dialect(method, path) is not None


# Both proto3 JSON spellings for content that makes Google fetch a URI. Inline
# base64, text, function, and thought parts don't fetch external data.
FETCH_PART_FIELDS = frozenset({"fileData", "file_data"})


# Function declarations execute inside the box. Accept both proto3 JSON
# spellings because the upstream accepts both; the REST client emits camelCase.
#
# Other fields make the upstream act for the box. Current discovery documents
# include search, URL context, code execution, retrieval, maps, computer use,
# and external API calls. Unknown future fields are refused.
CLIENT_TOOL_FIELDS = frozenset({"functionDeclarations", "function_declarations"})


@dataclass(frozen=True)
class BodyPolicy:
    """Limit capabilities declared by a Gemini request body.

    Gemini tools and content parts can ask Google to fetch, search, or execute
    outside the box. Allow only function declarations, which execute locally,
    and inline content. Apply the same checks to cached-content requests because
    they accept tool declarations too.

    Anthropic routes use ``anthropic_body_policy`` instead. Mount policy, rather
    than this body check, keeps credentials out of the box.
    """

    client_fields: frozenset[str] = CLIENT_TOOL_FIELDS

    def refuse(self, body: bytes) -> str | None:
        """Return a refusal reason, or ``None`` if the body is allowed.

        Cache deletion may use an empty body. Other requests must contain one
        unambiguous JSON object emitted by the REST SDK.
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

        reason = self._contents_refusal(payload.get("contents"))
        if reason is not None:
            return reason

        tools = payload.get("tools")
        if tools is None:
            return None
        if not isinstance(tools, list):
            return "`tools` must be an array"
        for tool in tools:
            if not isinstance(tool, dict):
                return "every tool must be a JSON object"
            for field in tool:
                if field not in self.client_fields:
                    return (
                        f"tool field {field!r} executes on the upstream, not in"
                        " the sandbox, and is not permitted"
                    )
        return None

    def _contents_refusal(self, contents: object) -> str | None:
        """Reject content parts that ask Google to fetch a URI."""
        if not isinstance(contents, list):
            return None
        for content in contents:
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict):
                    for field in part:
                        if field in FETCH_PART_FIELDS:
                            return (
                                f"content part field {field!r} makes the upstream"
                                " fetch a URL the box chose, and is not permitted"
                            )
        return None


# ``rawPredict`` accepts Anthropic Messages bodies verbatim, so reuse that policy.
# It handles untyped custom tools and URL-backed image or document blocks.
#
# Add fields observed from ChatAnthropicVertex. ``tool_choice`` can only select a
# tool already checked by the shared Anthropic policy.
#
# Remove ``model`` because this route selects it through the checked URL path.
VERTEX_ANTHROPIC_KEYS = (
    anthropic.ALLOWED_KEYS
    | {"anthropic_version", "stop_sequences", "tool_choice", "top_k", "top_p"}
) - {"model"}


def anthropic_body_policy() -> anthropic.BodyPolicy:
    """Create the policy for ``publishers/anthropic`` routes."""
    return anthropic.BodyPolicy(allowed_keys=VERTEX_ANTHROPIC_KEYS)


def _error(status: int, message: str, *, streaming: bool) -> web.Response:
    """Return a refusal in the response shape expected by Vertex clients.

    Streaming Gemini errors use a JSON array; other routes use an object. The
    client can then display the policy message instead of a parsing exception.
    """
    body: object = {
        "error": {"code": status, "message": message, "status": "PERMISSION_DENIED"}
    }
    if streaming:
        body = [body]
    return web.json_response(body, status=status)


# Resolve the bearer for every request because jobs outlive tokens.
TokenSource = Callable[[], Awaitable[str]]


def make_app(
    *,
    allowlist: Allowlist,
    token: TokenSource,
    location: str,
    rate: RateLimit | None = None,
    body: BodyPolicy | None = None,
    anthropic_body: anthropic.BodyPolicy | None = None,
    session_header: str | None = None,
    session_id: str | None = None,
) -> web.Application:
    limiter = rate or RateLimit()
    # Session affinity lets later turns reuse cached prefixes. The deployment
    # supplies the header name because upstreams use different conventions.
    #
    # Create one ID per box. Tests may inject a deterministic value.
    vertex_session = session_id or uuid.uuid4().hex
    # Select body policy from the matched route dialect. Unknown dialects fail
    # inside the denial wrapper.
    body_policies = {
        "google": body or BodyPolicy(),
        "anthropic": anthropic_body or anthropic_body_policy(),
    }
    warn = LogGate()
    upstream = _upstream(location)

    async def handle(request: web.Request) -> web.StreamResponse:
        path = request_path(request)
        # Match the final method exactly so similarly named paths don't receive
        # the streaming error envelope.
        #
        # Anthropic stream errors use a plain object rather than Gemini's array.
        streaming = path.endswith(":streamGenerateContent")
        # Convert policy exceptions to denials and retain the matching dialect.
        dialect = policy_decision(
            lambda: allowlist.dialect(request.method, path),
            subject=f"{request.method} {path}",
        )
        if dialect is None:
            warn.warning(log, "vertex proxy: refused %s %s", request.method, path)
            return _error(
                403, "path not permitted by the sandbox proxy", streaming=streaming
            )
        if not limiter.allow():
            return _error(429, "sandbox proxy rate limit exceeded", streaming=streaming)
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return _error(413, "request body too large", streaming=streaming)
        if len(body) > MAX_BODY_BYTES:
            return _error(413, "request body too large", streaming=streaming)

        # Check body policy before reading the credential. Convert policy and
        # dialect lookup errors to denials.
        reason = policy_refusal(
            lambda: body_policies[dialect].refuse(body),
            subject=f"body of {request.path}",
        )
        if reason is not None:
            warn.warning(log, "vertex proxy: refused body: %s", reason)
            return _error(403, reason, streaming=streaming)

        # Build every upstream header locally.
        headers = {
            "Authorization": f"Bearer {await token()}",
            "Content-Type": "application/json",
        }
        if session_header:
            headers[session_header] = vertex_session
        session = request.app[_SESSION]
        url = f"{upstream}{request.path_qs}"
        try:
            async with session.request(
                request.method,
                url,
                data=body or None,
                headers=headers,
                allow_redirects=False,
            ) as up:
                if is_redirect(up.status):
                    log.warning(
                        "vertex proxy: upstream %s answered %d, not followed",
                        upstream,
                        up.status,
                    )
                    return _error(
                        502,
                        "sandbox proxy does not follow redirects from its upstream",
                        streaming=streaming,
                    )
                out = web.StreamResponse(
                    status=up.status, headers=relayed_response_headers(up.headers)
                )
                try:
                    # ``prepare`` can fail if the box disconnects while the
                    # upstream is still working.
                    await out.prepare(request)
                    async for chunk in up.content.iter_chunked(_CHUNK):
                        await out.write(chunk)
                    await out.write_eof()
                except (ConnectionResetError, BrokenPipeError) as e:
                    # Don't report a downstream disconnect as an upstream error.
                    log.debug("vertex proxy: downstream closed: %s", type(e).__name__)
                return out
        except ClientError as e:
            # Return a readable failure instead of a retryable disconnect.
            log.warning("vertex proxy: upstream %s unreachable: %s", upstream, e)
            return _error(
                502,
                f"sandbox proxy cannot reach its upstream: {e}",
                streaming=streaming,
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
