# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side proxy for the OpenAI-compatible family: one route, our bearer.

The third transport, and the one that generalises. opencode reaches most of its
providers (147 of the 186 in its catalog at last count) through
`@ai-sdk/openai-compatible`, and that whole family speaks ONE wire shape,
measured against opencode 1.18.18 driving a local stub:

    POST {baseURL}/chat/completions
    Authorization: Bearer <key>
    "stream": true      # in the BODY; streaming is not negotiated by header

`baseURL` is whatever the box is told, so the proxy presents the family's
canonical form -- a bare loopback origin, no path prefix -- and every
provider-specific fact (the real base URL, `.../api/coding/paas/v4` and
friends) is data on the BACKEND, not on the transport. The path that crosses
this allowlist is therefore the same for every provider in the family, and
adding one is a matter of naming its catalog entry, not of touching policy.

Like `vertex.py` and unlike `anthropic.py`, no request header is ever
forwarded. That is not parsimony: the measured request carries nothing
protocol-shaped that the upstream needs (`content-type` and `accept` are
recomputed for the proxy's own connection; `x-session-id` and `user-agent` are
telemetry describing a client the host half did not open). So the vertex
invariant holds here in its strong form -- "the proxy builds its own headers,
so anything the box sends is data" -- and the day a provider of this family
measures otherwise, the fix is a header policy in this module, exactly as
`anthropic.py` grew one when a measurement forced it.

The body policy is the same posture as anthropic's with one measured
difference: every tool opencode declares here is `{"type": "function"}` (all
ten of them, across a full turn and the title-generation call), and absent
`type` has no meaning in this protocol to measure -- the type is a required
field, so an absent one is a malformed declaration rather than a variant that
resolves client-side. Refused rather than guessed at: this transport fronts
147 upstreams and "some of them might accept it" is the reasoning an allowlist
exists to stop.

Errors are answered in the family's own envelope, `{"error": {"message",
"type"}}`, because the message is the field the client surfaces: measured in
both directions against opencode 1.18.18, an openai-shaped body prints its
message verbatim ("Error: dummy endpoint refusal") while an anthropic-shaped
one collapses to a generic "Unexpected server error" that hides the reason.
A refusal the agent cannot read is a refusal that did not happen.

Request-size limits, rate limiting, and Unix-socket serving come from the neutral
`http` module. This module retains only OpenAI-compatible protocol and policy
behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from .http import (
    MAX_BODY_BYTES,
    RateLimit,
    bearer_token,
    load_json_unambiguous,
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

# Observed on the wire from opencode 1.18.18 over @ai-sdk/openai-compatible:
# the one route a turn needs, plus the title-generation call that rides it.
# Anchored and exact, as in `anthropic.py` -- a prefix match on
# "/chat/completions" would also permit "/chat/completions_evil" on an upstream
# we do not control. The family's other routes (`/models`, `/embeddings`) are
# deliberately absent: nothing in a turn reads them, measured, and a route on
# this list is a capability granted, not a URL that happens to exist.
ALLOWED_PATHS = (("POST", "/chat/completions"),)


@dataclass(frozen=True)
class PathAllowlist:
    """The request shapes the box may reach on the upstream.

    Uniform across the whole family by construction: the in-box client is
    pointed at the proxy's bare origin, so the provider's path prefix never
    crosses the namespace -- it lives in the upstream URL the backend dials,
    where scoping by provider is a fact about the route rather than a rule the
    box could trip over. What this allowlist bounds is the capability: a model
    call, and nothing else the key opens.
    """

    routes: tuple[tuple[str, str], ...] = ALLOWED_PATHS

    def permits(self, method: str, path: str) -> bool:
        return (method.upper(), path) in self.routes


# Tool types that execute IN THE BOX. In this protocol the client-side tool is
# `"function"`: the model emits a call, the client runs it, the result goes
# back as a message. Everything else in the union -- `web_search`,
# `code_interpreter`, whatever a provider adds next -- makes the UPSTREAM act
# on the box's behalf, which `unshare_net` does not reach.
#
# An ABSENT type is refused, unlike anthropic's policy. There it was measured
# that absent resolves to the client-side variant; here `type` is a required
# field, this client always sends it explicitly (measured), and an absent one
# is malformed rather than a variant -- across 147 upstreams there is no one
# resolution to measure, and permitting on the guess is the denylist's failure
# mode wearing an allowlist's clothes.
CLIENT_TOOL_TYPES = frozenset({"function"})

# Top-level keys that make the upstream act on the box's behalf, refused
# because a model call needs none of them. Not every server-side capability
# declares itself as a tool type: `web_search_options` turns on hosted web
# search with no `tools` entry (and carries a free-text `user_location`), so the
# tool gate above never sees it. That is a route off a box `unshare_net` means
# to have none, which is why the denylist is not empty; one discovered next year
# is a tag added here, not a new branch in code.
REFUSED_KEYS: tuple[str, ...] = ("web_search_options",)


@dataclass(frozen=True)
class BodyPolicy:
    """Which capabilities the body may declare.

    Same reasoning as anthropic's, one protocol over: the path allowlist gates
    the envelope, and the body is not inert -- a declared tool type can make
    the upstream fetch, execute, or open sessions on the box's behalf. So this
    is an allowlist of what runs in the box, and a type the policy does not
    name is a refusal, which is the only direction in which a protocol that
    accretes capabilities fails safely.
    """

    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    container_types: frozenset[str] = frozenset()
    refused_keys: tuple[str, ...] = REFUSED_KEYS

    def refuse(self, body: bytes) -> str | None:
        """The reason to refuse `body`, or None to permit it.

        Malformed or ambiguous input is refused. The proxy and upstream are
        separate JSON parsers, potentially from different implementations; if
        this parser cannot determine the request's capability-bearing shape,
        forwarding it would let the upstream make the security decision.
        """
        try:
            payload = parse_json_object(body)
        except (TypeError, ValueError) as e:
            return f"request body is not unambiguous JSON: {e}"

        for key in self.refused_keys:
            if key in payload:
                return f"`{key}` is not permitted by the sandbox proxy"

        if "tools" not in payload:
            return None
        return self.refuse_tools(payload["tools"])

    def refuse_tools(self, tools: object) -> str | None:
        """The reason to refuse a tool array, including nested containers."""
        if not isinstance(tools, list):
            return "`tools` must be an array"
        pending = list(tools)
        while pending:
            tool = pending.pop()
            if not isinstance(tool, dict):
                return "every tool must be a JSON object"
            # A MISSING key, checked as missing. `.get()` would flatten
            # absent and explicit null into one answer, and the two are
            # different declarations -- null is a value this policy cannot
            # resolve to any variant, and it is refused as such below rather
            # than waved through as "not stated".
            if "type" not in tool:
                return (
                    "a tool with no `type` is not a declaration this policy"
                    " can resolve, and is not permitted"
                )
            kind = tool["type"]
            if not isinstance(kind, str):
                return "tool `type` must be a string"
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


def parse_json_object(body: bytes) -> dict[str, object]:
    """Parse one unambiguous JSON object for a body policy to inspect.

    The strictness -- repeated keys and lenient constants both refused -- lives
    in the shared `load_json_unambiguous`, so this policy and the anthropic and
    vertex policies read a body the same way.
    """
    payload = load_json_unambiguous(body)
    if not isinstance(payload, dict):
        raise TypeError("request body must be a JSON object")
    return payload


def _error(status: int, kind: str, message: str) -> web.Response:
    """A refusal in the family's own error envelope.

    `message` is the model-facing surface: measured, it is the field
    opencode prints when a request fails, so it names the policy that refused
    rather than the code that ran. The shape is `{"error": {...}}` because
    that is what this family parses -- see the module docstring for the
    measurement in both directions.
    """
    return web.json_response(
        {"error": {"message": message, "type": kind}}, status=status
    )


# Returns the bearer to attach. Called PER REQUEST, never captured, for the
# same reason as anthropic's: the credential file belongs to the HOST's
# opencode, which rewrites it whenever the user re-runs /connect, and a stale
# read is worth less than a fresh one.
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
    base = upstream.rstrip("/")

    async def handle(request: web.Request) -> web.StreamResponse:
        if not token_matches(bearer_token(request), client_token):
            return _error(401, "authentication_error", "invalid aisan proxy token")
        # Through `policy_permits`, so an allowlist that RAISES denies rather
        # than tearing the connection down as a retryable transport error.
        if not policy_permits(
            lambda: path_allow.permits(request.method, request.path),
            subject=f"{request.method} {request.path}",
        ):
            log.warning(
                "openai-compat proxy: refused %s %s", request.method, request.path
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

        # Before resolving authorization: no reason to read the credential for
        # a request being refused -- same ordering as anthropic's, for the same
        # reason.
        reason = policy_refusal(
            lambda: body_policy.refuse(body), subject=f"body of {request.path}"
        )
        if reason is not None:
            log.warning("openai-compat proxy: refused body: %s", reason)
            return _error(403, "invalid_request_error", reason)

        try:
            if authorization is not None:
                authorization_headers = await authorization()
            else:
                assert token is not None
                authorization_headers = {"Authorization": f"Bearer {await token()}"}
        except Exception as e:
            # The credential went away mid-session -- the user re-ran /connect,
            # or the file moved. Answered as an error the agent can read, with
            # a reason naming a path or a provider, never a key value.
            log.error("openai-compat proxy: no credential to attach: %s", e)
            return _error(503, "api_error", f"sandbox proxy has no credential: {e}")

        # Our own headers only: nothing the box sent is forwarded. See the
        # module docstring for why that invariant is strong here and what it
        # takes for a provider to change it.
        upstream_headers = headers(body) if headers is not None else {}
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
                request.method, url, data=body or None, headers=upstream_headers
            ) as up:
                resp = web.StreamResponse(
                    status=up.status,
                    headers={
                        "Content-Type": up.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                try:
                    # `prepare` writes too, and is where an interrupt usually
                    # lands: the box hangs up while the upstream is still
                    # thinking, so the write that fails is the response's own
                    # header line, not a body chunk.
                    await resp.prepare(request)
                    async for chunk in up.content.iter_chunked(_CHUNK):
                        await resp.write(chunk)
                    await resp.write_eof()
                except (ConnectionResetError, BrokenPipeError) as e:
                    # The box hung up -- the agent exited, or its turn was
                    # cancelled -- before or mid-response. Nothing is lost:
                    # the only reader of these bytes is already gone. The
                    # guard must cover `prepare`, because aiohttp raises
                    # ClientConnectionResetError for a write to a closing
                    # transport, a ClientError by inheritance as well as a
                    # ConnectionResetError -- left to the outer ClientError
                    # branch, a cancelled turn logged as a dead upstream and
                    # a 502 written to a socket nobody is reading.
                    log.debug(
                        "openai-compat proxy: downstream closed: %s", type(e).__name__
                    )
                return resp
        except ClientError as e:
            # A dead upstream is one turn's failure with a legible reason,
            # not a torn connection the client retries into.
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
