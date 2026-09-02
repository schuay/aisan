# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side Anthropic proxy: the box reaches a model holding only a placeholder.

Same two-half shape as `vertex.py` -- a UNIX socket the box's relay splices to, a
path allowlist, an injected credential, a streamed response -- and deliberately a
SEPARATE module, because one of the three properties that make `vertex.py` a
security boundary is false here:

    "Request headers are never passed through. The proxy builds its own, so
    anything the box sends is data."

That cannot hold for this upstream. Measured against the real Claude Code
(2.1.219): a request built from our own headers only is answered
`400 anthropic-version: header is required`, and the `anthropic-beta` list the
client sends selects API behaviour it then expects (`claude-code-*`,
`interleaved-thinking-*`, `context-management-*`, `effort-*`, ...). Those headers
are protocol, not payload. Reusing `make_app` would mean giving that stated
invariant a caller-supplied parameter -- a boundary whose strength depends on
what the last caller passed -- so the header policy lives here, as its own
allowlist, and vertex's "none, ever" stays absolute.

The header allowlist is therefore the new policy surface, and it is an
allowlist in both directions:

- FORWARDED: what the API needs to answer correctly. `anthropic-version` and
  `anthropic-beta` (protocol selectors, measured required), `accept` (the client
  asks for SSE and parses what it asked for). Plus `x-claude-code-session-id`,
  which is none of those and is explained on its own below.
- DROPPED, always: `x-api-key` and `authorization`. Whichever of the two the
  box was dressed to use, its value is a per-box relay token by construction, so
  forwarding it forwards nothing useful -- and a prompt-injected agent that sets
  one would be choosing the string we send upstream. The drop is unconditional
  and comes BEFORE the allowlist loop, so the credential this proxy attaches is
  the only one on the wire no matter which header the box presented. Also every hop-by-hop and body-framing header (`host`, `connection`,
  `content-length`, `accept-encoding`): the client session computes those for the
  connection it actually opens, and a forwarded copy describes the wrong one.
- DROPPED, by judgement: the `x-stainless-*` set (arch, os, lang, runtime,
  runtime-version, package-version, retry-count, timeout), plus `user-agent`
  and `x-app`. These are SDK telemetry. Six of them describe the MACHINE --
  operating system, architecture, runtime version -- and the point of the box
  is that it is not the host. None is required: verified by driving a real
  `claude -p` through this proxy with them dropped. Dropping is the
  fail-closed direction and it costs nothing measurable.

`x-claude-code-session-id` is forwarded, and it is the one entry here that buys
something at a cost rather than nothing at a cost. It is a client-generated
identifier for the session making the request. The API does not require it --
`claude -p` completes with it dropped -- but an upstream that keys PER-SESSION
state off it cannot tell two boxed sessions apart without it, and collapses
every one of them into a single bucket. That is the whole reason it is here: an
operator pointing this proxy at such an upstream gets per-session behaviour that
matches an unsandboxed client, which is the parity the box is otherwise trying
to preserve.

What it costs is a difference in KIND from the other three. Those are protocol:
box-chosen bytes whose effect is bounded by the API's own parser. An identifier
is bytes an upstream may use as a state key, so forwarding it means the box can
name a session it did not open -- writing its own state into another session's
bucket, on an upstream that trusts the id. It is not a credential and it does
not describe the host, which is what separates it from the set above; it is a
capability only against an upstream that treats client-supplied ids as
authoritative, and such an upstream has to namespace or sanitise them anyway.

The narrower rule -- forward it only for upstreams that want it -- is
deliberately NOT taken: it would make this allowlist depend on which upstream a
caller configured, and a boundary whose strength depends on the last caller is
the property this module exists to avoid.

Everything not named is dropped. A new header the client starts sending arrives
as "the API behaves slightly differently", which is a debuggable outcome; the
alternative -- a denylist -- arrives as "the box's chosen bytes reached the
upstream", which is not.

The third surface is the BODY, and it exists because the first two gate only the
envelope of a model call. `POST /v1/messages` is not a leaf capability on this
upstream: the tools its body declares make Anthropic's servers fetch URLs the
box chose, so `unshare_net` stops the box dialling out while leaving it able to
ask the API to dial out on its behalf. `BodyPolicy` answers that, on the same
allowlist posture as the headers -- see its docstring for what is measured.

Request-size limits, rate limiting, and Unix-socket serving come from the neutral
`http` module. This module retains only Anthropic protocol and policy behavior.
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
    load_json_unambiguous,
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

# Lowercased; aiohttp's CIMultiDict compares case-insensitively but a frozenset
# does not, so the lookup lowercases too.
FORWARD_HEADERS = frozenset(
    {
        "anthropic-version",
        "anthropic-beta",
        "accept",
        # Not protocol: an identifier, forwarded so a session-keyed upstream can
        # tell boxed sessions apart. The box chooses its value -- see the module
        # docstring for what that is and is not worth.
        "x-claude-code-session-id",
    }
)

# Observed on the wire from claude-cli 2.1.219 in a full one-shot turn: the
# messages call, and the reachability check it makes at startup. Anchored and
# exact -- a prefix match on "/v1/messages" would also permit "/v1/messages_evil"
# on an upstream we do not control.
ALLOWED_PATHS = (
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("GET", "/api/hello"),
    ("HEAD", "/api/hello"),
)


@dataclass(frozen=True)
class PathAllowlist:
    """The request shapes the box may reach on the upstream.

    Narrow on purpose. The credential this proxy attaches is a subscription
    bearer, which is accepted by more of the API than a model call needs -- so
    without this, a prompt-injected agent's route off the machine is also a route
    to whatever else that token opens. The allowlist is what makes the box's
    capability "talk to a model" rather than "act as the user".
    """

    routes: tuple[tuple[str, str], ...] = ALLOWED_PATHS

    def permits(self, method: str, path: str) -> bool:
        return (method.upper(), path) in self.routes


# Tool types that execute IN THE BOX. Everything else in the upstream's union is
# the upstream acting on the box's behalf, which `unshare_net` does not reach.
#
# Only `"custom"` is named here, and an ABSENT `type` is permitted separately in
# `refuse` -- absent is not a value and encoding it as one would conflate it with
# an explicit `null`, which is a different thing to the upstream (see below).
#
# The upstream's `tools` is a discriminated union keyed on `type`, and an absent
# `type` resolves to the `custom` variant. Measured, from the upstream's own
# validation errors: an untyped tool carrying web_fetch's `max_uses` is refused
# `tools.0.custom.max_uses: Extra inputs are not permitted`, naming the variant
# it resolved to. So absent and `"custom"` are one variant under two spellings.
#
# The API also defines client-side tools that DO carry a type (`bash_*`,
# `text_editor_*`, `memory_*`). They are deliberately absent: Claude Code 2.1.219
# declares none of them (measured, 646 declarations across chat, Bash, edits,
# Skill, subagent, plan mode and a TUI session -- every one untyped), and adding
# them here on the strength of the API reference alone is the guess this policy
# exists to avoid. If a client adopts one, this refuses its own turn loudly on
# the next upgrade, and the fix is one measured tag.
CLIENT_TOOL_TYPES = frozenset({"custom"})

# Top-level keys that make the upstream act. `mcp_servers` pairs with the
# `mcp_toolset` tool type and the upstream requires both together, so either
# check alone closes that path; both are refused because MCP is the one
# capability here that is bidirectional -- results come back INTO the
# conversation, and the far side executes.
REFUSED_KEYS = ("mcp_servers", "container")

# Top-level keys measured on the wire from claude-cli 2.1.246 across a full
# tool round trip (the model turn, the topic-detection call; count_tokens takes
# a subset). An allowlist, not a denylist, for the same reason as the tool
# types: a server-acting field need not declare itself as a tool --
# `mcp_servers` does not -- so a key this policy does not know arrives as a
# loud refusal naming itself, and the fix is one measured tag.
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

# Content block types that carry no URL for the upstream to fetch. Measured
# from the same round trip: text (and system's text blocks), tool_use and
# tool_result on the turn after a tool call, thinking when extended thinking
# is on; redacted_thinking is the API's own transform of thinking and rides
# with it. `image` and `document` are deliberately absent: their url sources
# make ANTHROPIC's servers fetch a URL the box chose -- the same route the
# tool-type gate closes -- and their base64 sources are refused with them
# until a measurement forces the finer split.
ALLOWED_CONTENT_TYPES = frozenset(
    {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}
)

# The two additions a box on the HOST'S network may make, and nothing else.
#
# `web_search_20250305` executes upstream: it makes Anthropic's servers issue a
# query the box chose, which is exactly the route the isolated policy closes.
# `tool_choice` rides with it -- measured on 2.1.246, Claude Code issues a
# DEDICATED single-shot request carrying only the server tool plus a tool_choice
# forcing it, and folds the answer back into the conversation as an ordinary
# `tool_result`. So no new content block type is needed, and the main
# conversation's shape is unchanged.
#
# Deliberately measured-only, like every other tag here. Claude Code's WebFetch
# is NOT in this set because it does not need to be: with the host's network the
# box fetches the URL itself and sends the model text (measured -- a WebFetch of
# example.com completes with the isolated policy in force). A server-side fetch
# tool would be a new tag, added when something is measured sending one.
SHARED_NET_TOOL_TYPES = CLIENT_TOOL_TYPES | {"web_search_20250305"}
SHARED_NET_KEYS = ALLOWED_KEYS | {"tool_choice"}


@dataclass(frozen=True)
class BodyPolicy:
    """Which capabilities the body may declare.

    The path and header allowlists gate the ENVELOPE of a model call. Neither
    reads the body, and on this upstream the body is not inert: `POST
    /v1/messages` is not a leaf capability, because the tools it declares make
    ANTHROPIC's servers issue requests to URLs the box chose. `--unshare-net`
    stops the box dialling out; it does not stop the box asking the API to dial
    out for it, and the URL IS the payload -- a fetch of
    `https://x.example/<bytes>` has already exfiltrated them whatever the
    response.

    So this is an allowlist of what runs in the box, not a denylist of what does
    not. A new server-side tool type is then a refusal rather than a hole, which
    is the same posture the header allowlist takes and for the same reason.

    Three gates, one posture: the top-level keys (a server-acting field need
    not declare itself as a tool), the tool types, and the message content
    block types (an `image` or `document` url source makes the upstream fetch
    a URL the box chose, and the URL is the payload). All three are measured
    from the real client, so a key or block it grows next year is a refusal
    naming itself rather than a hole.

    Bounded, and worth stating so the claim is not overread: this refuses a route
    off the machine for bytes the box already holds. It is not what keeps the
    credential out -- that is the absence of any bind naming it.

    Which is also why there are two of these. The gate is worth exactly what the
    box's OWN route off the machine is not worth: under `unshare_net` there is
    none, and this is the boundary. Without it the box is in the host's network
    namespace with the host's full connectivity (`explain` prints it in those
    words), so it can dial `https://x.example/<bytes>` directly, and refusing to
    let it ask the upstream to do the same protects nothing while breaking web
    search. `for_shared_network` is that second policy, and it is DERIVED, never
    passed: the backend builds it in `serve_shared`, which the Box calls only
    when the spec says the network is shared. No caller can ask the isolated
    policy to relax, which is the property this module insists on everywhere
    else -- a boundary whose strength depends on what the last caller passed is
    not a boundary.
    """

    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    refused_keys: tuple[str, ...] = REFUSED_KEYS
    allowed_keys: frozenset[str] = ALLOWED_KEYS
    content_types: frozenset[str] = ALLOWED_CONTENT_TYPES

    @classmethod
    def for_shared_network(cls) -> BodyPolicy:
        """The policy for a box that already has the host's network.

        Narrow on purpose: two tags, both measured. Everything else the strict
        policy refuses is still refused here -- `mcp_servers` and `container`,
        every other server-side tool type, and `image`/`document` blocks whose
        url source would make the upstream fetch for the box. A box with the
        host's network needs none of those to reach the network, so relaxing
        them would buy nothing and lose the refusal that names them.
        """
        return cls(client_types=SHARED_NET_TOOL_TYPES, allowed_keys=SHARED_NET_KEYS)

    def refuse(self, body: bytes) -> str | None:
        """The reason to refuse `body`, or None to permit it.

        An empty body is permitted -- the hello routes send none, and a request
        with nothing in it declares nothing. Anything else this parser cannot
        read as one unambiguous JSON object IS refused. The old leniency ("the
        upstream rejects malformed bodies itself") assumed the two parsers
        agree on what is malformed, and a streaming upstream decoder does not:
        it reads `{...}{}` value by value and honors tools in the first object,
        which this policy never inspected. A body this side cannot classify
        would leave the capability decision to the upstream.
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
                "field(s) not permitted by the sandbox proxy:"
                f" {', '.join(sorted(unknown))}"
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
            # Absent is checked as a MISSING KEY, not as a `None` value. The two
            # are different to the upstream -- absent selects `custom`, while an
            # explicit null is refused outright as `Input tag 'None'` -- so a
            # `.get()` that flattened them together would permit a null this
            # policy cannot resolve to any variant.
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
        """Message content as an egress channel, gated like the tools.

        Neither the key nor the tool gate reads message content, and content is
        not inert: an `image` or `document` block with a url source makes the
        UPSTREAM fetch a URL the box chose, and the URL is the payload --
        `openai_responses` closed exactly this on day one. `system` takes the
        same block shape and is walked too.
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

        A `tool_result` nests another content value of the same shape, which
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
            if kind == "tool_result":
                reason = self._blocks_refusal(block.get("content"))
                if reason is not None:
                    return reason
        return None


@dataclass(frozen=True)
class HeaderAllowlist:
    """Which of the box's request headers reach the upstream.

    A predicate per header rather than a filter over the whole mapping, so the
    fail-closed wrapper applies to each decision: a matcher that raises on one
    header drops that header instead of failing (or forwarding) the rest.
    """

    allow: frozenset[str] = FORWARD_HEADERS

    def permits(self, name: str) -> bool:
        return name.lower() in self.allow


def _error(status: int, kind: str, message: str) -> web.Response:
    """A refusal in the API's own error shape.

    Model-facing, like vertex's: this reaches the client and, for an agent, the
    transcript. So it names the policy that refused rather than the code that
    ran. The shape is Anthropic's (`{"type": "error", "error": {...}}`) so the
    SDK surfaces the message instead of failing to parse a body it did not
    expect -- the same lesson vertex's array-wrapped errors record, arrived at
    from the other end.
    """
    return web.json_response(
        {"type": "error", "error": {"type": kind, "message": message}}, status=status
    )


# Returns the bearer to attach. Called PER REQUEST, never captured: the host's
# own Claude Code owns that credential and may rewrite the file at any moment, so
# a value captured once would miss either its refresh or the backend's delegated
# host-side refresh.
TokenSource = Callable[[], Awaitable[str]]
# The headers that authenticate ONE request upstream. A callable rather than a
# value for the same reason `TokenSource` is: the credential is re-read per
# request, never captured. A dict rather than a bearer string because the header
# depends on the credential kind -- a subscription authenticates with
# `authorization`, a static key with `x-api-key` -- and the kind is the caller's
# to know. Same shape as `openai_compat.AuthorizationSource`.
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
        # Either location, because the box's dress depends on the credential
        # kind: an API-key-dressed client sends `x-api-key`, a subscription
        # dressed one sends `authorization` (measured). What is checked is the
        # VALUE -- a per-box secret -- so a second place to present it adds no
        # second principal, and both arms reject duplicates and compare in
        # constant time. Neither header is ever forwarded, whichever carried it.
        presented = request.headers.getall("x-api-key", [])
        api_key = presented[0] if len(presented) == 1 else None
        if not (
            token_matches(api_key, client_token)
            or token_matches(bearer_token(request), client_token)
        ):
            return _error(401, "authentication_error", "invalid aisan proxy token")
        # Through `policy_permits`, so an allowlist that RAISES denies rather
        # than tearing the connection down as a retryable transport error.
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

        # Before `token()`: no reason to read the credential for a request being
        # refused. Gating the body rather than the route also covers
        # `count_tokens` for free -- it takes the same tool declarations.
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
            # The credential went away mid-session, or host Claude could not
            # refresh it. Answered as an error the agent can read rather than a
            # traceback. Credential failures name paths and expiry, never token
            # values.
            log.error("anthropic proxy: no usable credential to attach: %s", e)
            return _error(
                503,
                "authentication_error",
                f"sandbox proxy has no usable credential: {e}",
            )

        # A list of pairs, not a dict: `anthropic-beta` is a list the client may
        # send as several headers, and a dict keyed by name would keep only the
        # last -- dropping betas the request depends on. aiohttp forwards an
        # iterable of pairs verbatim, so every one survives.
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
                request.method, url, data=body or None, headers=out
            ) as up:
                resp = web.StreamResponse(
                    status=up.status, headers=relayed_response_headers(up.headers)
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
                        "anthropic proxy: downstream closed: %s", type(e).__name__
                    )
                return resp
        except ClientError as e:
            # The upstream is the user's own local proxy, which they may restart
            # under a running box. A dead upstream is one turn's failure with a
            # legible reason, not a torn connection the client will retry into.
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
