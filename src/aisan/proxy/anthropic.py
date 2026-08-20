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
  asks for SSE and parses what it asked for).
- DROPPED, always: `x-api-key` and `authorization`. The box's key is a
  placeholder by construction, so forwarding it forwards nothing useful -- and a
  prompt-injected agent that sets one would be choosing the string we send
  upstream. Also every hop-by-hop and body-framing header (`host`, `connection`,
  `content-length`, `accept-encoding`): the client session computes those for the
  connection it actually opens, and a forwarded copy describes the wrong one.
- DROPPED, by judgement: the `x-stainless-*` set (arch, os, lang, runtime,
  runtime-version, package-version, retry-count, timeout), plus `user-agent`,
  `x-app` and `x-claude-code-session-id`. These are SDK telemetry. Six of them
  describe the MACHINE -- operating system, architecture, runtime version -- and
  the point of the box is that it is not the host; the rest name a client
  session the host half did not open. None is required: verified by driving a
  real `claude -p` through this proxy with them dropped. Dropping is the
  fail-closed direction and it costs nothing measurable.

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

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from .http import MAX_BODY_BYTES, RateLimit, serve
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

    Bounded, and worth stating so the claim is not overread: this refuses a route
    off the machine for bytes the box already holds. It is not what keeps the
    credential out -- that is the absence of any bind naming it.
    """

    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    refused_keys: tuple[str, ...] = REFUSED_KEYS

    def refuse(self, body: bytes) -> str | None:
        """The reason to refuse `body`, or None to permit it.

        A body that is not JSON, or not an object, is PERMITTED here: this
        policy answers "which capabilities are declared", and a body with no
        declarations to read is not one it has an opinion on. The upstream
        rejects malformed bodies itself, and a proxy that second-guessed the
        parse would refuse requests on a disagreement about JSON rather than
        about capability.
        """
        try:
            payload = json.loads(body)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None

        for key in self.refused_keys:
            if key in payload:
                return f"`{key}` is not permitted by the sandbox proxy"

        tools = payload.get("tools")
        if not isinstance(tools, list):
            return None
        for tool in tools:
            if not isinstance(tool, dict):
                continue  # the upstream's schema check owns this, not ours
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
# a value read once goes stale in a way this process must not try to repair.
TokenSource = Callable[[], Awaitable[str]]


def make_app(
    *,
    token: TokenSource,
    upstream: str,
    paths: PathAllowlist | None = None,
    headers: HeaderAllowlist | None = None,
    body: BodyPolicy | None = None,
    rate: RateLimit | None = None,
) -> web.Application:
    path_allow = paths or PathAllowlist()
    header_allow = headers or HeaderAllowlist()
    body_policy = body or BodyPolicy()
    limiter = rate or RateLimit()
    base = upstream.rstrip("/")

    async def handle(request: web.Request) -> web.StreamResponse:
        # Through `policy_permits`, so an allowlist that RAISES denies rather
        # than tearing the connection down as a retryable transport error.
        if not policy_permits(
            lambda: path_allow.permits(request.method, request.path),
            subject=f"{request.method} {request.path}",
        ):
            log.warning("anthropic proxy: refused %s %s", request.method, request.path)
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
            log.warning("anthropic proxy: refused body: %s", reason)
            return _error(403, "permission_error", reason)

        try:
            bearer = await token()
        except Exception as e:
            # The credential went away mid-session -- the file was rewritten,
            # or its token expired past what preflight checked. Answered as an
            # error the agent can read rather than a traceback, and the reason
            # never carries the credential: `e` here is a file or a parse
            # failure, both of which name a path, not a value.
            log.error("anthropic proxy: no credential to attach: %s", e)
            return _error(
                503, "authentication_error", f"sandbox proxy has no credential: {e}"
            )

        out = {
            "authorization": f"Bearer {bearer}",
            "content-type": "application/json",
        }
        for name, value in request.headers.items():
            if policy_permits(
                lambda name=name: header_allow.permits(name), subject=f"header {name}"
            ):
                out[name.lower()] = value

        session = request.app[_SESSION]
        url = f"{base}{request.path_qs}"
        try:
            async with session.request(
                request.method, url, data=body or None, headers=out
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
