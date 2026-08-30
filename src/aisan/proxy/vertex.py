# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side Vertex proxy: the box's only route to a model, and it holds no token.

The sandbox runs with no network and no credential. This listens on a UNIX
socket bound into the box (reached through `relay.py`), checks each request
against a fixed allowlist, attaches the real bearer, and streams Vertex's
response back.

The socket decides only WHO may talk to the proxy -- anything in the box, which
includes a prompt-injected agent. What the box may DO is the allowlist, so that
is the security boundary and the part worth reviewing. Three properties make it
one:

- The project, location and model come from CONFIG and are interpolated into the
  matcher. A request naming another project does not match, so it cannot be
  forwarded -- the box cannot reach a resource we did not name.
- Request headers are never passed through. The proxy builds its own, so
  anything the box sends is data.
- Two request shapes are reachable, both POST-or-DELETE on one host: model
  inference, and the cached-content lifecycle the growing-prefix cache needs.
- The body may declare only the tools that run IN the box. Neither of the above
  reads the body, and on this upstream the body is not inert: a declared
  `googleSearch` or `urlContext` makes Google fetch a URL the box chose, which
  is a route off a machine the box otherwise has no route off. See `BodyPolicy`.

Token-file brokering is what this replaces: there the box held a
short-lived token, here it holds nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, web

# Re-exported for compatibility; new shared-infrastructure imports should use
# `aisan.proxy.http` directly.
from .http import (
    MAX_BODY_BYTES,
    AmbiguousBody,
    LogGate,
    RateLimit,
    load_json_unambiguous,
    relayed_response_headers,
    request_path,
    run_forever,
    serve,
)
from .policy import permits as policy_permits
from .policy import refusal as policy_refusal

__all__ = [
    "MAX_BODY_BYTES",
    "Allowlist",
    "BodyPolicy",
    "RateLimit",
    "make_app",
    "run_forever",
    "serve",
]

log = logging.getLogger(__name__)

# Typed key rather than a bare string: aiohttp warns on the latter, and the
# session is the one piece of shared state the handler reaches for.
_SESSION: web.AppKey[ClientSession] = web.AppKey("session")


# Vertex is regional; "global" alone has no host prefix.
def _upstream(location: str) -> str:
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return f"https://{host}"


# Forwarded per read so streaming stays incremental. A proxy that accumulates is
# still byte-correct, which is exactly why this is a named constant and a test
# rather than an implementation detail.
_CHUNK = 64 << 10

_TIMEOUT = ClientTimeout(total=None, sock_read=600, sock_connect=30)


@dataclass(frozen=True)
class Allowlist:
    """The reachable request shapes, anchored to one project/location/model.

    Built from config, never from the request. `re.escape` on every interpolated
    value: a project id with a regex metacharacter would otherwise silently widen
    the pattern, which is the classic way an allowlist stops being one.
    """

    project: str
    location: str
    models: tuple[str, ...]

    def _patterns(self) -> tuple[re.Pattern[str], ...]:
        base = rf"/v1beta1/projects/{re.escape(self.project)}/locations/{re.escape(self.location)}"
        models = "|".join(re.escape(m) for m in self.models)
        return (
            re.compile(
                rf"^{base}/publishers/google/models/({models})"
                r":(generateContent|streamGenerateContent)$"
            ),
            re.compile(rf"^{base}/cachedContents$"),
            re.compile(rf"^{base}/cachedContents/[0-9]+$"),
        )

    def permits(self, method: str, path: str) -> bool:
        if method not in ("POST", "DELETE"):
            return False
        return any(p.match(path) for p in self._patterns())


# `Part` keys that name a URI for GOOGLE to fetch. A `contents` part carrying
# `fileData: {fileUri: ...}` makes Vertex read that URI on the box's behalf --
# the same route off the machine the tool fields open, one level down in the
# messages rather than the tool list, and the URL is the payload. Both proto3
# JSON spellings, as with the tool fields. inlineData (base64) is inert and
# permitted; text and the function/thought parts carry no fetch.
FETCH_PART_FIELDS = frozenset({"fileData", "file_data"})


# The one `Tool` field that executes IN THE BOX: the model emits a call, the
# client runs it, the result goes back as a message. BOTH spellings, because the
# upstream accepts both and an allowlist that knew only one is bypassed by
# sending the other -- proto3 JSON requires a parser to take the original
# snake_case field name as well as the lowerCamelCase form (protobuf.dev, JSON
# mapping), and the REST transport this proxy fronts emits the camel one
# (measured: `MessageToJson` defaults to `functionDeclarations`).
#
# Every OTHER field of the union is the upstream acting on the box's behalf, and
# they are deliberately not enumerated here: an allowlist means a field Google
# adds next year is refused rather than forwarded. As of the v1/v1beta1 discovery
# documents that union also holds googleSearch, googleSearchRetrieval, urlContext,
# codeExecution, retrieval, enterpriseWebSearch, googleMaps, computerUse,
# parallelAiSearch and exaAiSearch -- retrieval.externalApi being the sharpest of
# them, since it names an arbitrary endpoint for Google's servers to call.
CLIENT_TOOL_FIELDS = frozenset({"functionDeclarations", "function_declarations"})


@dataclass(frozen=True)
class BodyPolicy:
    """Which capabilities the body may declare.

    The path allowlist gates the ENVELOPE of a model call, and on this upstream
    the body is not inert: the tools it declares -- and the `contents` parts it
    carries -- make GOOGLE's servers fetch, search or execute on the box's
    behalf. `unshare_net` stops the box dialling out; it does not stop the box
    asking Vertex to dial out for it, and the URL IS the payload -- a fetch of
    `https://x.example/<bytes>` has exfiltrated them whatever comes back. Same
    posture as the Anthropic and OpenAI-compatible policies, one protocol over.

    Two surfaces, then: the tool fields (only functionDeclarations runs in the
    box), and the message contents (a `fileData.fileUri` part is the same fetch
    vector, one level down); inline text and base64 carry no URL and pass.

    Both routes the path allowlist reaches are gated, not just inference:
    `cachedContents` takes the same `tools` array (measured against the real
    client, which posts one), so a policy that read only `generateContent` would
    leave "cache the tool, then reference the cache" open.

    Bounded, and worth stating so the claim is not overread: this refuses a route
    off the machine for bytes the box already holds. It is not what keeps the
    credential out -- that is the absence of any bind naming it.
    """

    client_fields: frozenset[str] = CLIENT_TOOL_FIELDS

    def refuse(self, body: bytes) -> str | None:
        """The reason to refuse `body`, or None to permit it.

        An empty body is permitted: a cache DELETE carries none, and a request
        with nothing in it declares nothing. Anything else that this parser
        cannot read as one unambiguous JSON object IS refused -- unlike the
        Anthropic policy, which has an upstream whose leniency was measured.
        Here the only client is a REST SDK emitting `MessageToJson` output, so a
        body this cannot classify is not a disagreement about JSON, it is a
        request nobody legitimate sent.
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
        """Refuse a `contents` part that asks Google to fetch a URI.

        The tools gate covers the tool list; this covers the messages, where a
        `fileData.fileUri` is the same fetch-a-URL-the-box-chose vector. Shapes
        it cannot read are left to the tools/parser gates -- a non-list or a
        non-dict part is not a fetch this needs an opinion on."""
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


def _error(status: int, message: str, *, streaming: bool) -> web.Response:
    """A refusal the client can actually parse.

    Measured against real Vertex: a streamGenerateContent error body is a JSON
    ARRAY, and the SDK handler calls .get on it -- so returning a bare object
    surfaces as `AttributeError: 'list' object has no attribute 'get'` and hides
    the reason. A refused agent should see the refusal, not a type error.

    Which makes `message` a MODEL-FACING surface: it reaches the client and, for
    an agent, the transcript. So it names the policy that refused ("path not
    permitted by the sandbox proxy") rather than the code that ran, because the
    reader deciding what to do next cannot act on an internal hook name.
    sandbox-runtime uses the same user-facing refusal rule.
    """
    body: object = {
        "error": {"code": status, "message": message, "status": "PERMISSION_DENIED"}
    }
    if streaming:
        body = [body]
    return web.json_response(body, status=status)


# Returns the current bearer. Called PER REQUEST, never captured: jobs outlive
# tokens (hours vs ~40 min), so a value read once would strand a long job.
TokenSource = Callable[[], Awaitable[str]]


def make_app(
    *,
    allowlist: Allowlist,
    token: TokenSource,
    location: str,
    rate: RateLimit | None = None,
    body: BodyPolicy | None = None,
) -> web.Application:
    limiter = rate or RateLimit()
    body_policy = body or BodyPolicy()
    warn = LogGate()
    upstream = _upstream(location)

    async def handle(request: web.Request) -> web.StreamResponse:
        path = request_path(request)
        # The method is the segment after the last `:`; a substring test would
        # also fire on `...streamGenerateContentEvil`. Only used to shape the
        # error envelope (streaming errors are array-wrapped), but a check whose
        # comment says "streaming" should mean the streaming method.
        streaming = path.endswith(":streamGenerateContent")
        # Through `policy_permits`, so an allowlist that RAISES denies rather
        # than tearing the connection down as a retryable transport error.
        if not policy_permits(
            lambda: allowlist.permits(request.method, path),
            subject=f"{request.method} {path}",
        ):
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

        # Before `token()`: no reason to read the credential for a request being
        # refused. Through `policy_refusal` for the same reason the path check
        # goes through `policy_permits` -- a policy that raises must deny, not
        # tear the connection down as something the client will retry into.
        reason = policy_refusal(
            lambda: body_policy.refuse(body), subject=f"body of {request.path}"
        )
        if reason is not None:
            warn.warning(log, "vertex proxy: refused body: %s", reason)
            return _error(403, reason, streaming=streaming)

        # Our own headers only: nothing the box sent is forwarded.
        headers = {
            "Authorization": f"Bearer {await token()}",
            "Content-Type": "application/json",
        }
        session = request.app[_SESSION]
        url = f"{upstream}{request.path_qs}"
        try:
            async with session.request(
                request.method, url, data=body or None, headers=headers
            ) as up:
                out = web.StreamResponse(
                    status=up.status, headers=relayed_response_headers(up.headers)
                )
                try:
                    # `prepare` writes too, and is where an interrupt usually
                    # lands: the box hangs up while the upstream is still
                    # thinking, so the write that fails is the response's own
                    # header line, not a body chunk. The guard must cover it,
                    # because aiohttp raises ClientConnectionResetError for a
                    # write to a closing transport -- a ClientError by
                    # inheritance as well as a ConnectionResetError, so left to
                    # the outer ClientError branch a cancelled turn would be
                    # logged as a dead upstream and answered with a 502 written
                    # to a socket nobody is reading.
                    await out.prepare(request)
                    async for chunk in up.content.iter_chunked(_CHUNK):
                        await out.write(chunk)
                    await out.write_eof()
                except (ConnectionResetError, BrokenPipeError) as e:
                    # The box hung up mid-response -- the agent exited, or its
                    # turn was cancelled. Ordinary, and nothing is lost: the only
                    # reader of these bytes is already gone. Left to propagate it
                    # reaches aiohttp's generic handler, which logs a full
                    # traceback at ERROR -- the noise the disabled access log
                    # (see serve) exists to prevent, and it buries the refusal
                    # warning that matters.
                    log.debug("vertex proxy: downstream closed: %s", type(e).__name__)
                return out
        except ClientError as e:
            # The upstream is unreachable -- a dead route to Vertex. One turn's
            # failure with a legible reason, not a torn connection the client
            # will retry into.
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
