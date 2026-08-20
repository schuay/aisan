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

Token-file brokering is what this replaces: there the box held a
short-lived token, here it holds nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, web

# Re-exported for compatibility; new shared-infrastructure imports should use
# `aisan.proxy.http` directly.
from .http import MAX_BODY_BYTES, RateLimit, run_forever, serve
from .policy import permits as policy_permits

__all__ = [
    "MAX_BODY_BYTES",
    "Allowlist",
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
) -> web.Application:
    limiter = rate or RateLimit()
    upstream = _upstream(location)

    async def handle(request: web.Request) -> web.StreamResponse:
        streaming = "streamGenerateContent" in request.path
        # Through `policy_permits`, so an allowlist that RAISES denies rather
        # than tearing the connection down as a retryable transport error.
        if not policy_permits(
            lambda: allowlist.permits(request.method, request.path),
            subject=f"{request.method} {request.path}",
        ):
            log.warning("vertex proxy: refused %s %s", request.method, request.path)
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

        # Our own headers only: nothing the box sent is forwarded.
        headers = {
            "Authorization": f"Bearer {await token()}",
            "Content-Type": "application/json",
        }
        session = request.app[_SESSION]
        url = f"{upstream}{request.path_qs}"
        async with session.request(
            request.method, url, data=body or None, headers=headers
        ) as up:
            out = web.StreamResponse(
                status=up.status,
                headers={
                    "Content-Type": up.headers.get("Content-Type", "application/json")
                },
            )
            await out.prepare(request)
            try:
                async for chunk in up.content.iter_chunked(_CHUNK):
                    await out.write(chunk)
                await out.write_eof()
            except (ConnectionResetError, BrokenPipeError) as e:
                # The box hung up mid-response -- the agent exited, or its turn
                # was cancelled. Ordinary, and nothing is lost: the only reader
                # of these bytes is already gone. Left to propagate it reaches
                # aiohttp's generic handler, which logs a full traceback at
                # ERROR -- the noise the disabled access log (see serve) exists
                # to prevent, and it buries the refusal warning that matters.
                log.debug("vertex proxy: downstream closed: %s", type(e).__name__)
            return out

    app = web.Application(client_max_size=MAX_BODY_BYTES + 1)
    app.router.add_route("*", "/{tail:.*}", handle)

    async def _open(a: web.Application) -> None:
        a[_SESSION] = ClientSession(timeout=_TIMEOUT)

    async def _close(a: web.Application) -> None:
        await a[_SESSION].close()

    app.on_startup.append(_open)
    app.on_cleanup.append(_close)
    return app
