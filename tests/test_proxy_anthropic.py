# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Anthropic transport: two allowlists, and the credential the box never has.

`vertex.py` has one policy surface (which paths) because it forwards no headers
at all. This has two, and the second exists because a measurement forced it: real
Claude Code's `anthropic-version` and `anthropic-beta` select API behaviour, and
an upstream answered from our own headers only refuses the first request. So
"which headers" became a decision, and a decision on a security boundary needs
its adversarial cases written down.

Most of this file is therefore refusals and drops. The happy path is one test;
the rest are the things that must NOT reach the upstream -- above all the box's
own `x-api-key`, which is a placeholder by construction and would be an
attacker-chosen string if a prompt-injected agent set one.

Everything here talks to the proxy over its UNIX socket directly. The relay's
loopback half is `test_proxy.py`'s, and the claim that a REAL `claude` reaches a
model through the whole chain needs a real box -- `test_aisan_claude_code.py`.
"""

from __future__ import annotations

import json
import logging

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.proxy.anthropic import (
    BodyPolicy,
    HeaderAllowlist,
    PathAllowlist,
    make_app,
    serve,
)
from aisan.proxy.http import RateLimit


def test_path_allowlist_permits_exactly_what_a_turn_needs():
    a = PathAllowlist()
    assert a.permits("POST", "/v1/messages")
    assert a.permits("POST", "/v1/messages/count_tokens")
    assert a.permits("GET", "/api/hello")
    assert a.permits("HEAD", "/api/hello")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        # The prefix trap an unanchored match would fall into, on an upstream we
        # do not control.
        ("POST", "/v1/messages_evil", "prefix"),
        ("POST", "/v1/messages/", "trailing slash"),
        # The subscription bearer opens far more of the API than a model call
        # needs. These are what the allowlist is FOR: the box's capability is
        # "talk to a model", not "act as the user".
        ("GET", "/v1/organizations/me", "not a model call"),
        ("POST", "/v1/files", "not a model call"),
        ("GET", "/v1/messages", "wrong method"),
        ("DELETE", "/v1/messages", "wrong method"),
    ],
)
def test_path_allowlist_refuses(method, path, why):
    assert not PathAllowlist().permits(method, path), why


def test_header_allowlist_forwards_protocol_and_drops_everything_else():
    h = HeaderAllowlist()
    # Protocol: measured required against the real API.
    assert h.permits("anthropic-version")
    assert h.permits("Anthropic-Version")  # the client sends mixed case
    assert h.permits("anthropic-beta")
    assert h.permits("accept")
    # Credentials: always ours, never the box's.
    assert not h.permits("x-api-key")
    assert not h.permits("authorization")
    # An identifier, not protocol: forwarded so a session-keyed upstream can
    # tell two boxed sessions apart, at the cost of the box choosing the value.
    assert h.permits("x-claude-code-session-id")
    # Telemetry describing a machine that is not the host.
    assert not h.permits("x-stainless-os")
    assert not h.permits("x-stainless-runtime-version")
    assert not h.permits("user-agent")
    assert not h.permits("x-app")
    # Framing the host half's own connection computes for itself.
    assert not h.permits("host")
    assert not h.permits("content-length")


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


class _Proxy:
    """The proxy on a socket, plus a session that dials it. Async CM."""

    def __init__(self, tmp_path, **kwargs) -> None:
        self._sock = tmp_path / "anthropic.sock"
        self._kwargs = kwargs

    async def __aenter__(self):
        self._runner = await serve(self._sock, make_app(**self._kwargs))
        self._session = ClientSession(connector=UnixConnector(path=str(self._sock)))
        return self._session

    async def __aexit__(self, *exc) -> None:
        await self._session.close()
        await self._runner.cleanup()


# The socket connector needs a host in the URL and ignores it. Named so a reader
# does not go looking for what resolves it.
URL = "http://anthropic.invalid"


async def _token(value: str = "real-bearer") -> str:
    return value


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-api-key": "wrong"},
        [("x-api-key", "expected"), ("x-api-key", "expected")],
    ],
)
async def test_shared_proxy_requires_one_exact_x_api_key(tmp_path, headers):
    async with (
        _Proxy(
            tmp_path,
            token=_token,
            upstream="http://127.0.0.1:1",
            client_token="expected",
        ) as session,
        session.post(f"{URL}/v1/messages", data=b"{}", headers=headers) as response,
    ):
        assert response.status == 401
        body = await response.json()
    assert body["error"]["type"] == "authentication_error"


async def test_the_boxs_key_is_dropped_and_the_real_bearer_attached(tmp_path):
    """The whole design, in one request.

    The box holds a placeholder; the upstream must see the credential and no
    trace of what the box presented. A test that only checked the bearer would
    pass while the placeholder rode along beside it.
    """
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(
                f"{URL}/v1/messages",
                data=b"{}",
                headers={
                    "x-api-key": "aisan-placeholder-not-a-credential",
                    "authorization": "Bearer smuggled-by-the-box",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "claude-code-20250219",
                    "x-stainless-os": "Linux",
                    "user-agent": "claude-cli/2.1.219",
                    "x-claude-code-session-id": "a-session-the-box-opened",
                },
            ) as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == "Bearer real-bearer"
    assert "x-api-key" not in got
    assert got["anthropic-version"] == "2023-06-01"
    assert got["anthropic-beta"] == "claude-code-20250219"
    assert "x-stainless-os" not in got
    # Forwarded VERBATIM, which is the trade: an upstream keying state off it
    # can tell boxed sessions apart, and the value is whatever the box sent.
    assert got["x-claude-code-session-id"] == "a-session-the-box-opened"
    # aiohttp sets its own user-agent on the outbound request, so the assertion
    # is that the BOX's did not survive, not that the header is absent.
    assert got.get("user-agent", "") != "claude-cli/2.1.219"


async def test_a_refused_path_never_reaches_the_upstream(tmp_path):
    """Refused before forwarding, not merely refused. The allowlist would be
    decorative if the request had already been made by the time it answered."""
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.get(f"{URL}/v1/organizations/me") as r,
        ):
            assert r.status == 403
            body = await r.json()
    finally:
        await up_runner.cleanup()

    assert reached == []
    # The API's own error shape: the SDK surfaces `error.message` and fails to
    # parse anything else, so a refusal in the wrong shape is a refusal the
    # agent cannot read.
    assert body["type"] == "error"
    assert body["error"]["type"] == "permission_error"
    assert "not permitted" in body["error"]["message"]


async def test_a_path_allowlist_that_raises_denies(tmp_path, caplog):
    """`policy.permits` is the fail-closed wrapper, and this is what it buys:
    a buggy predicate is an outage, not an approval nobody gave."""

    class Boom(PathAllowlist):
        def permits(self, method: str, path: str) -> bool:
            raise RuntimeError("bad matcher")

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        with caplog.at_level(logging.ERROR):
            async with _Proxy(tmp_path, token=_token, upstream=up, paths=Boom()) as s:
                async with s.post(f"{URL}/v1/messages", data=b"{}") as r:
                    assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert "policy check raised" in caplog.text


async def test_a_header_allowlist_that_raises_drops_only_that_header(tmp_path):
    """Per-header, so one bad decision costs one header rather than the request.

    The alternative -- a filter over the whole mapping inside one `permits` call
    -- fails the turn on a matcher bug, which is a worse outcome than dropping a
    protocol header and getting a legible 400 from the upstream.
    """
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    class BoomOnBeta(HeaderAllowlist):
        def permits(self, name: str) -> bool:
            if name.lower() == "anthropic-beta":
                raise RuntimeError("bad matcher")
            return super().permits(name)

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up, headers=BoomOnBeta()) as s,
            s.post(
                f"{URL}/v1/messages",
                data=b"{}",
                headers={
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "claude-code-20250219",
                },
            ) as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert "anthropic-beta" not in got
    assert got["anthropic-version"] == "2023-06-01"
    assert got["authorization"] == "Bearer real-bearer"


async def test_the_token_is_read_per_request_not_captured(tmp_path):
    """The host's own Claude Code rewrites the credential file on its own
    schedule, so a value captured at startup goes stale inside one session --
    while a re-read picks the new one up for free."""
    seen: list[str] = []
    tokens = iter(["t-1", "t-2", "t-3"])

    async def upstream(request: web.Request) -> web.Response:
        seen.append(request.headers["authorization"])
        return web.json_response({"ok": True})

    async def rotating() -> str:
        return next(tokens)

    up, up_runner = await _upstream_server(upstream)
    try:
        async with _Proxy(tmp_path, token=rotating, upstream=up) as s:
            for _ in range(3):
                async with s.post(f"{URL}/v1/messages", data=b"{}") as r:
                    assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert seen == ["Bearer t-1", "Bearer t-2", "Bearer t-3"]


async def test_a_credential_that_cannot_be_read_is_an_error_not_a_traceback(
    tmp_path,
):
    """The file went away mid-session. The agent gets a reason it can act on,
    and the reason names a path -- there is no value to leak here by
    construction, but the assertion pins that."""

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def gone() -> str:
        raise FileNotFoundError("/nonexistent/.credentials.json")

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=gone, upstream=up) as s,
            s.post(f"{URL}/v1/messages", data=b"{}") as r,
        ):
            assert r.status == 503
            body = await r.json()
    finally:
        await up_runner.cleanup()
    assert body["error"]["type"] == "authentication_error"
    assert ".credentials.json" in body["error"]["message"]


async def test_a_dead_upstream_is_502_not_a_torn_connection(tmp_path):
    """The upstream is the user's OWN local proxy, which they may restart under
    a running box. One turn fails with a legible reason; the client is not left
    retrying into a connection that was reset."""

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()  # the address is now dead, deliberately

    async with (
        _Proxy(tmp_path, token=_token, upstream=up) as s,
        s.post(f"{URL}/v1/messages", data=b"{}") as r,
    ):
        assert r.status == 502
        body = await r.json()
    assert "cannot reach its upstream" in body["error"]["message"]


async def test_the_rate_limit_refuses_in_the_apis_own_shape(tmp_path):
    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with _Proxy(
            tmp_path, token=_token, upstream=up, rate=RateLimit(per_minute=1)
        ) as s:
            async with s.post(f"{URL}/v1/messages", data=b"{}") as first:
                assert first.status == 200
            async with s.post(f"{URL}/v1/messages", data=b"{}") as second:
                assert second.status == 429
                body = await second.json()
    finally:
        await up_runner.cleanup()
    assert body["error"]["type"] == "rate_limit_error"


async def test_the_response_body_is_streamed_back_intact(tmp_path):
    """An SSE turn is many chunks, and the client parses them as they arrive.
    Byte-correctness across the chunk boundary is what makes the proxy
    invisible to it."""
    payload = b"".join(f"event: {i}\ndata: {'x' * 999}\n\n".encode() for i in range(64))

    async def upstream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(0, len(payload), 512):
            await resp.write(payload[i : i + 512])
        await resp.write_eof()
        return resp

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(
                f"{URL}/v1/messages",
                data=b"{}",
                headers={"accept": "text/event-stream"},
            ) as r,
        ):
            assert r.status == 200
            assert r.headers["Content-Type"] == "text/event-stream"
            got = await r.read()
    finally:
        await up_runner.cleanup()
    assert got == payload


# --- the body as an egress channel -----------------------------------------
#
# The path and header allowlists gate the envelope; these gate what the body
# asks the UPSTREAM to do. Measured against claude-cli 2.1.219 and the real API
# (see EGRESS-BODY.md): `tools` is a discriminated union keyed on `type`, an
# absent `type` resolves to the `custom` variant, and the client declares 646
# tools across every mode without one typed declaration among them.

# The shape a real turn sends, reduced to what the policy reads. Kept as a
# literal so a policy that started refusing real traffic fails here rather than
# in a session.
_REAL_TOOL = {
    "name": "Bash",
    "description": "Executes a bash command",
    "input_schema": {"type": "object", "properties": {}},
}


def test_body_policy_permits_a_real_claude_code_body():
    """The negative control. A policy that refuses everything would pass every
    other test in this section."""
    body = json.dumps(
        {
            "model": "claude-opus-4",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [{"type": "text", "text": "You are Claude Code"}],
            "tools": [_REAL_TOOL, {**_REAL_TOOL, "name": "Read"}],
            "metadata": {"user_id": "x"},
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "stream": True,
            "context_management": {"edits": []},
            "output_config": {},
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


@pytest.mark.parametrize(
    ("kind", "why"),
    [
        # The two whose whole purpose is to make the upstream fetch a URL. The
        # URL is the payload: the request having been made IS the exfiltration,
        # so the verb being a read constrains nothing.
        ("web_fetch_20250910", "upstream fetches a URL"),
        ("web_search_20250305", "upstream issues a query"),
        # Newer versions of the same capability. The reason this is an allowlist
        # of client types rather than a denylist of these: both of these tags
        # were live on the upstream and newer than anything the API reference
        # showed, so a hand-written denylist was already short on day one.
        ("web_fetch_20260318", "a version no denylist would have listed"),
        ("web_search_20260318", "a version no denylist would have listed"),
        ("code_execution_20260521", "runs in a container upstream"),
        # Bidirectional: results come back INTO the conversation and the far
        # side executes.
        ("mcp_toolset", "opens a session with a named server"),
        # The client-side typed family. Refused deliberately -- this client
        # declares none of them, and allowing them on the API reference alone is
        # the guess the policy exists to avoid.
        ("bash_20250124", "client-side, but unmeasured on this client"),
        ("memory_20250818", "client-side, but unmeasured on this client"),
        # Not in the upstream's union at all. Fails closed, which is the whole
        # point of naming what runs here rather than what does not.
        ("some_tool_type_invented_next_year", "unknown to us and to the upstream"),
    ],
)
def test_body_policy_refuses_tools_that_execute_upstream(kind, why):
    body = json.dumps({"tools": [{"type": kind, "name": "t"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None, why
    assert kind in reason  # the refusal names what it refused


def test_body_policy_permits_the_two_spellings_of_the_client_variant():
    """Absent and "custom" are one variant, not two rules.

    Measured from the upstream's own validation errors: an untyped tool is
    reported as `tools.0.custom.*`, so absent SELECTS custom. Refusing the
    explicit spelling would be a false positive waiting for the day the client
    starts sending it.
    """
    assert BodyPolicy().refuse(json.dumps({"tools": [_REAL_TOOL]}).encode()) is None
    explicit = {**_REAL_TOOL, "type": "custom"}
    assert BodyPolicy().refuse(json.dumps({"tools": [explicit]}).encode()) is None


def test_body_policy_refuses_an_explicit_null_type():
    """Absent and null are different to the upstream -- absent selects `custom`,
    null is refused outright as `Input tag 'None'` -- so null can never reach a
    server-side variant. Refused here as the fail-closed reading of a type this
    policy cannot resolve, rather than silently treated as absent."""
    body = json.dumps({"tools": [{**_REAL_TOOL, "type": None}]}).encode()
    assert BodyPolicy().refuse(body) is not None


@pytest.mark.parametrize("key", ["mcp_servers", "container"])
def test_body_policy_refuses_upstream_keys(key):
    body = json.dumps({"model": "m", key: [{"name": "s"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None
    assert key in reason


def test_body_policy_ignores_bodies_it_has_no_opinion_on():
    """Not JSON, not an object, and no tools: all permitted.

    This policy answers "which capabilities are declared". A body with no
    declarations to read is not one it has a view on, and a proxy that
    second-guessed the parse would refuse on a disagreement about JSON rather
    than about capability -- the upstream rejects malformed bodies itself.
    """
    p = BodyPolicy()
    assert p.refuse(b"") is None
    assert p.refuse(b"not json at all") is None
    assert p.refuse(b'["an", "array"]') is None
    assert p.refuse(b"{}") is None
    assert p.refuse(json.dumps({"tools": "not a list"}).encode()) is None
    assert p.refuse(json.dumps({"tools": ["not an object"]}).encode()) is None


def test_body_policy_refuses_duplicate_keys_at_any_depth():
    """A repeated key resolves one way here and possibly another upstream.

    Python keeps the LAST value, so a benign `custom` declaration written after
    a server-side one is what this policy would inspect while a first-wins
    upstream runs the tool it never saw. Both spellings of the seam are refused:
    a `tools` stated twice, and a `type` stated twice inside one declaration.
    """
    bodies = [
        b'{"tools": [{"type": "web_search_20250305"}], "tools": [{"type": "custom"}]}',
        b'{"tools": [{"type": "web_search_20250305", "type": "custom"}]}',
        b'{"mcp_servers": [{"url": "https://x.test"}], "mcp_servers": []}',
    ]
    for body in bodies:
        reason = BodyPolicy().refuse(body)
        assert reason is not None, body
        assert "unambiguous" in reason


def test_body_policy_still_has_no_opinion_on_a_body_it_cannot_read():
    """The leniency above it is deliberate and stays.

    Refusing a body that parses two ways is a decision about CAPABILITY -- the
    upstream may act on a declaration this side never inspected. A body that
    does not parse at all presents no such disagreement: the upstream rejects
    it, and refusing here would be this proxy second-guessing a parse rather
    than gating a capability.
    """
    assert BodyPolicy().refuse(b"{not json at all") is None


async def test_a_refused_body_never_reaches_the_upstream_or_the_credential(tmp_path):
    """Both halves of the claim, and the second is the easy one to lose.

    The check runs before `token()`, so a refused request does not cause the
    host's credential file to be read at all -- there is no reason to handle a
    secret for a request that is not going anywhere.
    """
    reached, tokens_read = [], []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    async def counting_token() -> str:
        tokens_read.append(1)
        return "real-bearer"

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=counting_token, upstream=up) as s,
            s.post(
                f"{URL}/v1/messages",
                data=json.dumps(
                    {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]}
                ).encode(),
            ) as r,
        ):
            assert r.status == 403
            payload = await r.json()
    finally:
        await up_runner.cleanup()

    assert reached == []
    assert tokens_read == []
    assert payload["error"]["type"] == "permission_error"
    assert "web_fetch_20250910" in payload["error"]["message"]


async def test_the_body_check_covers_count_tokens_too(tmp_path):
    """Gating the body rather than the route is what buys this: `count_tokens`
    takes the same tool declarations, and a route-shaped check would have needed
    to name it separately."""
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(
                f"{URL}/v1/messages/count_tokens",
                data=json.dumps({"mcp_servers": [{"name": "s"}]}).encode(),
            ) as r,
        ):
            assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert reached == []


async def test_a_real_body_still_reaches_the_upstream(tmp_path):
    """The end-to-end negative control: the policy is in the request path and a
    legitimate turn is unaffected by it."""
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(await request.read())
        return web.json_response({"ok": True})

    payload = json.dumps({"model": "claude-opus-4", "tools": [_REAL_TOOL]}).encode()
    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(f"{URL}/v1/messages", data=payload) as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()
    # Forwarded verbatim: the policy refuses or permits, it never edits.
    assert reached == [payload]


async def test_a_body_policy_that_raises_denies(tmp_path, caplog):
    """Same fail-closed contract as the path allowlist, over a check that
    returns a reason rather than a bool -- `policy.refusal` rather than
    `policy.permits`. A policy bug is an outage, not an approval nobody gave."""
    reached = []

    class Boom(BodyPolicy):
        def refuse(self, body: bytes) -> str | None:
            raise RuntimeError("bad policy")

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        with caplog.at_level(logging.ERROR):
            async with (
                _Proxy(tmp_path, token=_token, upstream=up, body=Boom()) as s,
                s.post(f"{URL}/v1/messages", data=b"{}") as r,
            ):
                assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert reached == []
    assert "policy check raised" in caplog.text


async def test_an_interrupted_turn_is_not_an_upstream_outage(tmp_path, caplog):
    """The interrupt window BEFORE the first response byte, faithfully.

    The client aborts its fetch while the upstream is still thinking -- the
    upstream holds the request and has answered nothing -- so the proxy's
    first write to the closing socket is the response's own header line in
    `prepare`, not a body chunk. aiohttp raises ClientConnectionResetError
    there, a ClientError by inheritance as well as a ConnectionResetError,
    and unguarded that write surfaced as "upstream ... unreachable: Cannot
    write to closing transport" -- a cancelled turn logged as a dead
    upstream, plus a 502 written to a socket nobody is reading.
    """
    import asyncio
    import logging

    arrived = asyncio.Event()

    async def upstream(request: web.Request) -> web.Response:
        arrived.set()
        await asyncio.sleep(0.3)  # the turn is "thinking"; outlive the client
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    sock = tmp_path / "a.sock"
    try:
        with caplog.at_level(logging.DEBUG):
            runner = await serve(sock, make_app(token=_token, upstream=up))
            try:
                _r, w = await asyncio.open_unix_connection(str(sock))
                w.write(
                    b"POST /v1/messages HTTP/1.1\r\n"
                    b"Host: anthropic.invalid\r\n"
                    b"Content-Length: 2\r\n\r\n{}"
                )
                await w.drain()
                await asyncio.wait_for(arrived.wait(), 5)
                w.transport.abort()  # the interrupt, before any response byte
                await asyncio.sleep(0.6)  # let the upstream answer into the void
            finally:
                await runner.cleanup()
    finally:
        await up_runner.cleanup()

    assert not [rec for rec in caplog.records if "unreachable" in rec.getMessage()], (
        "a cancelled turn logged as an upstream outage"
    )
    # The disconnect is attributed to the leg that broke.
    assert any("downstream closed" in rec.getMessage() for rec in caplog.records)
