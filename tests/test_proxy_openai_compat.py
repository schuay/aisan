# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The OpenAI-compatible transport: one allowlisted route, our bearer, the
family's own error envelope.

`anthropic.py`'s suite documents two policy surfaces because that transport
forwards protocol headers. This one forwards none -- vertex's strong invariant
-- so the adversarial cases here are the route, the body's tool declarations,
and the error SHAPE, which is load-bearing rather than cosmetic: measured
against opencode 1.18.18, an openai-shaped error body surfaces its message
verbatim while any other shape collapses to a generic "Unexpected server
error" that hides the refusal from the agent.

Everything here talks to the proxy over its UNIX socket directly. The relay's
loopback half is `test_proxy.py`'s, and the claim that a REAL opencode reaches
a model through the whole chain needs a real box -- `test_aisan_opencode.py`.
"""

from __future__ import annotations

import json
import logging

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.proxy.http import RateLimit
from aisan.proxy.openai_compat import (
    BodyPolicy,
    PathAllowlist,
    make_app,
    serve,
)


def test_path_allowlist_permits_exactly_what_a_turn_needs():
    """One route: the chat completion. Measured on the wire from opencode
    1.18.18 -- the turn and the title-generation call that rides it both post
    here, and nothing else in a session does."""
    assert PathAllowlist().permits("POST", "/chat/completions")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        # The prefix trap an unanchored match would fall into, on an upstream
        # we do not control.
        ("POST", "/chat/completions_evil", "prefix"),
        ("POST", "/chat/completions/", "trailing slash"),
        # Routes the key opens but a turn never reads. Measured absent across
        # a full session; present on the list they would be capabilities
        # granted, not URLs that happen to exist.
        ("GET", "/models", "catalog listing"),
        ("POST", "/embeddings", "not a model call"),
        ("POST", "/v1/chat/completions", "the prefix is host-side knowledge"),
        ("GET", "/chat/completions", "wrong method"),
        ("DELETE", "/chat/completions", "wrong method"),
    ],
)
def test_path_allowlist_refuses(method, path, why):
    assert not PathAllowlist().permits(method, path), why


# --- the body as an egress channel -----------------------------------------
#
# The path allowlist gates the envelope; this gates what the body asks the
# UPSTREAM to do. Measured against opencode 1.18.18: every tool the client
# declares is `{"type": "function", "function": {...}}` (all ten, across a
# full turn and the title-generation call), and `type` is a required field in
# this protocol -- unlike anthropic's, where an absent type resolves to the
# client-side variant.

# The shape a real turn sends, reduced to what the policy reads. Kept as a
# literal so a policy that started refusing real traffic fails here rather
# than in a session.
_REAL_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Executes a bash command",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_body_policy_permits_a_real_opencode_body():
    """The negative control. A policy that refuses everything would pass every
    other test in this section."""
    body = json.dumps(
        {
            "model": "glm-5.2",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 1.0,
            "reasoning_effort": "medium",
            "thinking": {"type": "enabled"},
            "tool_choice": "auto",
            "tools": [
                _REAL_TOOL,
                {**_REAL_TOOL, "function": {**_REAL_TOOL["function"], "name": "read"}},
            ],
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


@pytest.mark.parametrize(
    ("kind", "why"),
    [
        # Both make the upstream act on the box's behalf; `unshare_net` stops
        # the box dialling out, not the box asking the API to dial for it.
        ("web_search", "upstream issues a query"),
        ("code_interpreter", "runs in a container upstream"),
        # Not in the family's union at all. Fails closed, which is the point
        # of naming what runs in the box rather than what does not -- this
        # family has 147 upstreams and they accrete capabilities independently.
        ("some_tool_type_invented_next_year", "unknown to us and to the upstream"),
    ],
)
def test_body_policy_refuses_tools_that_execute_upstream(kind, why):
    body = json.dumps({"tools": [{"type": kind, "name": "t"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None, why
    assert kind in reason  # the refusal names what it refused


def test_body_policy_refuses_web_search_options():
    """A server-acting capability that is NOT a tool: `web_search_options` turns
    on hosted web search with no `tools` entry, so the tool-type gate never sees
    it -- a route off a box `unshare_net` means to have none. It is named in the
    top-level denylist, and the refusal names the key."""
    body = json.dumps(
        {"web_search_options": {"user_location": {"country": "US"}}, "messages": []}
    ).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None
    assert "web_search_options" in reason


def test_body_policy_refuses_an_absent_type():
    """The measured difference from anthropic's policy. There, absent was
    measured to resolve to the client-side variant; here `type` is a required
    field, this client always sends it, and an absent one is a malformed
    declaration with no variant to resolve to -- across a family of upstreams
    there is no one answer to measure, so it is refused rather than guessed
    at."""
    tool = dict(_REAL_TOOL)
    del tool["type"]
    reason = BodyPolicy().refuse(json.dumps({"tools": [tool]}).encode())
    assert reason is not None
    assert "no `type`" in reason


def test_body_policy_refuses_an_explicit_null_type():
    """Absent and null are different declarations, and null is not one this
    policy can resolve to any variant -- refused as the fail-closed reading,
    exactly as in anthropic's policy."""
    body = json.dumps({"tools": [{**_REAL_TOOL, "type": None}]}).encode()
    assert BodyPolicy().refuse(body) is not None


def test_body_policy_permits_valid_objects_without_tools():
    p = BodyPolicy()
    assert p.refuse(b"{}") is None
    assert p.refuse(json.dumps({"messages": []}).encode()) is None


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json at all",
        b'["an", "array"]',
        b'{"tools": "not a list"}',
        b'{"tools": ["not an object"]}',
        b'{"tools": [{"type": 1}]}',
        b'{"temperature": NaN}',
    ],
)
def test_body_policy_refuses_shapes_it_cannot_classify(body):
    assert BodyPolicy().refuse(body) is not None


def test_body_policy_refuses_duplicate_keys_at_any_depth():
    # A first-wins upstream and Python's default last-wins parser would inspect
    # different tool types. Refusing duplicates removes that parser seam.
    bodies = [
        b'{"tools": [], "tools": [{"type": "web_search"}]}',
        b'{"tools": [{"type": "web_search", "type": "function"}]}',
    ]
    assert all(BodyPolicy().refuse(body) is not None for body in bodies)


# --- the proxy end to end over its socket -----------------------------------


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
        self._sock = tmp_path / "openai.sock"
        self._kwargs = kwargs

    async def __aenter__(self):
        self._runner = await serve(self._sock, make_app(**self._kwargs))
        self._session = ClientSession(connector=UnixConnector(path=str(self._sock)))
        return self._session

    async def __aexit__(self, *exc) -> None:
        await self._session.close()
        await self._runner.cleanup()


# The socket connector needs a host in the URL and ignores it. Named so a
# reader does not go looking for what resolves it.
URL = "http://openai.invalid"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic expected"},
        {"Authorization": "Bearer wrong"},
        [("Authorization", "Bearer expected"), ("Authorization", "Bearer expected")],
    ],
)
async def test_shared_proxy_requires_one_exact_bearer(tmp_path, headers):
    async with (
        _Proxy(
            tmp_path,
            token=_token,
            upstream="http://127.0.0.1:1",
            client_token="expected",
        ) as session,
        session.post(
            f"{URL}/chat/completions", data=b"{}", headers=headers
        ) as response,
    ):
        assert response.status == 401
        body = await response.json()
    assert body["error"]["type"] == "authentication_error"


async def _token(value: str = "real-key") -> str:
    return value


def test_proxy_requires_exactly_one_host_authorization_source():
    async def authorization() -> dict[str, str]:
        return {"Authorization": "Bearer real-key"}

    with pytest.raises(ValueError, match="exactly one"):
        make_app(upstream="https://example.test")
    with pytest.raises(ValueError, match="exactly one"):
        make_app(
            token=_token,
            authorization=authorization,
            upstream="https://example.test",
        )


async def test_the_boxs_key_is_dropped_and_the_real_bearer_attached(tmp_path):
    """The whole design, in one request.

    The box presents its key as `Authorization: Bearer <placeholder>` -- that
    is the family's one auth header -- and the upstream must see the real key
    and no trace of what the box presented. A test that only checked the
    bearer would pass while the placeholder rode along beside it.
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
                f"{URL}/chat/completions",
                data=b"{}",
                headers={
                    "Authorization": "Bearer aisan-placeholder-not-a-credential",
                    "user-agent": "opencode/1.18.18",
                    "x-session-id": "ses_box_chosen",
                    "x-session-affinity": "ses_box_chosen",
                },
            ) as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == "Bearer real-key"
    # aiohttp sets its own user-agent on the outbound request, so the
    # assertion is that the BOX's did not survive, not that the header is
    # absent.
    assert got.get("user-agent", "") != "opencode/1.18.18"
    assert "x-session-id" not in got
    assert "x-session-affinity" not in got


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
            s.get(f"{URL}/models") as r,
        ):
            assert r.status == 403
            body = await r.json()
    finally:
        await up_runner.cleanup()

    assert reached == []
    # The family's own error envelope: measured, `error.message` is the field
    # opencode surfaces, and any other shape collapses to a generic
    # "Unexpected server error" that hides the reason from the agent.
    assert "not permitted" in body["error"]["message"]
    assert body["error"]["type"] == "invalid_request_error"


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
                async with s.post(f"{URL}/chat/completions", data=b"{}") as r:
                    assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert "policy check raised" in caplog.text


async def test_the_token_is_read_per_request_not_captured(tmp_path):
    """The host's opencode rewrites the credential file whenever the user
    re-runs /connect, so a value captured at startup survives a key the user
    has since rotated -- while a re-read picks the new one up for free."""
    seen: list[str] = []
    tokens = iter(["k-1", "k-2", "k-3"])

    async def upstream(request: web.Request) -> web.Response:
        seen.append(request.headers["authorization"])
        return web.json_response({"ok": True})

    async def rotating() -> str:
        return next(tokens)

    up, up_runner = await _upstream_server(upstream)
    try:
        async with _Proxy(tmp_path, token=rotating, upstream=up) as s:
            for _ in range(3):
                async with s.post(f"{URL}/chat/completions", data=b"{}") as r:
                    assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert seen == ["Bearer k-1", "Bearer k-2", "Bearer k-3"]


async def test_a_credential_that_cannot_be_read_is_an_error_not_a_traceback(
    tmp_path,
):
    """The file went away mid-session. The agent gets a reason it can act on,
    and the reason names a path -- there is no value to leak here by
    construction, but the assertion pins that."""

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def gone() -> str:
        raise FileNotFoundError("/nonexistent/auth.json")

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=gone, upstream=up) as s,
            s.post(f"{URL}/chat/completions", data=b"{}") as r,
        ):
            assert r.status == 503
            body = await r.json()
    finally:
        await up_runner.cleanup()
    assert body["error"]["type"] == "api_error"
    assert "auth.json" in body["error"]["message"]


async def test_a_dead_upstream_is_502_not_a_torn_connection(tmp_path):
    """One turn fails with a legible reason; the client is not left retrying
    into a connection that was reset."""

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()  # the address is now dead, deliberately

    async with (
        _Proxy(tmp_path, token=_token, upstream=up) as s,
        s.post(f"{URL}/chat/completions", data=b"{}") as r,
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
            async with s.post(f"{URL}/chat/completions", data=b"{}") as first:
                assert first.status == 200
            async with s.post(f"{URL}/chat/completions", data=b"{}") as second:
                assert second.status == 429
                body = await second.json()
    finally:
        await up_runner.cleanup()
    assert body["error"]["type"] == "rate_limit_error"


async def test_the_response_body_is_streamed_back_intact(tmp_path):
    """A streamed turn is many SSE chunks, and the client parses them as they
    arrive. Byte-correctness across the chunk boundary is what makes the
    proxy invisible to it."""
    payload = b"".join(
        f"data: {json.dumps({'choices': [{'delta': {'x': i}}]})}\n\n".encode()
        for i in range(64)
    )

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
                f"{URL}/chat/completions",
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
        return "real-key"

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=counting_token, upstream=up) as s,
            s.post(
                f"{URL}/chat/completions",
                data=json.dumps(
                    {"tools": [{"type": "web_search", "name": "w"}]}
                ).encode(),
            ) as r,
        ):
            assert r.status == 403
            payload = await r.json()
    finally:
        await up_runner.cleanup()

    assert reached == []
    assert tokens_read == []
    assert "web_search" in payload["error"]["message"]


async def test_a_real_body_still_reaches_the_upstream(tmp_path):
    """The end-to-end negative control: the policy is in the request path and
    a legitimate turn is unaffected by it."""
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(await request.read())
        return web.json_response({"ok": True})

    payload = json.dumps(
        {"model": "glm-5.2", "tools": [_REAL_TOOL], "stream": True}
    ).encode()
    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(f"{URL}/chat/completions", data=payload) as r,
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
                s.post(f"{URL}/chat/completions", data=b"{}") as r,
            ):
                assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert reached == []
    assert "policy check raised" in caplog.text


async def test_an_interrupted_turn_is_not_an_upstream_outage(tmp_path, caplog):
    """The interrupt window BEFORE the first response byte, faithfully.

    opencode aborts its fetch while the model is still thinking -- the
    upstream holds the request and has answered nothing -- so the proxy's
    first write to the closing socket is the response's own header line in
    `prepare`, not a body chunk. aiohttp raises ClientConnectionResetError
    there, a ClientError by inheritance as well as a ConnectionResetError,
    and unguarded that write surfaced as "upstream ... unreachable: Cannot
    write to closing transport" -- a cancelled turn logged as a dead
    upstream, plus a 502 written to a socket nobody is reading.

    The client aborts for real rather than closing: a graceful FIN can be
    absorbed by socket buffers and the whole small response may land before
    any write fails. It aborts only after the upstream has the request, so
    the abort cannot race the proxy's read of it.
    """
    import asyncio
    import logging

    arrived = asyncio.Event()

    async def upstream(request: web.Request) -> web.Response:
        arrived.set()
        await asyncio.sleep(0.3)  # the turn is "thinking"; outlive the client
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    sock = tmp_path / "oc.sock"
    try:
        with caplog.at_level(logging.DEBUG):
            runner = await serve(sock, make_app(token=_token, upstream=up))
            try:
                _r, w = await asyncio.open_unix_connection(str(sock))
                w.write(
                    b"POST /chat/completions HTTP/1.1\r\n"
                    b"Host: openai.invalid\r\n"
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
