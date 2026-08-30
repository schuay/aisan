# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The sandbox egress proxy: allowlist, streaming, and the no-credential claim.

The allowlist is the security boundary (the socket only decides who may knock),
so most of this file is adversarial rather than happy-path: the cases that must
be REFUSED are the ones worth writing down.

Everything here talks to the proxy over plain HTTP. The claim that a REAL model
client -- configured the way a consumer configures one, holding an anonymous
credential -- reaches upstream through this proxy needs that client, so it is
asserted in the backend tests rather than here. Which half broke is then
unambiguous: this file failing is the proxy, that one failing is the wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

import pytest
from aiohttp import web

from aisan.proxy import Allowlist, RateLimit, make_app, serve_relay
from aisan.proxy import serve as serve_proxy

PROJECT = "v8-proj"
LOCATION = "global"
MODEL = "gemini-3.1-pro-preview"
BASE = f"/v1beta1/projects/{PROJECT}/locations/{LOCATION}"


def _allow() -> Allowlist:
    return Allowlist(project=PROJECT, location=LOCATION, models=(MODEL,))


def test_the_log_gate_bounds_a_refusal_flood(caplog, monkeypatch):
    """The refusal keeps happening; only the logging is bounded. When the gate
    reopens, the first line through it accounts for what the flood suppressed,
    so the operator sees the volume without reading it."""
    from aisan.proxy.http import LogGate

    gate = LogGate(per_minute=5)
    logger = logging.getLogger("aisan-test-log-gate")
    with caplog.at_level(logging.WARNING, logger="aisan-test-log-gate"):
        for i in range(100):
            gate.warning(logger, "refused %d", i)
        real = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: real + 30.0)
        gate.warning(logger, "refused later")

    messages = [r.getMessage() for r in caplog.records]
    assert messages[:5] == [f"refused {i}" for i in range(5)]
    assert any("95 refusal warnings suppressed" in m for m in messages)
    assert messages[-1] == "refused later"


def test_allowlist_permits_exactly_the_two_shapes():
    a = _allow()
    assert a.permits("POST", f"{BASE}/publishers/google/models/{MODEL}:generateContent")
    assert a.permits(
        "POST", f"{BASE}/publishers/google/models/{MODEL}:streamGenerateContent"
    )
    assert a.permits("POST", f"{BASE}/cachedContents")
    assert a.permits("DELETE", f"{BASE}/cachedContents/12345")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        # Another project: the box must not reach a resource we did not name.
        ("POST", "/v1beta1/projects/other/locations/global/cachedContents", "project"),
        # Another location routes to a different regional host.
        (
            "POST",
            f"/v1beta1/projects/{PROJECT}/locations/us-central1/cachedContents",
            "location",
        ),
        # A model we did not configure.
        (
            "POST",
            f"{BASE}/publishers/google/models/gemini-1.0-ultra:generateContent",
            "model",
        ),
        # An unlisted method on a listed resource.
        (
            "POST",
            f"{BASE}/publishers/google/models/{MODEL}:countTokens",
            "method suffix",
        ),
        # GET is not on the list at all: listing caches is not ours to grant.
        ("GET", f"{BASE}/cachedContents", "http method"),
        # Trailing junk must not ride along on an otherwise-valid prefix.
        ("POST", f"{BASE}/cachedContents/12345/../../evil", "path traversal"),
        ("POST", f"{BASE}/cachedContents:batchDelete", "suffix smuggling"),
        # A prefix match would let this through; the pattern is anchored.
        ("POST", f"{BASE}/cachedContentsEXTRA", "anchoring"),
    ],
)
def test_allowlist_refuses(method, path, why):
    assert not _allow().permits(method, path), why


def test_allowlist_escapes_regex_metacharacters():
    # A project id carrying a metacharacter must not widen the pattern -- the
    # classic way an allowlist quietly stops being one.
    a = Allowlist(project="a.b", location=LOCATION, models=(MODEL,))
    assert a.permits(
        "POST", f"/v1beta1/projects/a.b/locations/{LOCATION}/cachedContents"
    )
    assert not a.permits(
        "POST", f"/v1beta1/projects/axb/locations/{LOCATION}/cachedContents"
    )


def test_rate_limit_bounds_a_runaway_then_refills():
    r = RateLimit(per_minute=60)
    assert all(r.allow() for _ in range(60))
    assert not r.allow()  # bucket drained
    r._last -= 1.0  # one second of refill == one token at 60/min
    assert r.allow()


def test_shared_http_api_keeps_existing_exports():
    from aisan.proxy import RateLimit as package_rate_limit
    from aisan.proxy import serve as package_serve
    from aisan.proxy.http import RateLimit as shared_rate_limit
    from aisan.proxy.http import serve as shared_serve
    from aisan.proxy.vertex import RateLimit as legacy_rate_limit
    from aisan.proxy.vertex import serve as legacy_serve

    assert package_rate_limit is shared_rate_limit is legacy_rate_limit
    assert package_serve is shared_serve is legacy_serve


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return f"http://127.0.0.1:{port}", runner


async def test_refusal_is_shaped_so_the_client_can_parse_it(tmp_path):
    # Streaming errors come back from real Vertex as a JSON ARRAY; the SDK calls
    # .get on the parsed body, so an object yields AttributeError and hides the
    # reason. A refused agent must see the refusal.
    from aiohttp import ClientSession

    calls = 0

    async def token() -> str:
        nonlocal calls
        calls += 1
        return "unused"

    app = make_app(allowlist=_allow(), token=token, location=LOCATION)
    sock = tmp_path / "vertex.sock"
    runner = await serve_proxy(sock, app)
    relay = await serve_relay(sock, 0)
    port = relay.sockets[0].getsockname()[1]
    try:
        async with ClientSession() as s:
            bad = f"http://127.0.0.1:{port}/v1beta1/projects/other/locations/global/cachedContents"
            async with s.post(bad) as r:
                assert r.status == 403
                assert isinstance(await r.json(), dict)
            stream_bad = (
                f"http://127.0.0.1:{port}/v1beta1/projects/other/locations/global"
                f"/publishers/google/models/{MODEL}:streamGenerateContent"
            )
            async with s.post(stream_bad) as r:
                assert r.status == 403
                body = await r.json()
                assert isinstance(body, list), "streaming refusals must be an array"
                assert body[0]["error"]["code"] == 403
    finally:
        relay.close()
        await relay.wait_closed()
        await runner.cleanup()
    # A refused request must never mint a token: refusal happens before auth.
    assert calls == 0


async def test_an_allowlist_that_raises_denies(tmp_path, caplog):
    """The fail-closed contract, asserted by breaking the policy.

    The allowlist is caller-supplied, so a bug in one is a real case: a config
    value that makes the pattern raise, a consumer's own matcher with a KeyError
    in it. Before the contract the exception unwound aiohttp's handler and the
    client saw a 500 with an empty body -- a transport-shaped failure the SDK
    retries, so a policy that never approved the request let it through on the
    second try.

    Refused, once, with the reason -- and no token minted, because the whole point
    is that a broken policy must not reach the credential.
    """
    from aiohttp import ClientSession

    calls = 0

    async def token() -> str:
        nonlocal calls
        calls += 1
        return "unused"

    class Exploding:
        def permits(self, method: str, path: str) -> bool:
            raise RuntimeError("the policy is broken")

    app = make_app(allowlist=Exploding(), token=token, location=LOCATION)
    sock = tmp_path / "vertex.sock"
    runner = await serve_proxy(sock, app)
    relay = await serve_relay(sock, 0)
    port = relay.sockets[0].getsockname()[1]
    try:
        with caplog.at_level(logging.ERROR, logger="aisan.proxy.policy"):
            async with ClientSession() as s:
                good = (
                    f"http://127.0.0.1:{port}{BASE}"
                    f"/publishers/google/models/{MODEL}:generateContent"
                )
                async with s.post(good) as r:
                    assert r.status == 403, "a raising policy must deny, not 500"
                    body = await r.json()
                    assert "not permitted" in body["error"]["message"]
    finally:
        relay.close()
        await relay.wait_closed()
        await runner.cleanup()
    assert calls == 0, "a request the policy never approved must not mint a token"
    # And it is loud: a mint failure is summarised because it repeats, but a
    # policy that raises is a bug in the boundary and gets a traceback every time.
    assert any(r.exc_info for r in caplog.records), "the bug must reach the log"


async def test_relay_streams_incrementally(tmp_path):
    """The relay must forward per chunk, not accumulate.

    A buffering relay is byte-correct, so this cannot be a correctness
    assertion -- it has to be a timing one. An earlier probe in this project
    passed a buffering forwarder precisely because it counted chunks instead.
    """
    from aiohttp import ClientSession

    # A UNIX server, not the TCP helper: this exercises the relay in isolation,
    # which is where the buffering risk lives.
    async def unix_stream(reader, writer):
        await reader.read(65536)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
        writer.write(b"Transfer-Encoding: chunked\r\n\r\n")
        await writer.drain()
        for i in range(3):
            payload = f'{{"chunk":{i}}}'.encode()
            writer.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
            await writer.drain()
            await asyncio.sleep(0.25)
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()

    sock = tmp_path / "stream.sock"
    server = await asyncio.start_unix_server(unix_stream, str(sock))
    relay = await serve_relay(sock, 0)
    port = relay.sockets[0].getsockname()[1]
    try:
        # Bounded: a relay that accumulates the REQUEST never forwards it (the
        # client half only sees EOF when the connection closes), so the failure
        # mode for buffering is a deadlock rather than a wrong value. Without
        # this the suite hangs instead of reporting, which is a worse signal
        # than a red test.
        async with asyncio.timeout(15):
            async with ClientSession() as s:
                arrivals = []
                t0 = time.monotonic()
                async with s.get(f"http://127.0.0.1:{port}/x") as r:
                    async for _ in r.content.iter_any():
                        arrivals.append(time.monotonic() - t0)
        assert len(arrivals) >= 3
        assert arrivals[-1] - arrivals[0] > 0.3, f"relay buffered: arrivals {arrivals}"
    finally:
        relay.close()
        await relay.wait_closed()
        server.close()
        await server.wait_closed()


async def test_relay_names_the_direction_that_broke(tmp_path, caplog):
    """A relay disconnect is ordinary, but which leg broke is diagnostic.

    Both directions are spliced concurrently, so an unlabelled line cannot say
    whether the box hung up or the host proxy did -- and "host proxy -> box"
    repeating across jobs means something very different from the former. The
    label is the whole point of logging here, so that is what this asserts.

    The box aborts while the host is still streaming, which is what forces the
    write leg to fail. A peer that simply goes away after writing delivers EOF
    instead, ending the loop normally -- no exception, nothing to log. That
    asymmetry is why this test streams with gaps rather than closing cleanly.
    """
    import logging

    async def unix_stream(reader, writer):
        await reader.read(65536)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
        writer.write(b"Transfer-Encoding: chunked\r\n\r\n")
        await writer.drain()
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            for i in range(20):
                payload = f'{{"chunk":{i}}}'.encode()
                writer.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.05)

    sock = tmp_path / "r.sock"
    server = await asyncio.start_unix_server(unix_stream, str(sock))
    relay = await serve_relay(sock, 0)
    port = relay.sockets[0].getsockname()[1]
    try:
        with caplog.at_level(logging.DEBUG, logger="aisan.proxy.relay"):
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.write(b"GET /x HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await w.drain()
            await r.read(64)
            # Abort, not close: the relay must still be mid-write for the write
            # leg to raise at all.
            w.transport.abort()
            await asyncio.sleep(0.5)
    finally:
        relay.close()
        await relay.wait_closed()
        server.close()
        await server.wait_closed()

    msgs = [r.getMessage() for r in caplog.records if r.name.endswith("proxy.relay")]
    assert any("host proxy -> box" in m for m in msgs), msgs


async def test_token_is_read_per_request_not_captured(tmp_path):
    """Jobs outlive tokens, so a value read once would strand a long job."""
    seen: list[str] = []
    nth = 0

    async def rotating() -> str:
        nonlocal nth
        nth += 1
        return f"token-{nth}"

    async def upstream(request: web.Request) -> web.Response:
        seen.append(request.headers.get("Authorization", ""))
        return web.json_response({"ok": True})

    up_url, up_runner = await _upstream_server(upstream)
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v

    # Point the proxy's upstream at the fake server.
    orig = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(allowlist=_allow(), token=rotating, location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as s:
                url = f"http://127.0.0.1:{port}{BASE}/cachedContents"
                for _ in range(3):
                    async with s.post(url, data=b"{}") as r:
                        assert r.status == 200
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig
        await up_runner.cleanup()

    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_dead_upstream_is_502_not_a_torn_connection(tmp_path, streaming):
    """A route to Vertex that is down is one turn's failure with a legible
    reason, not a torn connection the SDK retries into. And the refusal keeps
    the streaming shape: an array for streamGenerateContent, an object
    otherwise, so the client parses it rather than raising on the wrong type."""
    from aiohttp import ClientSession

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up_url, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()  # the address is now dead, deliberately

    import aisan.proxy.vertex as v

    orig = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(allowlist=_allow(), token=lambda: _ready("t"), location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            verb = "streamGenerateContent" if streaming else "generateContent"
            url = (
                f"http://127.0.0.1:{port}{BASE}/publishers/google/models/{MODEL}:{verb}"
            )
            async with ClientSession() as s, s.post(url, data=b"{}") as r:
                assert r.status == 502
                body = await r.json()
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig

    error = (body[0] if streaming else body)["error"]
    assert "cannot reach its upstream" in error["message"]


async def test_box_headers_are_never_forwarded(tmp_path):
    """Anything the box sends is data. It must not be able to set headers."""
    from aiohttp import ClientSession

    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update(request.headers)
        return web.json_response({"ok": True})

    up_url, up_runner = await _upstream_server(upstream)
    import aisan.proxy.vertex as v

    orig = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(
            allowlist=_allow(), token=lambda: _ready("real"), location=LOCATION
        )
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as s:
                await s.post(
                    f"http://127.0.0.1:{port}{BASE}/cachedContents",
                    data=b"{}",
                    headers={
                        "Authorization": "Bearer smuggled",
                        "X-Goog-User-Project": "someone-elses-project",
                    },
                )
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig
        await up_runner.cleanup()

    assert got["Authorization"] == "Bearer real"
    assert "X-Goog-User-Project" not in got


async def _ready(value: str) -> str:
    return value


async def test_oversized_body_is_refused_before_upstream(tmp_path):
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v

    reached = False

    async def upstream(request: web.Request) -> web.Response:
        nonlocal reached
        reached = True
        return web.json_response({})

    up_url, up_runner = await _upstream_server(upstream)
    orig_up, orig_max = v._upstream, v.MAX_BODY_BYTES
    v._upstream = lambda _loc: up_url
    v.MAX_BODY_BYTES = 1024
    try:
        app = make_app(allowlist=_allow(), token=lambda: _ready("t"), location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with (
                ClientSession() as s,
                s.post(
                    f"http://127.0.0.1:{port}{BASE}/cachedContents",
                    data=b"x" * 4096,
                ) as r,
            ):
                assert r.status == 413
                assert json.loads(await r.text())["error"]["code"] == 413
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream, v.MAX_BODY_BYTES = orig_up, orig_max
        await up_runner.cleanup()

    assert not reached, "an oversized body must not reach upstream"


async def test_large_multi_chunk_request_is_forwarded_intact(tmp_path):
    """Large request payloads spanning multiple buffer chunks must not be truncated."""
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v

    received_len = 0

    async def upstream(request: web.Request) -> web.Response:
        nonlocal received_len
        body = await request.read()
        received_len = len(body)
        return web.json_response({"ok": True})

    up_url, up_runner = await _upstream_server(upstream)
    orig = v._upstream
    v._upstream = lambda _loc: up_url
    # 500 KiB payload (well above a single 64 KiB chunk / buffer).
    payload = json.dumps({"data": "A" * 500_000}).encode()
    try:
        app = make_app(allowlist=_allow(), token=lambda: _ready("t"), location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as s:
                url = f"http://127.0.0.1:{port}{BASE}/cachedContents"
                async with s.post(url, data=payload) as r:
                    assert r.status == 200
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig
        await up_runner.cleanup()

    assert received_len == len(payload)


async def test_the_proxy_does_not_log_every_request(tmp_path, caplog):
    """aiohttp's access log is off, deliberately.

    Its default logs one INFO line per request on `aiohttp.access`. An agent turn
    is a stream of POSTs plus the DELETEs that tear each cached context down, so
    at sustained volume that is thousands of lines saying only that the proxy worked
    -- burying the lines that carry a decision, above all the refusal warning
    that is the allowlist doing its job. Both halves are asserted, since turning
    the logger off wholesale would silence the half worth keeping.
    """
    import logging

    async def token() -> str:
        return "unused"

    app = make_app(allowlist=_allow(), token=token, location=LOCATION)
    sock = tmp_path / "vertex.sock"
    runner = await serve_proxy(sock, app)
    try:
        with caplog.at_level(logging.INFO):
            from aiohttp import ClientSession, UnixConnector

            denied = "/v1beta1/projects/other/locations/global/cachedContents"
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as s,
                s.post(f"http://localhost{denied}") as r,
            ):
                await r.text()
    finally:
        await runner.cleanup()

    assert not [r for r in caplog.records if r.name.startswith("aiohttp.access")]
    # ...and the decision the proxy DID make is still on the record.
    assert any("refused" in r.getMessage() for r in caplog.records)


async def test_a_box_that_hangs_up_mid_stream_is_not_an_error(tmp_path, caplog):
    """A disconnect mid-response is the ordinary end of a turn, not a fault.

    The box goes away whenever the agent exits or its turn is cancelled, leaving
    the proxy writing a response nobody will read. Unhandled, that surfaces as a
    full ClientConnectionResetError traceback at ERROR from aiohttp's generic
    handler -- exactly the noise the disabled access log exists to prevent, and
    it buries the refusal warning that is the allowlist doing its job.

    The client here aborts rather than closes: a graceful FIN can be absorbed by
    socket buffers, so a whole small response may land before the write fails
    and the bug would not reproduce.
    """
    import logging

    async def upstream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "application/json"})
        await resp.prepare(request)
        # Chunks with gaps, so the abort lands mid-body rather than racing the
        # whole response into the socket buffer. Suppressed because the proxy
        # drops its upstream connection once the box is gone, so this double
        # takes a reset of its own -- noise from the stand-in, not the subject.
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            for i in range(5):
                await resp.write(json.dumps({"chunk": i}).encode())
                await asyncio.sleep(0.05)
            await resp.write_eof()
        return resp

    up_url, up_runner = await _upstream_server(upstream)
    import aisan.proxy.vertex as v

    orig = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(
            allowlist=_allow(), token=lambda: _ready("unused"), location=LOCATION
        )
        # The socket path lands in sun_path (~108 bytes), and this test's name
        # is already most of a pytest tmp_path, so keep the basename short.
        sock = tmp_path / "v.sock"
        runner = await serve_proxy(sock, app)
        try:
            with caplog.at_level(logging.DEBUG):
                # Raw connection, not a ClientSession: we need to vanish
                # mid-stream, which a client that drains its response cannot do.
                r, w = await asyncio.open_unix_connection(str(sock))
                path = f"{BASE}/publishers/google/models/{MODEL}:streamGenerateContent"
                w.write(
                    f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                    f"Content-Length: 2\r\n\r\n{{}}".encode()
                )
                await w.drain()
                await r.read(64)  # headers plus the first chunk or so
                w.transport.abort()
                # Let the proxy notice and finish the handler.
                await asyncio.sleep(0.5)
        finally:
            await runner.cleanup()
    finally:
        v._upstream = orig
        await up_runner.cleanup()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, (
        f"disconnect logged as an error: {[e.getMessage() for e in errors]}"
    )
    assert any("downstream closed" in r.getMessage() for r in caplog.records)


async def test_run_with_relay_serves_during_command(tmp_path):
    import socket
    import sys

    from aisan.proxy import run_with_relay

    sock = tmp_path / "test.sock"

    async def handle(reader, writer):
        d = await reader.read(100)
        writer.write(b"echo:" + d)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    srv = await asyncio.start_unix_server(handle, str(sock))
    try:
        test_script = (
            "import socket, sys\n"
            "s = socket.create_connection(('127.0.0.1', int(sys.argv[1])))\n"
            "s.sendall(b'hello')\n"
            "assert s.recv(100) == b'echo:hello'\n"
            "s.close()\n"
        )
        # Choose a free port
        tmp_s = socket.socket()
        tmp_s.bind(("127.0.0.1", 0))
        port = tmp_s.getsockname()[1]
        tmp_s.close()

        rc = await run_with_relay(
            sock, port, [sys.executable, "-c", test_script, str(port)]
        )
        assert rc == 0
    finally:
        srv.close()
        await srv.wait_closed()


# --- the body policy -------------------------------------------------------
#
# The allowlists above gate the ENVELOPE of a model call. The body is not inert:
# a tool the body declares makes GOOGLE fetch, search or execute for the box, so
# `unshare_net` stops the box dialling out while leaving it able to ask Vertex to
# dial out on its behalf.
#
# The permitted shape is measured against the real client rather than read off
# the API reference: langchain_google_vertexai builds every tool through
# `_format_to_gapic_tool`, and both airc paths (chat and the growing-prefix
# cache) emit exactly `functionDeclarations`. The refused set is deliberately
# NOT enumerated in the policy -- it is everything else in the union.


_REAL_FUNCTION_TOOL = {
    "functionDeclarations": [
        {
            "name": "read_file",
            "description": "read",
            "parameters": {
                "type": "OBJECT",
                "properties": {"path": {"type": "STRING"}},
                "required": ["path"],
            },
        }
    ]
}


def test_body_policy_permits_the_real_clients_tools():
    """The negative control: the exact tool shape the SDK emits.

    Taken from `MessageToJson` output for a `Tool` built by
    `_format_to_gapic_tool`, which is what both the inference and cachedContents
    paths post. A policy that refused everything would pass every adversarial
    case below and break the product.
    """
    from aisan.proxy.vertex import BodyPolicy

    inference = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [_REAL_FUNCTION_TOOL],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {"temperature": 1},
        }
    ).encode()
    cache = json.dumps(
        {
            "model": f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}",
            "systemInstruction": {"parts": [{"text": "sys"}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [_REAL_FUNCTION_TOOL],
            "ttl": "600s",
        }
    ).encode()

    assert BodyPolicy().refuse(inference) is None
    assert BodyPolicy().refuse(cache) is None
    # A cache DELETE carries no body, and a body with no tools declares nothing.
    assert BodyPolicy().refuse(b"") is None
    assert BodyPolicy().refuse(b'{"contents": []}') is None
    # The snake_case spelling of the SAME client field. Proto3 JSON requires the
    # upstream to accept it, so refusing it would refuse a legitimate request.
    assert BodyPolicy().refuse(b'{"tools": [{"function_declarations": []}]}') is None


def test_body_policy_refuses_a_contents_part_that_fetches_a_uri():
    """NEW-1: the tools gate covers the tool list, but a `contents` part carrying
    `fileData.fileUri` is the same fetch-a-URL-the-box-chose vector, one level
    down in the messages -- and it slips a body with NO tools past the old gate
    (which returned early when tools was absent). Both proto3 spellings."""
    import json

    import aisan.proxy.vertex as v

    for field in ("fileData", "file_data"):
        body = json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{field: {"fileUri": "https://x.example/exfil"}}],
                    }
                ]
            }
        ).encode()
        reason = v.BodyPolicy().refuse(body)
        assert reason is not None, field
        assert field in reason
    # inline text and base64 are inert and permitted.
    ok = json.dumps(
        {
            "contents": [
                {"role": "user", "parts": [{"text": "build it"}]},
                {"role": "user", "parts": [{"inlineData": {"data": "AAAA"}}]},
            ]
        }
    ).encode()
    assert v.BodyPolicy().refuse(ok) is None


@pytest.mark.parametrize(
    ("field", "why"),
    [
        ("googleSearch", "Google runs the search and the query is the payload"),
        ("googleSearchRetrieval", "the same capability under its older name"),
        ("urlContext", "Google fetches a URL the box chose"),
        ("codeExecution", "the upstream executes, not the box"),
        ("retrieval", "retrieval.externalApi names an endpoint for Google to call"),
        ("enterpriseWebSearch", "search on the upstream's side of the namespace"),
        ("googleMaps", "an upstream lookup on box-chosen input"),
        ("computerUse", "the upstream drives a computer for the box"),
        ("someToolInventedNextYear", "unknown to us; an allowlist refuses it"),
    ],
)
def test_body_policy_refuses_tools_that_execute_upstream(field, why):
    from aisan.proxy.vertex import BodyPolicy

    body = json.dumps({"tools": [{field: {}}]}).encode()
    reason = BodyPolicy().refuse(body)

    assert reason is not None, why
    assert field in reason  # the refusal names what it refused


def test_body_policy_refuses_the_snake_case_spelling_of_a_server_tool():
    """The bypass an allowlist keyed on one spelling would have.

    Proto3 JSON parsers are required to accept the original proto field name as
    well as the lowerCamelCase form, so the upstream honours `url_context`
    exactly as it honours `urlContext` -- and a policy that only knew the camel
    one would forward it.
    """
    from aisan.proxy.vertex import BodyPolicy

    for field in ("url_context", "google_search", "code_execution"):
        body = json.dumps({"tools": [{field: {}}]}).encode()
        assert BodyPolicy().refuse(body) is not None, field


def test_body_policy_refuses_a_server_tool_beside_a_permitted_one():
    """Smuggling past the entry that looks right. Every field of every entry is
    checked, not the first one that matches."""
    from aisan.proxy.vertex import BodyPolicy

    beside = json.dumps({"tools": [_REAL_FUNCTION_TOOL, {"urlContext": {}}]}).encode()
    within = json.dumps({"tools": [{**_REAL_FUNCTION_TOOL, "urlContext": {}}]}).encode()

    assert BodyPolicy().refuse(beside) is not None
    assert BodyPolicy().refuse(within) is not None


def test_body_policy_refuses_shapes_it_cannot_classify():
    """The only client is a REST SDK emitting MessageToJson output, so a body
    this cannot read is not a disagreement about JSON -- it is not a request the
    client sent. Refused, including the duplicate-key seam."""
    from aisan.proxy.vertex import BodyPolicy

    bodies = [
        b"{not json",
        b'["an", "array"]',
        b'{"tools": "not a list"}',
        b'{"tools": ["not an object"]}',
        b'{"tools": [{"functionDeclarations": []}], "tools": [{"urlContext": {}}]}',
        # NaN and the infinities are lenient extensions the shared parser now
        # rejects, so the three policies read a body the same way.
        b'{"tools": [], "pad": NaN}',
    ]
    for body in bodies:
        assert BodyPolicy().refuse(body) is not None, body


async def test_a_refused_body_never_reaches_the_upstream_or_the_credential(tmp_path):
    """Both halves, and the second is the easy one to lose: the check runs
    before `token()`, so a refused request does not spend a mint."""
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v

    reached = False
    minted = 0

    async def upstream(request: web.Request) -> web.Response:
        nonlocal reached
        reached = True
        return web.json_response({})

    async def token() -> str:
        nonlocal minted
        minted += 1
        return "real-bearer"

    up_url, up_runner = await _upstream_server(upstream)
    orig_up = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(allowlist=_allow(), token=token, location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with (
                ClientSession() as s,
                s.post(
                    f"http://127.0.0.1:{port}{BASE}"
                    f"/publishers/google/models/{MODEL}:generateContent",
                    data=json.dumps({"tools": [{"urlContext": {}}]}).encode(),
                ) as r,
            ):
                assert r.status == 403
                assert "urlContext" in json.loads(await r.text())["error"]["message"]
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig_up
        await up_runner.cleanup()

    assert not reached, "a refused body must not reach upstream"
    assert minted == 0, "a refused body must not spend a credential mint"
