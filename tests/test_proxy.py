# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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


ANTHROPIC_MODEL = "claude-sonnet-4-5@20250929"
ANTHROPIC_BASE = f"/v1/projects/{PROJECT}/locations/{LOCATION}"
ANTHROPIC_MODELS = f"{ANTHROPIC_BASE}/publishers/anthropic/models"


def _allow() -> Allowlist:
    return Allowlist(project=PROJECT, location=LOCATION, models=(MODEL,))


def _allow_both() -> Allowlist:
    return Allowlist(
        project=PROJECT,
        location=LOCATION,
        models=(MODEL,),
        anthropic_models=(ANTHROPIC_MODEL,),
    )


def test_the_log_gate_bounds_a_refusal_flood(caplog, monkeypatch):
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


def test_allowlist_permits_exactly_the_two_gemini_shapes():
    a = _allow()
    assert a.permits("POST", f"{BASE}/publishers/google/models/{MODEL}:generateContent")
    assert a.permits(
        "POST", f"{BASE}/publishers/google/models/{MODEL}:streamGenerateContent"
    )
    assert a.permits("POST", f"{BASE}/cachedContents")
    assert a.permits("DELETE", f"{BASE}/cachedContents/12345")


def test_allowlist_permits_the_anthropic_shape_only_where_configured():
    a = _allow_both()
    assert a.permits("POST", f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict")
    assert a.permits("POST", f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:streamRawPredict")

    assert a.permits("POST", f"{BASE}/publishers/google/models/{MODEL}:generateContent")
    assert a.permits("POST", f"{BASE}/cachedContents")

    assert not _allow().permits(
        "POST", f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict"
    )


def test_an_empty_model_list_permits_no_model_path_at_all():
    no_claude = _allow()
    assert not no_claude.permits("POST", f"{ANTHROPIC_MODELS}/:rawPredict")
    assert not no_claude.permits("POST", f"{ANTHROPIC_MODELS}/:streamRawPredict")
    assert no_claude.dialect("POST", f"{ANTHROPIC_MODELS}/:rawPredict") is None

    no_gemini = Allowlist(project=PROJECT, location=LOCATION, models=())
    assert not no_gemini.permits(
        "POST", f"{BASE}/publishers/google/models/:generateContent"
    )

    assert no_gemini.permits("POST", f"{BASE}/cachedContents")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{MODEL}:rawPredict",
            "a Gemini name on the Anthropic route",
        ),
        (
            "POST",
            f"{BASE}/publishers/google/models/{ANTHROPIC_MODEL}:generateContent",
            "a Claude name on the Gemini route",
        ),
        (
            "POST",
            (
                f"/v1beta1/projects/{PROJECT}/locations/{LOCATION}"
                f"/publishers/anthropic/models/{ANTHROPIC_MODEL}:rawPredict"
            ),
            "the Anthropic shape under the Gemini version",
        ),
        (
            "POST",
            f"{ANTHROPIC_BASE}/publishers/google/models/{MODEL}:generateContent",
            "the Gemini shape under the Anthropic version",
        ),
        (
            "POST",
            f"{ANTHROPIC_BASE}/publishers/google/models/{ANTHROPIC_MODEL}:rawPredict",
            "publisher",
        ),
        ("POST", "/v1/projects/other/locations/global/cachedContents", "project"),
        (
            "POST",
            (
                f"/v1/projects/{PROJECT}/locations/us-central1"
                f"/publishers/anthropic/models/{ANTHROPIC_MODEL}:rawPredict"
            ),
            "location",
        ),
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:predict",
            "method suffix",
        ),
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:countTokens",
            "method suffix",
        ),
        ("POST", f"{ANTHROPIC_MODELS}/count-tokens:rawPredict", "pseudo-model"),
        ("GET", f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict", "http method"),
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredictEvil",
            "suffix anchoring",
        ),
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}EXTRA:rawPredict",
            "model anchoring",
        ),
        (
            "POST",
            f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict/../../evil",
            "path traversal",
        ),
    ],
)
def test_allowlist_refuses_on_the_anthropic_route(method, path, why):
    assert not _allow_both().permits(method, path), why


def test_dialect_names_the_protocol_behind_the_path_that_matched():
    a = _allow_both()
    assert (
        a.dialect("POST", f"{BASE}/publishers/google/models/{MODEL}:generateContent")
        == "google"
    )
    assert a.dialect("POST", f"{BASE}/cachedContents") == "google"
    assert a.dialect("DELETE", f"{BASE}/cachedContents/12345") == "google"
    assert (
        a.dialect("POST", f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict")
        == "anthropic"
    )
    assert a.dialect("POST", f"{BASE}/nope") is None


def test_allowlist_escapes_regex_metacharacters_in_an_anthropic_model():

    a = Allowlist(
        project=PROJECT,
        location=LOCATION,
        models=(),
        anthropic_models=("claude-4.5@1",),
    )
    assert a.permits("POST", f"{ANTHROPIC_MODELS}/claude-4.5@1:rawPredict")
    assert not a.permits("POST", f"{ANTHROPIC_MODELS}/claude-4X5@1:rawPredict")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        ("POST", "/v1beta1/projects/other/locations/global/cachedContents", "project"),
        (
            "POST",
            f"/v1beta1/projects/{PROJECT}/locations/us-central1/cachedContents",
            "location",
        ),
        (
            "POST",
            f"{BASE}/publishers/google/models/gemini-1.0-ultra:generateContent",
            "model",
        ),
        (
            "POST",
            f"{BASE}/publishers/google/models/{MODEL}:countTokens",
            "method suffix",
        ),
        ("GET", f"{BASE}/cachedContents", "http method"),
        ("POST", f"{BASE}/cachedContents/12345/../../evil", "path traversal"),
        ("POST", f"{BASE}/cachedContents:batchDelete", "suffix smuggling"),
        ("POST", f"{BASE}/cachedContentsEXTRA", "anchoring"),
    ],
)
def test_allowlist_refuses(method, path, why):
    assert not _allow().permits(method, path), why


def test_allowlist_escapes_regex_metacharacters():

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
    assert not r.allow()
    r._last -= 1.0
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

    assert calls == 0


async def test_an_allowlist_that_raises_denies(tmp_path, caplog):
    from aiohttp import ClientSession

    calls = 0

    async def token() -> str:
        nonlocal calls
        calls += 1
        return "unused"

    class Exploding:
        def dialect(self, method: str, path: str) -> str | None:
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

    assert any(r.exc_info for r in caplog.records), "the bug must reach the log"


async def test_relay_streams_incrementally(tmp_path):
    from aiohttp import ClientSession

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


async def test_an_upstream_redirect_is_refused_not_followed(tmp_path):
    from aiohttp import ClientSession

    second_hits: list[str] = []

    async def second(request: web.Request) -> web.Response:
        second_hits.append(request.headers.get("Authorization", ""))
        return web.json_response({"ok": True})

    elsewhere, second_runner = await _upstream_server(second)

    async def upstream(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(location=f"{elsewhere}{request.path}")

    up_url, up_runner = await _upstream_server(upstream)

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
            url = (
                f"http://127.0.0.1:{port}{BASE}"
                f"/publishers/google/models/{MODEL}:generateContent"
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
        await up_runner.cleanup()
        await second_runner.cleanup()

    assert second_hits == []
    assert "does not follow redirects" in body["error"]["message"]


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_dead_upstream_is_502_not_a_torn_connection(tmp_path, streaming):
    from aiohttp import ClientSession

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up_url, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()

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


_SESSION_HEADER = "X-Test-Session-Id"

_UNSET = object()


async def _forwarded_headers(
    tmp_path, path, *, session_header=_UNSET, session_id=None, sent=None
):
    from aiohttp import ClientSession

    tmp_path.mkdir(parents=True, exist_ok=True)
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
            allowlist=_allow_both(),
            token=lambda: _ready("real"),
            location=LOCATION,
            session_header=(
                _SESSION_HEADER if session_header is _UNSET else session_header
            ),
            session_id=session_id,
        )
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as s:
                await s.post(
                    f"http://127.0.0.1:{port}{path}",
                    data=b"{}",
                    headers=sent or {},
                )
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig
        await up_runner.cleanup()
    return got


async def test_claude_requests_carry_the_proxys_session_id(tmp_path):
    got = await _forwarded_headers(
        tmp_path,
        f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict",
        session_id="box-session",
    )
    assert got[_SESSION_HEADER] == "box-session"


async def test_a_box_cannot_choose_its_own_session_id(tmp_path):
    got = await _forwarded_headers(
        tmp_path,
        f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict",
        session_id="ours",
        sent={_SESSION_HEADER: "chosen-by-the-box"},
    )
    assert got[_SESSION_HEADER] == "ours"


async def test_gemini_requests_carry_no_session_id(tmp_path):
    got = await _forwarded_headers(tmp_path, f"{BASE}/cachedContents")
    assert _SESSION_HEADER not in got


async def test_no_session_header_means_no_affinity_at_all(tmp_path):
    claude = f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict"
    named = await _forwarded_headers(tmp_path / "named", claude)
    unnamed = await _forwarded_headers(
        tmp_path / "unnamed", claude, session_header=None
    )

    assert _SESSION_HEADER in named
    assert set(named) - set(unnamed) == {_SESSION_HEADER}
    assert set(unnamed) - set(named) == set()


async def test_each_proxy_mints_its_own_session_id(tmp_path):
    claude = f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:rawPredict"
    first = await _forwarded_headers(tmp_path / "a", claude)
    second = await _forwarded_headers(tmp_path / "b", claude)
    assert first[_SESSION_HEADER] != second[_SESSION_HEADER]
    assert len(first[_SESSION_HEADER]) == 32


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

    assert any("refused" in r.getMessage() for r in caplog.records)


async def test_a_box_that_hangs_up_mid_stream_is_not_an_error(tmp_path, caplog):
    import logging

    async def upstream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "application/json"})
        await resp.prepare(request)

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

        sock = tmp_path / "v.sock"
        runner = await serve_proxy(sock, app)
        try:
            with caplog.at_level(logging.DEBUG):
                r, w = await asyncio.open_unix_connection(str(sock))
                path = f"{BASE}/publishers/google/models/{MODEL}:streamGenerateContent"
                w.write(
                    f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                    f"Content-Length: 2\r\n\r\n{{}}".encode()
                )
                await w.drain()
                await r.read(64)
                w.transport.abort()

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

    assert BodyPolicy().refuse(b"") is None
    assert BodyPolicy().refuse(b'{"contents": []}') is None

    assert BodyPolicy().refuse(b'{"tools": [{"function_declarations": []}]}') is None


def test_body_policy_refuses_a_contents_part_that_fetches_a_uri():
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
    assert field in reason


def test_body_policy_refuses_the_snake_case_spelling_of_a_server_tool():
    from aisan.proxy.vertex import BodyPolicy

    for field in ("url_context", "google_search", "code_execution"):
        body = json.dumps({"tools": [{field: {}}]}).encode()
        assert BodyPolicy().refuse(body) is not None, field


def test_body_policy_refuses_a_server_tool_beside_a_permitted_one():
    from aisan.proxy.vertex import BodyPolicy

    beside = json.dumps({"tools": [_REAL_FUNCTION_TOOL, {"urlContext": {}}]}).encode()
    within = json.dumps({"tools": [{**_REAL_FUNCTION_TOOL, "urlContext": {}}]}).encode()

    assert BodyPolicy().refuse(beside) is not None
    assert BodyPolicy().refuse(within) is not None


def test_body_policy_refuses_shapes_it_cannot_classify():
    from aisan.proxy.vertex import BodyPolicy

    bodies = [
        b"{not json",
        b'["an", "array"]',
        b'{"tools": "not a list"}',
        b'{"tools": ["not an object"]}',
        b'{"tools": [{"functionDeclarations": []}], "tools": [{"urlContext": {}}]}',
        b'{"tools": [], "pad": NaN}',
    ]
    for body in bodies:
        assert BodyPolicy().refuse(body) is not None, body


async def test_a_refused_body_never_reaches_the_upstream_or_the_credential(tmp_path):
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


_REAL_ANTHROPIC_TOOL = {
    "name": "read_file",
    "description": "read",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


_REAL_ANTHROPIC_BODY = {
    "anthropic_version": "vertex-2023-10-16",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "system": [
        {
            "type": "text",
            "text": "be brief",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ],
    "temperature": 0.0,
    "top_k": 40,
    "top_p": 0.95,
    "stop_sequences": ["STOP"],
    "stream": True,
    "tool_choice": {"type": "any"},
    "tools": [_REAL_ANTHROPIC_TOOL],
}


def test_the_anthropic_body_policy_permits_what_the_real_client_sends():
    from aisan.proxy.vertex import BodyPolicy, anthropic_body_policy

    body = json.dumps(_REAL_ANTHROPIC_BODY).encode()
    assert anthropic_body_policy().refuse(body) is None

    turn = json.dumps(
        {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "ok"}],
                        }
                    ],
                }
            ],
        }
    ).encode()
    assert anthropic_body_policy().refuse(turn) is None

    assert BodyPolicy().refuse(body) is not None


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "search"),
        ({"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]}, "fetch"),
        ({"mcp_servers": [{"url": "https://x.example"}]}, "mcp"),
        ({"container": "c"}, "container"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://x.example/exfil",
                                },
                            }
                        ],
                    }
                ]
            },
            "an image url source is the upstream fetching for the box",
        ),
        ({"betas": ["something"]}, "a key nobody measured is refused, not forwarded"),
    ],
)
def test_the_anthropic_body_policy_still_refuses_upstream_capabilities(extra, why):
    from aisan.proxy.vertex import anthropic_body_policy

    body = json.dumps({**_REAL_ANTHROPIC_BODY, **extra}).encode()
    assert anthropic_body_policy().refuse(body) is not None, why


def test_the_anthropic_body_policy_refuses_a_model_key():
    from aisan.proxy.vertex import anthropic_body_policy

    body = json.dumps({**_REAL_ANTHROPIC_BODY, "model": "claude-opus-4"}).encode()
    reason = anthropic_body_policy().refuse(body)
    assert reason is not None
    assert "model" in reason


def test_tool_choice_cannot_force_a_tool_the_policy_refuses():
    from aisan.proxy.vertex import anthropic_body_policy

    forced = json.dumps(
        {
            **_REAL_ANTHROPIC_BODY,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "tool", "name": "web_search"},
        }
    ).encode()
    reason = anthropic_body_policy().refuse(forced)
    assert reason is not None
    assert "web_search" in reason


async def test_a_claude_request_reaches_the_upstream_with_our_bearer(tmp_path):
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v

    seen: dict = {}

    async def upstream(request: web.Request) -> web.Response:
        seen["path"] = request.path
        seen["auth"] = request.headers.get("Authorization")
        seen["x-api-key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(await request.text())
        return web.json_response({"type": "message", "content": []})

    async def token() -> str:
        return "real-bearer"

    up_url, up_runner = await _upstream_server(upstream)
    orig_up = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(allowlist=_allow_both(), token=token, location=LOCATION)
        sock = tmp_path / "vertex.sock"
        runner = await serve_proxy(sock, app)
        relay = await serve_relay(sock, 0)
        port = relay.sockets[0].getsockname()[1]
        try:
            async with (
                ClientSession() as s,
                s.post(
                    f"http://127.0.0.1:{port}{ANTHROPIC_MODELS}"
                    f"/{ANTHROPIC_MODEL}:streamRawPredict",
                    data=json.dumps(_REAL_ANTHROPIC_BODY).encode(),
                    headers={"x-api-key": "a-key-the-box-made-up"},
                ) as r,
            ):
                assert r.status == 200, await r.text()
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig_up
        await up_runner.cleanup()

    assert seen["path"] == f"{ANTHROPIC_MODELS}/{ANTHROPIC_MODEL}:streamRawPredict"
    assert seen["auth"] == "Bearer real-bearer"
    assert seen["x-api-key"] is None, "nothing the box sent is forwarded"
    assert seen["body"]["tools"] == [_REAL_ANTHROPIC_TOOL]


async def test_a_gemini_path_is_never_read_with_the_anthropic_policy(tmp_path):
    from aiohttp import ClientSession

    import aisan.proxy.vertex as v
    from aisan.proxy.vertex import anthropic_body_policy

    smuggled = json.dumps({"tools": [{"googleSearch": {}}]}).encode()

    assert anthropic_body_policy().refuse(smuggled) is None

    reached = False

    async def upstream(request: web.Request) -> web.Response:
        nonlocal reached
        reached = True
        return web.json_response({})

    async def token() -> str:
        return "real-bearer"

    up_url, up_runner = await _upstream_server(upstream)
    orig_up = v._upstream
    v._upstream = lambda _loc: up_url
    try:
        app = make_app(allowlist=_allow_both(), token=token, location=LOCATION)
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
                    data=smuggled,
                ) as r,
            ):
                assert r.status == 403
                assert "googleSearch" in json.loads(await r.text())["error"]["message"]
        finally:
            relay.close()
            await relay.wait_closed()
            await runner.cleanup()
    finally:
        v._upstream = orig_up
        await up_runner.cleanup()

    assert not reached
