# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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
    assert PathAllowlist().permits("POST", "/chat/completions")


@pytest.mark.parametrize(
    ("method", "path", "why"),
    [
        ("POST", "/chat/completions_evil", "prefix"),
        ("POST", "/chat/completions/", "trailing slash"),
        ("GET", "/models", "catalog listing"),
        ("POST", "/embeddings", "not a model call"),
        ("POST", "/v1/chat/completions", "the prefix is host-side knowledge"),
        ("GET", "/chat/completions", "wrong method"),
        ("DELETE", "/chat/completions", "wrong method"),
    ],
)
def test_path_allowlist_refuses(method, path, why):
    assert not PathAllowlist().permits(method, path), why


_REAL_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Executes a bash command",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_body_policy_permits_a_real_opencode_body():
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
        ("web_search", "upstream issues a query"),
        ("code_interpreter", "runs in a container upstream"),
        ("some_tool_type_invented_next_year", "unknown to us and to the upstream"),
    ],
)
def test_body_policy_refuses_tools_that_execute_upstream(kind, why):
    body = json.dumps({"tools": [{"type": kind, "name": "t"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None, why
    assert kind in reason


def test_body_policy_refuses_web_search_options():
    body = json.dumps(
        {"web_search_options": {"user_location": {"country": "US"}}, "messages": []}
    ).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None
    assert "web_search_options" in reason


def test_body_policy_refuses_an_absent_type():
    tool = dict(_REAL_TOOL)
    del tool["type"]
    reason = BodyPolicy().refuse(json.dumps({"tools": [tool]}).encode())
    assert reason is not None
    assert "no `type`" in reason


def test_body_policy_refuses_an_explicit_null_type():
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


def test_body_policy_refuses_fetchable_content_parts():
    p = BodyPolicy()
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/exfil"},
                        }
                    ],
                }
            ]
        }
    ).encode()
    reason = p.refuse(body)
    assert reason is not None
    assert "image_url" in reason
    ok = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        }
    ).encode()
    assert p.refuse(ok) is None


def test_body_policy_refuses_an_unmeasured_top_level_key():
    p = BodyPolicy()
    for key in ("functions", "a_key_invented_next_year"):
        body = json.dumps({"model": "m", "messages": [], key: []}).encode()
        reason = p.refuse(body)
        assert reason is not None, key
        assert key in reason


def test_body_policy_refuses_duplicate_keys_at_any_depth():

    bodies = [
        b'{"tools": [], "tools": [{"type": "web_search"}]}',
        b'{"tools": [{"type": "web_search", "type": "function"}]}',
    ]
    assert all(BodyPolicy().refuse(body) is not None for body in bodies)


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


class _Proxy:
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

    assert got.get("user-agent", "") != "opencode/1.18.18"
    assert "x-session-id" not in got
    assert "x-session-affinity" not in got


async def test_a_refused_path_never_reaches_the_upstream(tmp_path):
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

    assert "not permitted" in body["error"]["message"]
    assert body["error"]["type"] == "invalid_request_error"


async def test_a_path_allowlist_that_raises_denies(tmp_path, caplog):

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


async def test_an_upstream_redirect_is_refused_not_followed(tmp_path):
    second_hits: list[dict[str, str]] = []

    async def second(request: web.Request) -> web.Response:
        second_hits.append({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    elsewhere, second_runner = await _upstream_server(second)

    async def upstream(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(location=f"{elsewhere}/chat/completions")

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(f"{URL}/chat/completions", data=b"{}") as r,
        ):
            assert r.status == 502
            body = await r.json()
    finally:
        await up_runner.cleanup()
        await second_runner.cleanup()

    assert second_hits == []
    assert "does not follow redirects" in body["error"]["message"]


async def test_a_dead_upstream_is_502_not_a_torn_connection(tmp_path):

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()

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

    assert reached == [payload]


async def test_a_body_policy_that_raises_denies(tmp_path, caplog):
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
    import asyncio
    import logging

    arrived = asyncio.Event()

    async def upstream(request: web.Request) -> web.Response:
        arrived.set()
        await asyncio.sleep(0.3)
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
                w.transport.abort()
                await asyncio.sleep(0.6)
            finally:
                await runner.cleanup()
    finally:
        await up_runner.cleanup()

    assert not [rec for rec in caplog.records if "unreachable" in rec.getMessage()], (
        "a cancelled turn logged as an upstream outage"
    )

    assert any("downstream closed" in rec.getMessage() for rec in caplog.records)
