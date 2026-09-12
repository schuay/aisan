# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import json
import logging

import pytest
from aiohttp import ClientSession, UnixConnector, web
from yarl import URL as YURL

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
        ("POST", "/v1/messages_evil", "prefix"),
        ("POST", "/v1/messages/", "trailing slash"),
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

    assert h.permits("anthropic-version")
    assert h.permits("Anthropic-Version")
    assert h.permits("anthropic-beta")
    assert h.permits("accept")

    assert not h.permits("x-api-key")
    assert not h.permits("authorization")

    assert h.permits("x-claude-code-session-id")

    assert not h.permits("x-stainless-os")
    assert not h.permits("x-stainless-runtime-version")
    assert not h.permits("user-agent")
    assert not h.permits("x-app")

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


async def test_a_percent_encoded_path_is_refused(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(YURL(f"{URL}/v1%2Fmessages", encoded=True)) as r,
        ):
            assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert reached == []


async def test_repeated_anthropic_beta_headers_all_survive(tmp_path):
    got: list[str] = []

    async def upstream(request: web.Request) -> web.Response:
        got.extend(request.headers.getall("anthropic-beta", []))
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(
                f"{URL}/v1/messages",
                data=b"{}",
                headers=[
                    ("anthropic-version", "2023-06-01"),
                    ("anthropic-beta", "claude-code-20250219"),
                    ("anthropic-beta", "interleaved-thinking-2025-05-14"),
                ],
            ) as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()
    assert got == ["claude-code-20250219", "interleaved-thinking-2025-05-14"]


async def test_diagnostic_response_headers_are_relayed(tmp_path):

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response(
            {"type": "error"},
            status=429,
            headers={
                "retry-after": "12",
                "anthropic-ratelimit-requests-remaining": "0",
                "x-request-id": "req_abc",
                "x-should-not-relay": "secret-infra-detail",
            },
        )

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, token=_token, upstream=up) as s,
            s.post(f"{URL}/v1/messages", data=b"{}") as r,
        ):
            assert r.status == 429
            assert r.headers["retry-after"] == "12"
            assert r.headers["anthropic-ratelimit-requests-remaining"] == "0"
            assert r.headers["x-request-id"] == "req_abc"
            assert "x-should-not-relay" not in r.headers
    finally:
        await up_runner.cleanup()


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

    assert got["x-claude-code-session-id"] == "a-session-the-box-opened"

    assert got.get("user-agent", "") != "claude-cli/2.1.219"


async def test_a_refused_path_never_reaches_the_upstream(tmp_path):
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

    assert body["type"] == "error"
    assert body["error"]["type"] == "permission_error"
    assert "not permitted" in body["error"]["message"]


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
                async with s.post(f"{URL}/v1/messages", data=b"{}") as r:
                    assert r.status == 403
    finally:
        await up_runner.cleanup()
    assert "policy check raised" in caplog.text


async def test_a_header_allowlist_that_raises_drops_only_that_header(tmp_path):
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

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    await up_runner.cleanup()

    async with (
        _Proxy(tmp_path, token=_token, upstream=up) as s,
        s.post(f"{URL}/v1/messages", data=b"{}") as r,
    ):
        assert r.status == 502
        body = await r.json()
    assert "cannot reach its upstream" in body["error"]["message"]


async def test_an_upstream_redirect_is_refused_not_followed(tmp_path):
    second_hits: list[dict[str, str]] = []

    async def second(request: web.Request) -> web.Response:
        second_hits.append({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    elsewhere, second_runner = await _upstream_server(second)

    async def upstream(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(location=f"{elsewhere}/v1/messages")

    up, up_runner = await _upstream_server(upstream)
    try:
        async with (
            _Proxy(tmp_path, authorization=_api_key_headers, upstream=up) as s,
            s.post(f"{URL}/v1/messages", data=b"{}") as r,
        ):
            assert r.status == 502
            body = await r.json()
    finally:
        await up_runner.cleanup()
        await second_runner.cleanup()

    assert second_hits == []
    assert "does not follow redirects" in body["error"]["message"]


async def _api_key_headers() -> dict[str, str]:
    return {"x-api-key": "real-key"}


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


_REAL_TOOL = {
    "name": "Bash",
    "description": "Executes a bash command",
    "input_schema": {"type": "object", "properties": {}},
}


def test_body_policy_permits_a_real_claude_code_body():
    body = json.dumps(
        {
            "model": "claude-opus-4",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [{"type": "text", "text": "You are Claude Code"}],
            "tools": [_REAL_TOOL, {**_REAL_TOOL, "name": "Read"}],
            "metadata": {"user_id": "x"},
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "temperature": 1.0,
            "stream": True,
            "context_management": {"edits": []},
            "output_config": {},
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


@pytest.mark.parametrize(
    ("kind", "why"),
    [
        ("web_fetch_20250910", "upstream fetches a URL"),
        ("web_search_20250305", "upstream issues a query"),
        ("web_fetch_20260318", "a version no denylist would have listed"),
        ("web_search_20260318", "a version no denylist would have listed"),
        ("code_execution_20260521", "runs in a container upstream"),
        ("mcp_toolset", "opens a session with a named server"),
        ("bash_20250124", "client-side, but unmeasured on this client"),
        ("memory_20250818", "client-side, but unmeasured on this client"),
        ("some_tool_type_invented_next_year", "unknown to us and to the upstream"),
    ],
)
def test_body_policy_refuses_tools_that_execute_upstream(kind, why):
    body = json.dumps({"tools": [{"type": kind, "name": "t"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None, why
    assert kind in reason


def test_body_policy_permits_the_two_spellings_of_the_client_variant():
    assert BodyPolicy().refuse(json.dumps({"tools": [_REAL_TOOL]}).encode()) is None
    explicit = {**_REAL_TOOL, "type": "custom"}
    assert BodyPolicy().refuse(json.dumps({"tools": [explicit]}).encode()) is None


def test_body_policy_refuses_an_explicit_null_type():
    body = json.dumps({"tools": [{**_REAL_TOOL, "type": None}]}).encode()
    assert BodyPolicy().refuse(body) is not None


@pytest.mark.parametrize("key", ["mcp_servers", "container"])
def test_body_policy_refuses_upstream_keys(key):
    body = json.dumps({"model": "m", key: [{"name": "s"}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None
    assert key in reason


def test_body_policy_refuses_shapes_it_cannot_classify():
    p = BodyPolicy()
    assert p.refuse(b"") is None
    assert p.refuse(b"{}") is None
    assert p.refuse(b"not json at all") is not None
    assert p.refuse(b'["an", "array"]') is not None
    assert p.refuse(json.dumps({"tools": "not a list"}).encode()) is not None
    assert p.refuse(json.dumps({"tools": ["not an object"]}).encode()) is not None

    smuggle = b'{"tools": [{"type": "web_search_20250305", "name": "w"}]}{}'
    reason = p.refuse(smuggle)
    assert reason is not None
    assert "classify" in reason


def test_body_policy_refuses_duplicate_keys_at_any_depth():
    bodies = [
        b'{"tools": [{"type": "web_search_20250305"}], "tools": [{"type": "custom"}]}',
        b'{"tools": [{"type": "web_search_20250305", "type": "custom"}]}',
        b'{"mcp_servers": [{"url": "https://x.test"}], "mcp_servers": []}',
    ]
    for body in bodies:
        reason = BodyPolicy().refuse(body)
        assert reason is not None, body
        assert "unambiguous" in reason


def test_body_policy_refuses_a_lenient_json_constant():
    p = BodyPolicy()
    for body in (b'{"pad": NaN}', b'{"x": Infinity}', b'{"y": -Infinity}'):
        reason = p.refuse(body)
        assert reason is not None, body
        assert "unambiguous" in reason
    behind_nan = b'{"container": [{"name": "s"}], "pad": NaN}'
    assert p.refuse(behind_nan) is not None


def test_body_policy_permits_a_measured_tool_round_trip_history():
    body = json.dumps(
        {
            "model": "claude-opus-4",
            "max_tokens": 32000,
            "system": [{"type": "text", "text": "You are Claude Code"}],
            "messages": [
                {"role": "user", "content": "run echo measured"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "m", "signature": "s"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "echo measured"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "measured",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "measured"}],
                        },
                    ],
                },
            ],
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


@pytest.mark.parametrize(
    ("block", "why"),
    [
        (
            {
                "type": "image",
                "source": {"type": "url", "url": "https://x.test/exfil"},
            },
            "the upstream fetches the URL; the URL is the payload",
        ),
        (
            {
                "type": "document",
                "source": {"type": "url", "url": "https://x.test/e"},
            },
            "the same fetch, spelled document",
        ),
        (
            {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
            "Files API state, which nothing reaches through the allowed paths",
        ),
        (
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "x"},
            },
            "no fetch, but unmeasured on this client; refused until measured",
        ),
        ({"type": "image"}, "a block with no source at all"),
        (
            {"type": "image", "source": "https://x.test/e"},
            "a source that is not an object",
        ),
        (
            {"type": "a_block_type_invented_next_year"},
            "fails closed, like the tool types",
        ),
    ],
)
def test_body_policy_refuses_content_that_asks_the_upstream_to_fetch(block, why):
    body = json.dumps({"messages": [{"role": "user", "content": [block]}]}).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None, why
    assert block["type"] in reason


def test_body_policy_permits_the_measured_inline_image_and_document():
    png = {
        "type": "image",
        "source": {"type": "base64", "data": "iVBOR", "media_type": "image/png"},
    }
    pdf = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBER",
        },
    }
    body = json.dumps(
        {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32000,
            "messages": [
                {"role": "user", "content": [png, {"type": "text", "text": "what?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "/w/red.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [png, pdf],
                        }
                    ],
                },
            ],
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None
    assert BodyPolicy.for_shared_network().refuse(body) is None


def test_body_policy_refuses_a_source_that_carries_two_answers():
    hybrid = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBOR",
            "url": "https://x.test/exfil",
        },
    }
    body = json.dumps({"messages": [{"role": "user", "content": [hybrid]}]}).encode()
    for policy in (BodyPolicy(), BodyPolicy.for_shared_network()):
        reason = policy.refuse(body)
        assert reason is not None
        assert "url" in reason


def test_every_permitted_content_type_is_inert_or_source_gated():
    from aisan.proxy.anthropic import (
        ALLOWED_CONTENT_TYPES,
        INERT_CONTENT_TYPES,
        SOURCED_CONTENT_TYPES,
    )

    assert ALLOWED_CONTENT_TYPES == INERT_CONTENT_TYPES | SOURCED_CONTENT_TYPES
    assert not INERT_CONTENT_TYPES & SOURCED_CONTENT_TYPES
    url = {"type": "url", "url": "https://x.test/e"}
    for kind in SOURCED_CONTENT_TYPES:
        body = json.dumps(
            {"messages": [{"role": "user", "content": [{"type": kind, "source": url}]}]}
        ).encode()
        assert BodyPolicy().refuse(body) is not None, kind


def test_a_refusal_never_lets_the_box_write_the_host_log():
    injected = "x\nanthropic proxy: forwarded body: fine"
    unknown_key = json.dumps({"messages": [], injected: 1}).encode()
    bad_source = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AA",
                                injected: 1,
                            },
                        }
                    ],
                }
            ]
        }
    ).encode()
    for body in (unknown_key, bad_source):
        reason = BodyPolicy().refuse(body)
        assert reason is not None
        assert "\n" not in reason
        assert "\\n" in reason


def test_the_content_gate_reaches_tool_results_and_system():
    p = BodyPolicy()
    url_image = {"type": "image", "source": {"type": "url", "url": "https://x.test/e"}}
    nested = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [url_image],
                        }
                    ],
                }
            ]
        }
    ).encode()
    assert p.refuse(nested) is not None
    in_system = json.dumps({"system": [url_image], "messages": []}).encode()
    assert p.refuse(in_system) is not None


def test_body_policy_refuses_an_unmeasured_top_level_key():
    body = json.dumps(
        {"model": "m", "messages": [], "a_key_invented_next_year": 1}
    ).encode()
    reason = BodyPolicy().refuse(body)
    assert reason is not None
    assert "a_key_invented_next_year" in reason


async def test_a_refused_body_never_reaches_the_upstream_or_the_credential(tmp_path):
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
                s.post(f"{URL}/v1/messages", data=b"{}") as r,
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


async def test_refusal_warnings_are_rate_limited(tmp_path, caplog):

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    try:
        with caplog.at_level(logging.WARNING, logger="aisan.proxy.anthropic"):
            async with _Proxy(tmp_path, token=_token, upstream=up) as s:
                for _ in range(40):
                    async with s.get(f"{URL}/v1/organizations/me") as r:
                        assert r.status == 403
    finally:
        await up_runner.cleanup()

    warnings = [r for r in caplog.records if "refused" in r.getMessage()]
    assert warnings, "the refusal must still be logged at all"
    assert len(warnings) <= 13, f"{len(warnings)} warnings for 40 refusals"


_WEB_SEARCH_BODY = {
    "model": "m",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "foo"}]}],
    "metadata": {},
    "stream": True,
    "system": [{"type": "text", "text": "s"}],
    "temperature": 1,
    "thinking": {"type": "disabled"},
    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    "tool_choice": {"type": "tool", "name": "web_search"},
}


def test_the_isolated_policy_still_refuses_the_web_search_request():
    reason = BodyPolicy().refuse(json.dumps(_WEB_SEARCH_BODY).encode())
    assert reason is not None
    assert "tool_choice" in reason
    tools_only = {k: v for k, v in _WEB_SEARCH_BODY.items() if k != "tool_choice"}
    reason = BodyPolicy().refuse(json.dumps(tools_only).encode())
    assert reason is not None
    assert "web_search_20250305" in reason


def test_the_shared_network_policy_permits_the_web_search_request():
    assert (
        BodyPolicy.for_shared_network().refuse(json.dumps(_WEB_SEARCH_BODY).encode())
        is None
    )


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ({"mcp_servers": [{"url": "https://x.test"}]}, "mcp_servers"),
        ({"container": "c"}, "container"),
        ({"tools": [{"type": "code_execution_20250522"}]}, "code_execution"),
        ({"tools": [{"type": "computer_20250124"}]}, "computer_20250124"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "url", "url": "u"}}
                        ],
                    }
                ]
            },
            "image",
        ),
        ({"unmeasured_key": 1}, "unmeasured_key"),
    ],
)
def test_the_shared_network_policy_relaxes_two_tags_and_no_others(patch, needle):
    body = json.dumps({**_WEB_SEARCH_BODY, **patch}).encode()
    reason = BodyPolicy.for_shared_network().refuse(body)
    assert reason is not None
    assert needle in reason
