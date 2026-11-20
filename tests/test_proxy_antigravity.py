# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Policy tests pinned to what Antigravity CLI 1.5.11 sent on 2026-11-19."""

from __future__ import annotations

import json
import logging
from types import MappingProxyType
from typing import Self

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.proxy.antigravity import (
    ALLOWED_ROUTES,
    USERINFO,
    LocalAnswer,
    Profile,
    RouteAllowlist,
    _envelope_refusal,
    make_app,
    serve,
)
from aisan.proxy.gemini import BodyPolicy

BOX_TOKEN = "box-token-for-tests"
HOST_TOKEN = "host-token-for-tests"
URL = "http://antigravity.invalid"
USER_AGENT = (
    "antigravity/cli/1.5.11 (aidev_client; os_type=linux; arch=amd64;"
    " cl=1024248524; auth_method=consumer)"
)

MEASURED_ROUTES = [
    ("POST", "/v1internal:loadCodeAssist", ""),
    ("POST", "/v1internal:fetchUserInfo", ""),
    ("POST", "/v1internal:fetchAdminControls", ""),
    ("POST", "/v1internal:retrieveUserQuotaSummary", ""),
    ("POST", "/v1internal:fetchAvailableModels", ""),
    ("POST", "/v1internal:listExperiments", ""),
    ("POST", "/v1internal:writeTrajectoryAcls", ""),
    ("GET", "/v1internal/cascadeNuxes", "page_size=1000"),
    ("POST", "/v1internal:streamGenerateContent", "alt=sse"),
    ("POST", "/v1internal:generateContent", ""),
]


def _declaration(name: str) -> dict:
    return {
        "functionDeclarations": [
            {"name": name, "description": "d", "parameters": {"type": "OBJECT"}}
        ]
    }


def _agent_turn() -> dict:
    """An agent turn after one tool call, with long values shortened."""
    return {
        "project": "aicode-consumers",
        "requestId": "agent/c0189a30/1795102802093/4303c4bc/3",
        "model": "claude-opus-5-5-low",
        "userAgent": "antigravity",
        "requestType": "agent",
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "run echo"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "toolu_vrtx_01",
                                "name": "run_command",
                                "args": {"command": "echo marker"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "toolu_vrtx_01",
                                "name": "run_command",
                                "response": {"output": "marker"},
                            }
                        }
                    ],
                },
            ],
            "systemInstruction": {"role": "user", "parts": [{"text": "s"}]},
            "tools": [
                _declaration(name)
                for name in (
                    "read_file",
                    "run_command",
                    "manage_task",
                    "send_message",
                    "schedule",
                    "invoke_subagent",
                    "define_subagent",
                    "manage_subagents",
                    "write_file",
                    "edit_file",
                    "search_web",
                    "run_workflow",
                    "swarm",
                    "ask_question",
                )
            ],
            "generationConfig": {
                "maxOutputTokens": 128000,
                "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "LOW"},
            },
            "labels": {"cascade_id": "c0189a30", "used_claude": "true"},
            "sessionId": "-3750763034362895579",
        },
    }


def _checkpoint() -> dict:
    """The title-generation call: no tools and no labels."""
    return {
        "project": "aicode-consumers",
        "requestId": "checkpoint/f9e2acac",
        "model": "gemini-3.5-flash-lite",
        "userAgent": "antigravity",
        "requestType": "checkpoint",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "t"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "s"}]},
            "generationConfig": {
                "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "LOW"}
            },
            "sessionId": "-3750763034362895579",
        },
    }


def _web_search() -> dict:
    """The call the client makes to execute its ``search_web`` tool."""
    return {
        "project": "aicode-consumers",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "bubblewrap sandbox"}]}],
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": "You are a search engine bot."}],
            },
            "tools": [
                {
                    "googleSearch": {
                        "enhancedContent": {"imageSearch": {"maxResultCount": 5}}
                    }
                }
            ],
            "generationConfig": {"candidateCount": 1},
        },
        "model": "gemini-3.5-flash-lite",
        "userAgent": "antigravity",
        "requestType": "web_search",
    }


def _refusal(payload: object, policy: BodyPolicy | None = None) -> str | None:
    return _envelope_refusal(json.dumps(payload).encode(), policy or BodyPolicy())


@pytest.mark.parametrize(("method", "path", "query"), MEASURED_ROUTES)
def test_the_measured_routes_are_permitted(method, path, query):
    assert RouteAllowlist().permits(method, path, query)


def test_the_allowlist_is_exactly_the_measured_set():
    assert {(m, p, q) for (m, p), qs in ALLOWED_ROUTES.items() for q in qs} == set(
        MEASURED_ROUTES
    )


@pytest.mark.parametrize(
    ("method", "path", "query", "why"),
    [
        ("POST", "/v1internal:setUserSettings", "", "writes account settings"),
        ("POST", "/v1internal:recordTrajectoryAnalytics", "", "uploads analytics"),
        ("GET", "/v1internal:loadCodeAssist", "", "wrong method"),
        ("POST", "/v1internal/cascadeNuxes", "page_size=1000", "wrong method"),
        ("POST", "/v1internal:loadCodeAssistant", "", "suffix"),
        ("POST", "/v2/v1internal:loadCodeAssist", "", "prefix"),
        ("POST", "/v1internal:streamGenerateContent", "", "missing query"),
        ("POST", "/v1internal:streamGenerateContent", "alt=sse&key=k", "extra query"),
        ("POST", "/v1internal:generateContent", "alt=sse", "unmeasured query"),
        ("POST", "/v1internal:loadCodeAssist", "access_token=t", "query credential"),
        ("POST", "/v1internal%3AloadCodeAssist", "", "escaped separator"),
        ("POST", "/v1internal:onboardUser", "", "not measured"),
        ("GET", "/", "", "root"),
    ],
)
def test_other_routes_are_refused(method, path, query, why):
    assert not RouteAllowlist().permits(method, path, query), why


@pytest.mark.parametrize("payload", [_agent_turn(), _checkpoint()])
def test_the_measured_model_calls_pass_the_isolated_policy(payload):
    assert _refusal(payload) is None


def test_search_is_refused_in_isolation_and_permitted_on_a_shared_network():
    assert "googleSearch" in _refusal(_web_search())
    assert _refusal(_web_search(), BodyPolicy.for_shared_network()) is None


@pytest.mark.parametrize("payload", [_agent_turn(), _checkpoint()])
def test_the_shared_policy_still_permits_ordinary_turns(payload):
    assert _refusal(payload, BodyPolicy.for_shared_network()) is None


def test_the_shared_policy_holds_search_to_its_measured_shape():
    shared = BodyPolicy.for_shared_network()

    beside_functions = _web_search()
    beside_functions["request"]["tools"].append(_declaration("run_command"))
    assert "only tool" in _refusal(beside_functions, shared)

    two_fields = _web_search()
    two_fields["request"]["tools"][0]["functionDeclarations"] = []
    assert "only tool" in _refusal(two_fields, shared)

    with_history = _web_search()
    with_history["request"]["contents"].append(
        {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]}
    )
    assert "only text parts" in _refusal(with_history, shared)

    no_contents = _web_search()
    del no_contents["request"]["contents"]
    assert "contents" in _refusal(no_contents, shared)


@pytest.mark.parametrize(
    "field", ["urlContext", "codeExecution", "googleSearchRetrieval", "google_search"]
)
def test_the_shared_policy_admits_no_other_upstream_tool(field):
    payload = _web_search()
    payload["request"]["tools"] = [{field: {}}]
    assert field in _refusal(payload, BodyPolicy.for_shared_network())


@pytest.mark.parametrize("policy", [BodyPolicy(), BodyPolicy.for_shared_network()])
def test_a_part_that_makes_the_upstream_fetch_is_refused(policy):
    payload = _agent_turn()
    payload["request"]["contents"][0]["parts"].append(
        {"fileData": {"fileUri": "https://example.invalid/x"}}
    )
    assert "fileData" in _refusal(payload, policy)


def test_an_envelope_field_outside_the_measured_set_is_refused():
    payload = _agent_turn()
    payload["tools"] = [{"googleSearch": {}}]
    assert "'tools'" in _refusal(payload)


def test_a_model_allowlist_refuses_an_unlisted_or_unnamed_model():
    listed = frozenset({"claude-opus-5-5-low"})
    turn = _agent_turn()
    body = json.dumps(turn).encode()
    assert _envelope_refusal(body, BodyPolicy(), listed) is None
    other = _envelope_refusal(body, BodyPolicy(), frozenset({"gemini-3.5-flash-lite"}))
    assert "claude-opus-5-5-low" in other

    del turn["model"]
    unnamed = _envelope_refusal(json.dumps(turn).encode(), BodyPolicy(), listed)
    assert "not permitted" in unnamed


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"[]",
        b"not json",
        b'{"project": "p"}',
        b'{"request": []}',
        b'{"request": {"tools": {}}}',
        b'{"request": {}, "request": {"tools": [{"googleSearch": {}}]}}',
        b'{"request": {"tools": [], "tools": [{"googleSearch": {}}]}}',
    ],
)
def test_a_model_call_the_policy_cannot_classify_is_refused(body):
    assert _envelope_refusal(body, BodyPolicy()) is not None


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


async def _host_token() -> str:
    return HOST_TOKEN


class _Proxy:
    """The proxy on a Unix socket, with a session presenting the box token."""

    def __init__(self, tmp_path, upstream: str = "http://127.0.0.1:1", **kwargs):
        self._sock = tmp_path / "antigravity.sock"
        kwargs.setdefault("token", _host_token)
        kwargs.setdefault("profile", Profile(upstream=upstream))
        self._kwargs = kwargs

    async def __aenter__(self) -> ClientSession:
        self._runner = await serve(
            self._sock, make_app(client_token=BOX_TOKEN, **self._kwargs)
        )
        self._session = ClientSession(
            connector=UnixConnector(path=str(self._sock)),
            headers={"Authorization": f"Bearer {BOX_TOKEN}"},
        )
        return self._session

    async def __aexit__(self, *exc) -> None:
        await self._session.close()
        await self._runner.cleanup()


class _Upstream:
    """A fake Cloud Code host that records what reaches it."""

    def __init__(self, respond=None) -> None:
        self.requests: list[web.Request] = []
        self.bodies: list[bytes] = []
        self._respond = respond

    async def __aenter__(self) -> Self:
        self.url, self._runner = await _upstream_server(self._handle)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(request)
        self.bodies.append(await request.read())
        if self._respond is not None:
            return await self._respond(request)
        return web.json_response({})


STREAM = "/v1internal:streamGenerateContent?alt=sse"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": ""},
        {"Authorization": "Bearer wrong"},
        {"Authorization": BOX_TOKEN},
        {"Authorization": f"Basic {BOX_TOKEN}"},
        [
            ("Authorization", f"Bearer {BOX_TOKEN}"),
            ("Authorization", f"Bearer {BOX_TOKEN}"),
        ],
    ],
)
@pytest.mark.parametrize("path", ["/v1internal:loadCodeAssist", "/userinfo"])
async def test_a_request_without_the_one_box_token_is_refused(tmp_path, headers, path):
    read = []

    async def token() -> str:
        read.append(1)
        return HOST_TOKEN

    async with _Upstream() as up, _Proxy(tmp_path, up.url, token=token) as session:
        method = "GET" if path == "/userinfo" else "POST"
        async with session.request(
            method, f"{URL}{path}", headers=headers, skip_auto_headers=["Authorization"]
        ) as response:
            assert response.status == 401
            assert (await response.json())["error"]["status"] == "UNAUTHENTICATED"
    assert not up.requests
    assert not read


async def test_a_turn_reaches_the_upstream_with_the_host_token(tmp_path):
    events = b'data: {"response": {}}\n\ndata: {"response": {"done": true}}\n\n'

    async def respond(request: web.Request) -> web.Response:
        return web.Response(body=events, content_type="text/event-stream")

    sent = json.dumps(_agent_turn()).encode()

    async def chunks():
        # The client sends turns chunked, with no Content-Length.
        yield sent[:100]
        yield sent[100:]

    async with (
        _Upstream(respond) as up,
        _Proxy(tmp_path, up.url) as session,
        session.post(
            f"{URL}{STREAM}",
            data=chunks(),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
                "X-Goog-User-Project": "box-project",
                "X-Goog-Api-Key": "box-key",
                "Cookie": "box=1",
            },
        ) as response,
    ):
        assert response.status == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        assert await response.read() == events

    [request] = up.requests
    assert request.path_qs == STREAM
    assert up.bodies == [sent]
    assert request.headers["Authorization"] == f"Bearer {HOST_TOKEN}"
    assert request.headers["User-Agent"] == USER_AGENT
    assert request.headers["Content-Type"] == "application/json"
    for name in ("X-Goog-User-Project", "X-Goog-Api-Key", "Cookie"):
        assert name not in request.headers


async def test_a_control_call_is_forwarded_unchanged(tmp_path):
    async def respond(request: web.Request) -> web.Response:
        return web.json_response({"currentTier": {"id": "free-tier"}})

    body = b'{"metadata": {"ideType": "ANTIGRAVITY"}}'
    async with _Upstream(respond) as up, _Proxy(tmp_path, up.url) as session:
        async with session.post(
            f"{URL}/v1internal:loadCodeAssist", data=body
        ) as response:
            assert await response.json() == {"currentTier": {"id": "free-tier"}}
        async with session.get(
            f"{URL}/v1internal/cascadeNuxes?page_size=1000"
        ) as response:
            assert response.status == 200

    assert [r.path_qs for r in up.requests] == [
        "/v1internal:loadCodeAssist",
        "/v1internal/cascadeNuxes?page_size=1000",
    ]
    assert up.bodies == [body, b""]
    assert "Content-Type" not in up.requests[1].headers


async def test_the_token_is_read_for_every_request(tmp_path):
    tokens = iter(["first", "second"])

    async def token() -> str:
        return next(tokens)

    async with _Upstream() as up, _Proxy(tmp_path, up.url, token=token) as session:
        for _ in range(2):
            async with session.post(f"{URL}/v1internal:listExperiments", data=b"{}"):
                pass
    assert [r.headers["Authorization"] for r in up.requests] == [
        "Bearer first",
        "Bearer second",
    ]


async def test_userinfo_is_answered_without_the_upstream_or_the_credential(tmp_path):
    async def token() -> str:
        raise AssertionError("the profile stub needs no credential")

    async with _Upstream() as up, _Proxy(tmp_path, up.url, token=token) as session:
        async with session.get(f"{URL}/userinfo") as response:
            assert response.status == 200
            profile = await response.json()
        async with session.post(f"{URL}/userinfo") as response:
            assert response.status == 403
        async with session.get(f"{URL}/userinfo?alt=json") as response:
            assert response.status == 403
    assert profile == json.loads(USERINFO.body)
    assert profile["picture"] == ""
    assert not up.requests


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (STREAM, _web_search()),
        ("/v1internal:generateContent", _web_search()),
        (STREAM, {**_agent_turn(), "extra": 1}),
    ],
)
async def test_a_refused_body_reaches_neither_upstream_nor_credential(
    tmp_path, path, payload
):
    async def token() -> str:
        raise AssertionError("a refused body must not read the credential")

    async with (
        _Upstream() as up,
        _Proxy(tmp_path, up.url, token=token) as session,
        session.post(f"{URL}{path}", json=payload) as response,
    ):
        assert response.status == 403
        assert (await response.json())["error"]["status"] == "PERMISSION_DENIED"
    assert not up.requests


async def test_the_shared_policy_forwards_the_search_call(tmp_path):
    async with (
        _Upstream() as up,
        _Proxy(tmp_path, up.url, body=BodyPolicy.for_shared_network()) as session,
        session.post(
            f"{URL}/v1internal:generateContent", json=_web_search()
        ) as response,
    ):
        assert response.status == 200
    assert json.loads(up.bodies[0]) == _web_search()


async def test_a_credential_failure_is_503_and_sends_nothing_upstream(tmp_path, caplog):
    async def token() -> str:
        raise RuntimeError("keyring is locked")

    async with (
        _Upstream() as up,
        _Proxy(tmp_path, up.url, token=token) as session,
        session.post(f"{URL}/v1internal:loadCodeAssist", data=b"{}") as response,
    ):
        assert response.status == 503
        assert "keyring is locked" in (await response.json())["error"]["message"]
    assert not up.requests


async def test_account_writes_are_refused_quietly_and_strays_are_logged(
    tmp_path, caplog
):
    async with _Upstream() as up, _Proxy(tmp_path, up.url) as session:
        with caplog.at_level(logging.WARNING, logger="aisan.proxy.antigravity"):
            for path in (
                "/v1internal:setUserSettings",
                "/v1internal:recordTrajectoryAnalytics",
                "/v1internal:onboardUser",
            ):
                async with session.post(f"{URL}{path}", data=b"{}") as response:
                    assert response.status == 403
    assert not up.requests
    assert [r.getMessage() for r in caplog.records] == [
        "antigravity proxy: refused POST /v1internal:onboardUser"
    ]


@pytest.mark.parametrize("status", [429, 503])
async def test_an_upstream_error_is_relayed_unchanged(tmp_path, status):
    error = json.dumps({"error": {"code": status, "status": "X"}}).encode()

    async def respond(request: web.Request) -> web.Response:
        return web.Response(
            status=status,
            body=error,
            content_type="application/json",
            headers={"Retry-After": "7"},
        )

    async with (
        _Upstream(respond) as up,
        _Proxy(tmp_path, up.url) as session,
        session.post(f"{URL}{STREAM}", json=_agent_turn()) as response,
    ):
        assert response.status == status
        assert response.headers["Retry-After"] == "7"
        assert await response.read() == error


async def test_an_upstream_redirect_is_refused_not_followed(tmp_path):
    followed = []

    async def elsewhere(request: web.Request) -> web.Response:
        followed.append(request.headers.get("Authorization"))
        return web.json_response({})

    other, other_runner = await _upstream_server(elsewhere)

    async def respond(request: web.Request) -> web.Response:
        return web.Response(status=307, headers={"Location": f"{other}/steal"})

    try:
        async with (
            _Upstream(respond) as up,
            _Proxy(tmp_path, up.url) as session,
            session.post(f"{URL}/v1internal:loadCodeAssist", data=b"{}") as response,
        ):
            assert response.status == 502
    finally:
        await other_runner.cleanup()
    assert not followed


async def test_a_profile_adds_routes_and_answers_but_cannot_lift_a_refusal(tmp_path):
    def profile(upstream: str) -> Profile:
        return Profile(
            upstream=upstream,
            routes=MappingProxyType(
                {
                    ("POST", "/v1internal:rewriteUri"): frozenset({""}),
                    ("POST", "/v1internal:setUserSettings"): frozenset({""}),
                    ("POST", "/v1internal:loadCodeAssist"): frozenset({"x=1"}),
                }
            ),
            local=MappingProxyType(
                {
                    ("GET", "/avatar"): LocalAnswer(b"png", content_type="image/png"),
                    ("GET", "/userinfo"): LocalAnswer(b'{"email": "other"}'),
                }
            ),
        )

    async with _Upstream() as up, _Proxy(tmp_path, profile=profile(up.url)) as session:
        async with session.post(f"{URL}/v1internal:rewriteUri", data=b"{}") as r:
            assert r.status == 200
        async with session.post(f"{URL}/v1internal:setUserSettings", data=b"{}") as r:
            assert r.status == 403
        async with session.post(
            f"{URL}/v1internal:loadCodeAssist?x=1", data=b"{}"
        ) as r:
            assert r.status == 403
        async with session.get(f"{URL}/avatar") as r:
            assert (r.content_type, await r.read()) == ("image/png", b"png")
        async with session.get(f"{URL}/userinfo") as r:
            assert await r.json() == json.loads(USERINFO.body)
    assert [r.path for r in up.requests] == ["/v1internal:rewriteUri"]
