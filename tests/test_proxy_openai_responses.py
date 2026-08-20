# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Measured policy surface for Codex's OpenAI Responses transport."""

from __future__ import annotations

import json

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.proxy.openai_responses import (
    BodyPolicy,
    PathAllowlist,
    make_app,
    serve,
)


def _function(name: str = "exec_command") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Runs locally in the Codex client",
        "parameters": {"type": "object", "properties": {}},
    }


def test_path_allowlist_is_the_one_measured_codex_route():
    paths = PathAllowlist()
    assert paths.permits("POST", "/responses")
    for method, path in [
        ("GET", "/responses"),
        ("POST", "/responses/compact"),
        ("POST", "/v1/responses"),
        ("POST", "/chat/completions"),
    ]:
        assert not paths.permits(method, path)


def test_body_policy_permits_measured_functions_and_namespaces():
    body = json.dumps(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "measure"}],
                }
            ],
            "tools": [
                _function(),
                {
                    "type": "namespace",
                    "name": "multi_agent_v1",
                    "tools": [_function("spawn_agent"), _function("wait_agent")],
                },
            ],
            "store": False,
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


def test_body_policy_permits_measured_compaction_body_without_tools():
    body = json.dumps(
        {
            "input": [
                {"type": "function_call", "name": "exec_command"},
                {"type": "function_call_output", "output": "done"},
                {"type": "reasoning", "encrypted_content": "opaque"},
            ],
            "tools": [],
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
    )
    assert BodyPolicy().refuse(body.encode()) is None


def test_body_policy_permits_measured_gpt_5_6_additional_tools_envelope():
    body = json.dumps(
        {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "tools": [
                                {"type": "custom", "name": "exec"},
                                _function("wait"),
                            ],
                        }
                    ],
                }
            ],
            "store": False,
            "stream": True,
            "text": {"verbosity": "low"},
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


@pytest.mark.parametrize("kind", ["web_search", "code_interpreter", "unknown"])
def test_body_policy_refuses_server_side_tools_at_any_depth(kind):
    direct = json.dumps({"tools": [{"type": kind}]}).encode()
    nested = json.dumps(
        {"tools": [{"type": "namespace", "name": "n", "tools": [{"type": kind}]}]}
    ).encode()
    assert kind in BodyPolicy().refuse(direct)
    assert kind in BodyPolicy().refuse(nested)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "namespace"},
        {"type": "namespace", "tools": "not-an-array"},
        {"type": "namespace", "tools": ["not-an-object"]},
    ],
)
def test_body_policy_refuses_unclassifiable_namespaces(tool):
    assert BodyPolicy().refuse(json.dumps({"tools": [tool]}).encode()) is not None


@pytest.mark.parametrize(
    "change",
    [
        {"background": True},
        {"store": True},
        {"stream": False},
        {"include": ["web_search_call.action.sources"]},
        {"text": {"format": {"type": "json_schema"}}},
        {"text": {"verbosity": "unbounded"}},
        {"input": [{"type": "computer_call"}]},
        {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "user",
                    "tools": [_function()],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "web_search"}],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "message",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.test/x"}
                    ],
                }
            ]
        },
    ],
)
def test_body_policy_refuses_unmeasured_response_capabilities(change):
    body = {"input": [], "tools": [], "store": False, "stream": True, **change}
    assert BodyPolicy().refuse(json.dumps(body).encode()) is not None


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}/v1", runner


async def _credential() -> tuple[str, str]:
    return "real-key", "real-account"


async def test_valid_response_request_reaches_the_prefixed_upstream(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(
            (
                request.method,
                request.path,
                request.headers["authorization"],
                request.headers["chatgpt-account-id"],
                await request.read(),
            )
        )
        return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

    upstream_url, upstream_runner = await _upstream_server(upstream)
    socket = tmp_path / "responses.sock"
    proxy_runner = await serve(
        socket, make_app(credential=_credential, upstream=upstream_url)
    )
    session = ClientSession(connector=UnixConnector(path=str(socket)))
    body = json.dumps(
        {"input": [], "tools": [_function()], "store": False, "stream": True}
    ).encode()
    try:
        async with session.post(
            "http://codex.invalid/responses", data=body
        ) as response:
            assert response.status == 200
            assert await response.read() == b"data: [DONE]\n\n"
    finally:
        await session.close()
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()

    assert reached == [
        ("POST", "/v1/responses", "Bearer real-key", "real-account", body)
    ]


async def test_responses_lite_headers_are_reconstructed_not_forwarded(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append({key.lower(): value for key, value in request.headers.items()})
        return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

    upstream_url, upstream_runner = await _upstream_server(upstream)
    socket = tmp_path / "responses-lite.sock"
    proxy_runner = await serve(
        socket, make_app(credential=_credential, upstream=upstream_url)
    )
    session = ClientSession(connector=UnixConnector(path=str(socket)))
    body = json.dumps(
        {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [_function()],
                }
            ],
            "store": False,
            "stream": True,
        }
    ).encode()
    try:
        async with session.post(
            "http://codex.invalid/responses",
            data=body,
            headers={
                "X-Codex-Beta-Features": "box-chosen-feature",
                "X-OpenAI-Internal-Codex-Responses-Lite": "false",
                "X-Arbitrary": "box-data",
                "Authorization": "Bearer box-token",
                "ChatGPT-Account-Id": "box-account",
            },
        ) as response:
            assert response.status == 200
    finally:
        await session.close()
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()

    assert reached[0]["x-codex-beta-features"] == "remote_compaction_v2"
    assert reached[0]["x-openai-internal-codex-responses-lite"] == "true"
    assert reached[0]["authorization"] == "Bearer real-key"
    assert reached[0]["chatgpt-account-id"] == "real-account"
    assert reached[0]["originator"] == "codex_cli_rs"
    assert "x-arbitrary" not in reached[0]
