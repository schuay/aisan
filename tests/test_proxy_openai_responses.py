# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Measured policy surface for Codex's OpenAI Responses transport."""

from __future__ import annotations

import json

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.proxy.openai_responses import (
    ALLOWED_INPUT_TYPES,
    CONTAINERS,
    INERT_PART_KEYS,
    INLINE_URL_KEYS,
    PART_KEYS,
    TAG_SETTLED_INPUT_TYPES,
    TOOL_ENVELOPE_INPUT_TYPE,
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


def test_body_policy_permits_the_agent_message_the_upstream_echoed_back():
    """The multi-agent assistant turn. Codex has no code that builds one -- it
    holds one because the upstream emitted it, and replays it verbatim into the
    next request. Its content is text or the model's own encrypted blob, and
    `reasoning`'s blob already rides through on the same body."""
    body = json.dumps(
        {
            "input": [
                {
                    "type": "agent_message",
                    "id": "amsg_1",
                    "author": "assistant",
                    "recipient": "user",
                    "content": [
                        {"type": "input_text", "text": "measured"},
                        {"type": "encrypted_content", "encrypted_content": "gAAAA"},
                    ],
                }
            ],
            "store": False,
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


def test_body_policy_permits_the_measured_tool_result_shapes():
    """A tool result carries its parts under `output`, not `content`, and the
    field is a bare string as often as it is an array. Measured: text parts,
    and the screenshot a local tool returns inline."""
    for output in [
        "done",
        [{"type": "input_text", "text": "done"}],
        [{"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}],
        [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBOR",
                "detail": "auto",
            }
        ],
    ]:
        for kind in ("function_call_output", "custom_tool_call_output"):
            body = json.dumps(
                {
                    "input": [{"type": kind, "call_id": "c", "output": output}],
                    "store": False,
                    "stream": True,
                }
            ).encode()
            assert BodyPolicy().refuse(body) is None, (kind, output)


@pytest.mark.parametrize("kind", sorted(TAG_SETTLED_INPUT_TYPES))
@pytest.mark.parametrize("field", ["content", "output", "summary"])
def test_a_tag_settled_item_may_not_carry_parts_at_all(kind, field):
    """The tag settles these because they hold no parts. If one ever does, the
    answer has to be a refusal and not a walk that never happens -- which is
    what made a fetch under `output` invisible."""
    body = json.dumps(
        {
            "input": [
                {
                    "type": kind,
                    field: [{"type": "input_image", "image_url": "https://evil/x"}],
                }
            ],
            "store": False,
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is not None


def test_body_policy_permits_the_measured_reasoning_shapes():
    """An empty summary and a null content, which is every reasoning item on
    record. Gated when present, not required to be."""
    for extra in [
        {"summary": []},
        {"summary": [], "content": None},
        {"summary": [{"type": "summary_text", "text": "s"}]},
        {"content": [{"type": "reasoning_text", "text": "r"}]},
    ]:
        body = json.dumps(
            {
                "input": [{"type": "reasoning", "encrypted_content": "x", **extra}],
                "store": False,
                "stream": True,
            }
        ).encode()
        assert BodyPolicy().refuse(body) is None, extra


def test_body_policy_permits_an_inline_image_in_either_container():
    """The same bytes in the same part type, so the same answer: a screenshot a
    tool returned and an image a user attached both ride, and neither may name
    a url for the upstream to fetch. Codex strips a remote one before the
    socket sees it; the box is not Codex."""
    inline = {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}
    remote = {"type": "input_image", "image_url": "https://evil.test/x"}
    for item, field in [
        ({"type": "message", "role": "user"}, "content"),
        ({"type": "custom_tool_call_output", "call_id": "c"}, "output"),
    ]:
        for part, permitted in [(inline, True), (remote, False)]:
            body = json.dumps(
                {
                    "input": [{**item, field: [part]}],
                    "store": False,
                    "stream": True,
                }
            ).encode()
            assert (BodyPolicy().refuse(body) is None) is permitted, (item, part)


def test_body_policy_permits_the_recorded_compaction_item():
    """A box compacts locally and never produces this, but a history recorded
    outside one carries it, and resuming that session inside a box replays it.
    An opaque blob, an id and the metadata every item may carry: the tag
    settles it."""
    body = json.dumps(
        {
            "input": [
                {
                    "type": "compaction",
                    "id": "cmp_1",
                    "encrypted_content": "gAAAA",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "t"},
                }
            ],
            "store": False,
            "stream": True,
        }
    ).encode()
    assert BodyPolicy().refuse(body) is None


def test_every_permitted_input_type_is_sorted_into_exactly_one_bucket():
    """Permitting a type means choosing how it is gated. The buckets are
    derived into ALLOWED_INPUT_TYPES rather than spelled beside it, so a type
    cannot arrive permitted without one -- which is how `function_call_output`
    carried an unread `image_url`."""
    buckets = [TAG_SETTLED_INPUT_TYPES, set(CONTAINERS), {TOOL_ENVELOPE_INPUT_TYPE}]
    assert not any(CONTAINERS.get(kind) for kind in TAG_SETTLED_INPUT_TYPES)
    assert set.union(*(set(b) for b in buckets)) == set(ALLOWED_INPUT_TYPES)
    for i, one in enumerate(buckets):
        for other in buckets[i + 1 :]:
            assert not set(one) & set(other), (one, other)


def test_every_part_a_container_holds_is_pinned_and_url_gated():
    """The gate reads a part's keys, so every part a container names needs a
    key set; and a key that names a url needs the inline check, or the part
    becomes the fetch the whole policy exists to refuse."""
    for containers in CONTAINERS.values():
        for container in containers:
            assert container.parts <= set(PART_KEYS), container
    for part_kind, keys in PART_KEYS.items():
        # Not a guess from the key's name: anything outside the inert
        # vocabulary has to be the key the inline gate reads, so a part added
        # with a `url`, `src` or `file_id` cannot slip in ungated.
        assert keys - INERT_PART_KEYS <= {INLINE_URL_KEYS.get(part_kind)}, part_kind
    assert set(INLINE_URL_KEYS) <= set(PART_KEYS)


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
        # The same fetch, moved into the item whose own union has no url in it.
        # Reached only because the content gate runs for `agent_message` too;
        # an item type permitted by tag alone would carry this straight
        # through.
        {
            "input": [
                {
                    "type": "agent_message",
                    "author": "assistant",
                    "recipient": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.test/x"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "agent_message",
                    "author": "assistant",
                    "recipient": "user",
                    "content": [{"type": "output_text", "text": "wrong union"}],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "encrypted_content", "encrypted_content": "gAAAA"}
                    ],
                }
            ]
        },
        # The fetch as it actually reached the allowlist: under `output`, in a
        # tool result, which the gate settled by its tag and never walked.
        {
            "input": [
                {
                    "type": "function_call_output",
                    "output": [
                        {"type": "input_image", "image_url": "https://evil.test/x"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "output": [
                        {"type": "input_image", "image_url": "https://evil.test/x"}
                    ],
                }
            ]
        },
        # A scheme that is not `data:` however it is spelled, and a part that
        # names no url at all where the type says it must.
        {
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "output": [
                        {"type": "input_image", "image_url": " data:image/png;base64,x"}
                    ],
                }
            ]
        },
        {
            "input": [
                {"type": "custom_tool_call_output", "output": [{"type": "input_image"}]}
            ]
        },
        # The tag says text, a second key names a payload. Refusing on the tag
        # alone is what lets these two answers ride in one part.
        {
            "input": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hi",
                            "image_url": "https://evil.test/x",
                        }
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "agent_message",
                    "author": "a",
                    "recipient": "u",
                    "content": [
                        {
                            "type": "encrypted_content",
                            "encrypted_content": "x",
                            "image_url": "https://evil.test/x",
                        }
                    ],
                }
            ]
        },
        # A part array in a field this item type does not keep parts in.
        {
            "input": [
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "input_image", "image_url": "https://evil.test/x"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "message",
                    "content": "hi",
                    "output": [
                        {"type": "input_image", "image_url": "https://evil.test/x"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "function_call_output",
                    "output": "ok",
                    "content": [
                        {"type": "input_image", "image_url": "https://evil.test/x"}
                    ],
                }
            ]
        },
        # Absent is not false, and the API's default for absent is to retain.
        {"store": None},
        # Tools ride in the envelope and nowhere else.
        {
            "input": [
                {
                    "type": "agent_message",
                    "author": "a",
                    "recipient": "u",
                    "content": [],
                    "tools": [{"type": "web_search"}],
                }
            ]
        },
    ],
)
def test_body_policy_refuses_unmeasured_response_capabilities(change):
    body = {"input": [], "tools": [], "store": False, "stream": True, **change}
    if body["store"] is None:
        del body["store"]
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
