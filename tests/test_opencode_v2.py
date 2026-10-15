# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Live compatibility checks for the measured OpenCode V2 client."""

import asyncio
import dataclasses
import json
import os
import subprocess

import pytest
from aiohttp import web

from aisan import Box
from aisan.egress.openai_compat import OpenAICompatBackend
from aisan.opencode_config import read_config
from aisan.opencode_provider import SDK, resolve_provider
from aisan.presets.opencode import opencode, opencode_binary
from aisan.session_mcp import MCPAllowlist, opencode_host_mcp

pytestmark = pytest.mark.live
MARKER = "AISAN_TEST_CONTENT"


@pytest.fixture
def v2_binary(tmp_path):
    binary = opencode_binary()
    if binary is None:
        pytest.skip("opencode is not installed")
    version = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env={**os.environ, "XDG_DATA_HOME": str(tmp_path / "version-data")},
    ).stdout.strip()
    if version != "opencode v2.0.29":
        pytest.skip(
            f"custom-provider compatibility baseline is 2.0.29; installed {version}"
        )
    return binary


async def sse(request, delta, finish, *, usage=None):
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for content, reason in ((delta, None), ({}, finish)):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "example-model",
            "choices": [{"index": 0, "delta": content, "finish_reason": reason}],
        }
        if reason is not None and usage is not None:
            chunk["usage"] = usage
        await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def server(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


def spec_for(
    tmp_path,
    upstream,
    *,
    shared=False,
    auth="none",
    conflict="provider",
    mcp=False,
    model="example-model",
    context=16384,
    output=2048,
):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "probe.txt").write_text(MARKER + "\n")
    options = "options" if conflict == "provider" else "settings"
    (worktree / "opencode.json").write_text(
        json.dumps(
            {
                conflict: {
                    "local": {
                        options: {
                            "baseURL": "http://127.0.0.1:1",
                            "apiKey": "project-key",
                        }
                    }
                }
            }
        )
    )
    state = tmp_path / "state"
    state.mkdir()
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    source = tmp_path / "host.json"
    document = {
        "provider": {
            "local": {
                "npm": SDK,
                "options": {"baseURL": upstream + "/v1", "apiKey": "ignored-host-key"},
                "models": {
                    "alias": {
                        "id": model,
                        "limit": {"context": context, "output": output},
                    }
                },
            }
        }
    }
    extra_env = ()
    if mcp:
        script = worktree / "mcp.py"
        script.write_text("""import json, sys
from pathlib import Path
for line in sys.stdin:
    message = json.loads(line)
    Path("mcp-started").write_text(message.get("method", ""))
    if "id" not in message:
        continue
    method = message["method"]
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "demo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "Return a fixed test value", "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "MCP_TEST_RESULT"}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
""")
        document["mcp"] = {
            "demo": {"type": "local", "command": ["/usr/bin/python3", str(script)]}
        }
    source.write_text(json.dumps(document))
    snapshot = read_config(source, strict=True)
    selected = resolve_provider(
        "local", "alias", source=source, document=snapshot, catalog=catalog
    )
    if mcp:
        imported = opencode_host_mcp(
            source, document=snapshot, allow=MCPAllowlist(("demo",))
        )
        mcp_file = state / "mcp.json"
        imported.write(mcp_file)
        extra_env = (("OPENCODE_CONFIG", str(mcp_file)),)
    credentials = tmp_path / "auth.json"
    if auth == "api-key":
        credentials.write_text('{"local":{"type":"api","key":"host-test-key"}}')
    backend = OpenAICompatBackend(
        provider="local",
        model="alias",
        resolved=selected,
        upstream_auth=auth,
        credentials=credentials,
    )
    spec = opencode(
        worktree,
        state=state,
        models=catalog,
        egress=(backend,),
        unshare_net=not shared,
        extra_env=extra_env,
    )
    return (
        dataclasses.replace(
            spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
        ),
        source,
        credentials,
    )


@pytest.mark.parametrize(
    "shared,auth,conflict,mcp",
    [
        (False, "none", "provider", False),
        (False, "api-key", "providers", False),
        (True, "none", "providers", False),
        (False, "none", "providers", True),
    ],
)
async def test_custom_v2_tool_roundtrip(
    tmp_path, v2_binary, shared, auth, conflict, mcp
):
    requests = []
    title_seen = asyncio.Event()

    async def upstream(request):
        assert request.headers.get("Authorization") == (
            "Bearer host-test-key"
            if auth == "api-key" and request.method == "POST"
            else None
        )
        if request.method == "GET":
            assert request.path == "/v1/models"
            return web.Response(status=404)
        assert request.path == "/v1/chat/completions"
        body = await request.json()
        requests.append(body)
        (tmp_path / "requests.json").write_text(json.dumps(requests, indent=2))
        assert body["model"] == "example-model"
        replay = [
            message for message in body["messages"] if message.get("role") == "tool"
        ]
        if not body.get("tools"):
            title_seen.set()
        if body.get("tools") and not replay:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps(
                                {"path": str(tmp_path / "wt/probe.txt")}
                            ),
                        },
                    }
                ],
            }
            return await sse(request, delta, "tool_calls")
        return await sse(
            request, {"role": "assistant", "content": "AISAN_COMPLETE"}, "stop"
        )

    up, runner = await server(upstream)
    spec, source, credentials = spec_for(
        tmp_path, up, shared=shared, auth=auth, conflict=conflict, mcp=mcp
    )
    try:
        async with Box(spec, box_id="v2-test") as box:
            script = (
                "import json,socket; from pathlib import Path; "
                f"assert not Path({str(source)!r}).exists(); assert not Path({str(credentials)!r}).exists(); "
                "s=socket.socket(); s.settimeout(2); "
                f"print(json.dumps(s.connect_ex(('127.0.0.1',{runner.addresses[0][1]}))))"
            )
            check = await asyncio.to_thread(
                subprocess.run,
                box.command(["/usr/bin/python3", "-c", script]),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env={**os.environ, **box.env},
            )
            assert check.returncode == 0, check.stderr
            assert (json.loads(check.stdout) == 0) is shared
            result = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    [
                        str(v2_binary),
                        "run",
                        "--standalone",
                        "--auto",
                        "--format",
                        "json",
                        "Read probe.txt using the read tool, then answer.",
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env={**os.environ, **box.env},
            )
            assert result.returncode == 0, result.stdout + result.stderr
            await asyncio.wait_for(title_seen.wait(), timeout=10)
        tool_messages = [
            message
            for body in requests
            for message in body["messages"]
            if message.get("role") == "tool"
        ]
        assert any(MARKER in message["content"] for message in tool_messages), requests
        assert any(body.get("max_completion_tokens") == 2048 for body in requests)
        assert all(body.get("store") is False for body in requests)
        if mcp:
            assert (spec.root / "mcp-started").read_text() == "tools/list"
    finally:
        await runner.cleanup()


async def test_custom_v2_tui_exit(tmp_path, v2_binary):
    import shlex

    replayed = asyncio.Event()

    async def upstream(request):
        if request.method == "GET":
            return web.Response(status=404)
        body = await request.json()
        replay = [m for m in body["messages"] if m.get("role") == "tool"]
        if replay:
            assert any(MARKER in m["content"] for m in replay)
            replayed.set()
        if body.get("tools") and not replay:
            return await sse(
                request,
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps(
                                    {"path": str(tmp_path / "wt/probe.txt")}
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
            )
        return await sse(
            request, {"role": "assistant", "content": "AISAN_COMPLETE"}, "stop"
        )

    up, runner = await server(upstream)
    spec, _, _ = spec_for(tmp_path, up)
    spec = dataclasses.replace(spec, env=(*spec.env, ("TERM", "xterm-256color")))
    try:
        async with Box(spec, box_id="v2-tui") as box:
            command = shlex.join(
                [
                    str(v2_binary),
                    "--standalone",
                    "--auto",
                    "--prompt",
                    "Read probe.txt with the read tool, then answer.",
                ]
            )
            command = (
                "stty -g > tui-before; "
                + command
                + '; tui_status=$?; stty -g > tui-after; exit "$tui_status"'
            )
            proc = await asyncio.create_subprocess_exec(
                *box.command(
                    ["/usr/bin/script", "-q", "-e", "-c", command, "/dev/null"]
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **box.env},
            )
            output = asyncio.create_task(proc.stdout.read())
            try:
                await asyncio.wait_for(replayed.wait(), timeout=30)
                await asyncio.sleep(1)
                proc.stdin.write(b"\x03")
                await proc.stdin.drain()
                await asyncio.sleep(0.5)
                if proc.returncode is None:
                    proc.stdin.write(b"\x03")
                    await proc.stdin.drain()
                await asyncio.wait_for(proc.wait(), timeout=10)
                assert proc.returncode == 0
                assert (spec.root / "tui-before").read_text() == (
                    spec.root / "tui-after"
                ).read_text()
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                (tmp_path / "terminal.txt").write_bytes(await output)
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("automatic_title", [False, True])
async def test_confined_ollama_turn(tmp_path, v2_binary, automatic_title):
    from aiohttp import ClientSession, ClientTimeout
    from yarl import URL

    from aisan.upstream import validate_upstream

    settings = {
        name: os.environ.get(f"AISAN_TEST_OLLAMA_{name}")
        for name in ("URL", "MODEL", "CONTEXT", "OUTPUT")
    }
    if not all(settings.values()):
        pytest.skip("set AISAN_TEST_OLLAMA_{URL,MODEL,CONTEXT,OUTPUT} to opt in")
    target = validate_upstream(settings["URL"])
    context = int(settings["CONTEXT"])
    output = int(settings["OUTPUT"])
    records = []

    async def upstream(request):
        assert "Authorization" not in request.headers
        if request.method == "GET":
            return web.Response(status=404)
        body = await request.json()
        records.append(body)
        index = len(records)
        (tmp_path / "requests.json").write_text(json.dumps(records, indent=2))
        async with (
            ClientSession(timeout=ClientTimeout(total=150)) as session,
            session.post(
                target + request.path, json=body, allow_redirects=False
            ) as response,
        ):
            (tmp_path / f"response-{index}-headers.json").write_text(
                json.dumps(
                    {"status": response.status, "headers": dict(response.headers)},
                    indent=2,
                )
            )
            assert response.status == 200
            assert response.content_type == "text/event-stream"
            result = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await result.prepare(request)
            with (tmp_path / f"response-{index}.sse").open("wb") as capture:
                async for chunk in response.content.iter_any():
                    capture.write(chunk)
                    await result.write(chunk)
            await result.write_eof()
            return result

    up, runner = await server(upstream)
    spec, _, _ = spec_for(
        tmp_path, up, model=settings["MODEL"], context=context, output=output
    )
    try:
        async with Box(spec, box_id="v2-ollama") as box:
            target_url = URL(target)
            check = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    [
                        "/usr/bin/python3",
                        "-c",
                        (
                            "import json,socket; s=socket.socket(); s.settimeout(2); "
                            f"print(json.dumps(s.connect_ex(({target_url.host!r},{target_url.port!r}))))"
                        ),
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env={**os.environ, **box.env},
            )
            assert check.returncode == 0, check.stderr
            assert json.loads(check.stdout) != 0
            (tmp_path / "direct.json").write_text(check.stdout)
            result = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    [
                        str(v2_binary),
                        "run",
                        "--standalone",
                        "--auto",
                        *(
                            []
                            if automatic_title
                            else ["--title", "Aisan Ollama acceptance"]
                        ),
                        "--format",
                        "json",
                        "Read probe.txt using the read tool, then reply with its exact contents. Do not run other tools.",
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                env={**os.environ, **box.env},
            )
            (tmp_path / "stdout.jsonl").write_text(result.stdout)
            (tmp_path / "stderr.txt").write_text(result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
        assert any(
            MARKER in m.get("content", "")
            for b in records
            for m in b["messages"]
            if m.get("role") == "tool"
        )
        assert any(b.get("max_completion_tokens") == output for b in records)
    finally:
        await runner.cleanup()


async def test_custom_v2_automatic_compaction(tmp_path, v2_binary):
    requests = []

    async def upstream(request):
        if request.method == "GET":
            return web.Response(status=404)
        body = await request.json()
        requests.append(body)
        (tmp_path / "requests.json").write_text(json.dumps(requests, indent=2))
        if len(requests) == 1:
            return await sse(
                request,
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps(
                                    {"path": str(tmp_path / "wt/probe.txt")}
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
                usage={
                    "prompt_tokens": 16000,
                    "completion_tokens": 1,
                    "total_tokens": 16001,
                },
            )
        content = "The file was read. The task is complete."
        if len(requests) == 2:
            content = """## Objective
- Read probe.txt and answer.
## Requirements
- (none)
## Decisions
- (none)
## Work State
### Completed
- Read probe.txt: AISAN_TEST_CONTENT.
### Active
- Answer the user.
### Blocked
- (none)
## Next Move
1. Reply with the file contents.
## Relevant Files
- (none)
## Important Context
- (none)
"""
        return await sse(
            request,
            {"role": "assistant", "content": content},
            "stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        )

    up, runner = await server(upstream)
    spec, _, _ = spec_for(tmp_path, up)
    try:
        async with Box(spec, box_id="v2-compact") as box:
            result = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    [
                        str(v2_binary),
                        "run",
                        "--standalone",
                        "--auto",
                        "--title",
                        "Compaction acceptance",
                        "--format",
                        "json",
                        "Read probe.txt, then answer.",
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env={**os.environ, **box.env},
            )
            (tmp_path / "stdout.jsonl").write_text(result.stdout)
            assert result.returncode == 0, result.stdout + result.stderr
        assert len(requests) >= 3
        assert requests[1]["messages"][-1]["content"].startswith(
            "You MUST summarize the conversation above"
        )
        assert all(request["model"] == "example-model" for request in requests)
    finally:
        await runner.cleanup()
