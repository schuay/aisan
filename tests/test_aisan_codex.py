# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Codex backend and preset, including one real boxed client turn."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan import Box
from aisan.egress.base import PLACEHOLDER_KEY, PreflightError
from aisan.egress.openai_responses import (
    CLIENT_KEY_ENV,
    PORT,
    CodexBackend,
    _refresh_chatgpt_login,
)
from aisan.presets import PRESETS
from aisan.presets.codex import codex, codex_argv, codex_binary, codex_default
from aisan.sandbox import RO, RW

FAKE_ACCOUNT = "fake-chatgpt-account-for-tests"


def _jwt(expires_at: int) -> str:
    def part(value: dict) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
        )

    return f"{part({'alg': 'none'})}.{part({'exp': expires_at})}.signature"


def _auth(path: Path, *, expires_at: int = 4_102_444_800) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": _jwt(expires_at),
                    "access_token": _jwt(expires_at),
                    "refresh_token": "host-refresh-token-for-tests",
                    "account_id": FAKE_ACCOUNT,
                },
            }
        )
    )
    path.chmod(0o600)
    return path


def _backend(tmp_path: Path, **kw) -> CodexBackend:
    if "credentials" not in kw:
        kw["credentials"] = _auth(tmp_path / "auth.json")
    return CodexBackend(**kw)


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}/v1", runner


async def _hello(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def test_cli_config_pins_the_measured_route_and_disables_other_egress(tmp_path):
    backend = _backend(tmp_path, model="gpt-test", upstream="https://example.test/v1")
    assert backend.config_overrides() == (
        'model_provider="aisan"',
        'web_search="disabled"',
        "check_for_update_on_startup=false",
        "analytics.enabled=false",
        "feedback.enabled=false",
        "features.apps=false",
        'model_providers.aisan.name="aisan host proxy"',
        f'model_providers.aisan.base_url="http://127.0.0.1:{PORT}"',
        f'model_providers.aisan.env_key="{CLIENT_KEY_ENV}"',
        'model_providers.aisan.wire_api="responses"',
        "model_providers.aisan.requires_openai_auth=false",
        "model_providers.aisan.supports_websockets=false",
        "model_providers.aisan.supports_standalone_web_search=false",
        'model="gpt-test"',
    )
    assert backend.box_binds(tmp_path / "runtime") == []
    assert backend.client_env() == {CLIENT_KEY_ENV: PLACEHOLDER_KEY}


async def test_api_key_login_is_refused(tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "secret"}))
    backend = _backend(tmp_path, credentials=auth)
    with pytest.raises(PreflightError) as error:
        await backend.preflight()
    assert "not a ChatGPT subscription login" in error.value.reason
    assert "codex login" in error.value.fix
    assert "secret" not in str(error.value)


async def test_near_expiry_login_is_refreshed_by_host_codex(tmp_path, monkeypatch):
    import aisan.egress.openai_responses as codex_egress

    auth = _auth(tmp_path / "auth.json", expires_at=1)
    upstream, runner = await _upstream_server(_hello)
    calls = []

    async def refresh(command, codex_home):
        calls.append((command, codex_home))
        _auth(auth)

    monkeypatch.setattr(codex_egress, "_refresh_chatgpt_login", refresh)
    backend = _backend(
        tmp_path,
        credentials=auth,
        upstream=upstream,
        codex_command=("host-codex",),
    )
    try:
        await backend.preflight()
    finally:
        await runner.cleanup()
    assert calls == [(("host-codex",), auth.parent)]


async def test_host_codex_refresh_uses_the_managed_account_rpc(tmp_path):
    script = tmp_path / "fake-codex.py"
    capture = tmp_path / "codex-home" / "capture.json"
    capture.parent.mkdir()
    script.write_text(
        "import json, os, pathlib, sys\n"
        "seen = []\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    seen.append(message)\n"
        "    if message.get('id') == 1:\n"
        "        pathlib.Path(os.environ['CODEX_HOME'], 'capture.json').write_text(\n"
        "            json.dumps(seen))\n"
        "        print(json.dumps({'id': 1, 'result': {\n"
        "            'account': {'type': 'chatgpt'}}}), flush=True)\n"
    )

    await _refresh_chatgpt_login((sys.executable, str(script)), capture.parent)

    messages = json.loads(capture.read_text())
    assert [message["method"] for message in messages] == [
        "initialize",
        "initialized",
        "account/read",
    ]
    assert messages[-1]["params"] == {"refreshToken": True}


async def test_backend_injects_host_subscription_and_preserves_upstream_prefix(
    tmp_path,
):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(
            (
                request.path,
                request.headers["authorization"],
                request.headers["chatgpt-account-id"],
                request.headers["originator"],
            )
        )
        return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

    upstream_url, upstream_runner = await _upstream_server(upstream)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    backend = _backend(tmp_path, upstream=upstream_url)
    body = json.dumps(
        {"input": [], "tools": [], "store": False, "stream": True}
    ).encode()
    try:
        async with (
            backend.serve(runtime),
            ClientSession(
                connector=UnixConnector(path=str(backend.socket_path(runtime)))
            ) as session,
            session.post("http://codex.invalid/responses", data=body) as response,
        ):
            assert response.status == 200
    finally:
        await upstream_runner.cleanup()
    assert reached == [
        (
            "/v1/responses",
            f"Bearer {_jwt(4_102_444_800)}",
            FAKE_ACCOUNT,
            "codex_cli_rs",
        )
    ]


async def test_shared_backend_serves_the_activated_tcp_endpoint(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

    upstream_url, upstream_runner = await _upstream_server(upstream)
    backend = _backend(tmp_path, upstream=upstream_url)
    body = json.dumps(
        {"input": [], "tools": [], "store": False, "stream": True}
    ).encode()
    try:
        async with backend.serve_shared(tmp_path) as activation:
            client_token = activation.client_env[CLIENT_KEY_ENV]
            async with (
                ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{activation.port}/responses",
                    data=body,
                    headers={"Authorization": f"Bearer {client_token}"},
                ) as response,
            ):
                assert response.status == 200
    finally:
        await upstream_runner.cleanup()
    assert reached == ["/v1/responses"]


def test_codex_preset_is_registered_and_uses_isolated_state(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    config = tmp_path / "git-config"
    config.mkdir()
    backend = _backend(tmp_path)
    spec = codex(worktree, state=state, egress=(backend,), extra_ro=(config,))

    assert PRESETS["codex"] is codex_default
    assert dict(spec.env)["CODEX_HOME"] == str(state)
    assert [(bind.path, bind.mode) for bind in spec.binds] == [
        (config, RO),
        (state, RW),
    ]
    assert codex_argv(("exec", "hello"), overrides=('model_provider="aisan"',)) == [
        "codex",
        "--dangerously-bypass-approvals-and-sandbox",
        "--config",
        'model_provider="aisan"',
        "exec",
        "hello",
    ]


def test_responses_port_is_distinct_from_every_existing_backend():
    from aisan.egress.anthropic import PORT as ANTHROPIC_PORT
    from aisan.egress.openai_compat import PORT as COMPAT_PORT
    from aisan.egress.reapi import PORT as REAPI_PORT
    from aisan.egress.vertex import PORT as VERTEX_PORT

    assert len({PORT, ANTHROPIC_PORT, COMPAT_PORT, REAPI_PORT, VERTEX_PORT}) == 5


async def _run_in_box(spec, script: str) -> subprocess.CompletedProcess:
    box = Box(spec, box_id="codex-test")
    async with box:
        return await asyncio.to_thread(
            subprocess.run,
            box.command([sys.executable, "-c", script]),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, **box.env},
        )


async def test_subscription_credential_is_absent_inside_a_real_box(tmp_path):
    upstream, runner = await _upstream_server(_hello)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(upstream=upstream, credentials=credential)
    spec = dataclasses.replace(
        codex(worktree, state=state, egress=(backend,)),
        limits=dataclasses.replace(
            codex(worktree, state=state, egress=(backend,)).limits,
            use_cgroup=False,
        ),
    )
    script = (
        "import os\n"
        f"print('credential=' + str(os.path.exists({str(credential)!r})))\n"
        f"print('client_key=' + os.environ[{CLIENT_KEY_ENV!r}])\n"
    )
    try:
        result = await _run_in_box(spec, script)
    finally:
        await runner.cleanup()
    assert result.returncode == 0, result.stderr
    assert "credential=False" in result.stdout
    assert f"client_key={PLACEHOLDER_KEY}" in result.stdout


@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_codex_tui_can_persist_repository_trust(tmp_path):
    upstream, runner = await _upstream_server(_hello)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(upstream=upstream, credentials=credential)
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    key_path = f"projects.{json.dumps(str(worktree))}.trust_level"
    initialize = {
        "method": "initialize",
        "id": 0,
        "params": {"clientInfo": {"name": "aisan-test", "version": "1"}},
    }
    write = {
        "method": "config/batchWrite",
        "id": 1,
        "params": {
            "edits": [
                {
                    "keyPath": key_path,
                    "value": "trusted",
                    "mergeStrategy": "replace",
                }
            ],
            "filePath": None,
            "expectedVersion": None,
            "reloadUserConfig": True,
        },
    }
    box = Box(spec, box_id="codex-config-write")
    try:
        async with box:
            process = await asyncio.create_subprocess_exec(
                *box.command(
                    codex_argv(
                        ("app-server", "--stdio"),
                        overrides=backend.config_overrides(),
                    )
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **box.env},
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write((json.dumps(initialize) + "\n").encode())
                await process.stdin.drain()
                initialized = json.loads(
                    await asyncio.wait_for(process.stdout.readline(), timeout=60)
                )
                assert initialized.get("id") == 0, initialized

                requests = (
                    {"method": "initialized", "params": {}},
                    write,
                )
                for request in requests:
                    process.stdin.write((json.dumps(request) + "\n").encode())
                await process.stdin.drain()
                while True:
                    response = json.loads(
                        await asyncio.wait_for(process.stdout.readline(), timeout=60)
                    )
                    if response.get("id") == 1:
                        break
            finally:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    # The write response is the operation's completion boundary.
                    # app-server can keep background work alive after stdin EOF;
                    # stopping that idle process is cleanup, not a failed write.
                    process.terminate()
                    await process.wait()
    finally:
        await runner.cleanup()

    assert response["result"]["status"] == "ok"
    config = tomllib.loads((state / "config.toml").read_text())
    assert config["projects"][str(worktree)]["trust_level"] == "trusted"


@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_real_codex_reaches_a_stub_only_through_the_responses_backend(
    tmp_path,
):
    marker = "aisan-codex-e2e-refusal-marker"
    seen = []

    async def upstream(request: web.Request) -> web.Response:
        seen.append((request.method, request.path))
        if request.method == "GET":
            return web.json_response({"ok": True})
        return web.json_response(
            {"error": {"message": marker, "type": "invalid_request_error"}},
            status=400,
        )

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text(
        'model_provider = "box-chosen"\n\n'
        "[model_providers.box-chosen]\n"
        'name = "box-chosen"\n'
        'base_url = "http://127.0.0.1:1"\n'
        'wire_api = "responses"\n'
    )
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(upstream=upstream_url, credentials=credential)
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="codex-e2e")
    try:
        async with box:
            result = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    codex_argv(
                        ("exec", "--skip-git-repo-check", "say hi"),
                        overrides=backend.config_overrides(),
                    )
                ),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env={**os.environ, **box.env},
            )
    finally:
        await runner.cleanup()
    assert result.returncode != 0
    assert marker in result.stdout + result.stderr
    assert ("POST", "/v1/responses") in seen
