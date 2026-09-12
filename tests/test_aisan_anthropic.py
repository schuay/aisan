# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.egress.anthropic import (
    API_KEY_ENV,
    OAUTH_TOKEN_ENV,
    PLACEHOLDER_KEY,
    PORT,
    SUBSCRIPTION_ENV,
    AnthropicBackend,
)
from aisan.egress.base import PreflightError

FAKE_TOKEN = "fake-access-token-for-tests"
FAKE_REFRESH_TOKEN = "fake-refresh-token-for-tests"


def _credentials(
    path: Path,
    *,
    token: str = FAKE_TOKEN,
    refresh_token: str | None = None,
    ttl_s: float = 3600,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": token,
                    "refreshToken": refresh_token or FAKE_REFRESH_TOKEN,
                    "expiresAt": int((time.time() + ttl_s) * 1000),
                    "subscriptionType": "max",
                }
            }
        )
    )
    path.chmod(0o600)
    return path


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


async def _hello(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def test_the_box_is_told_a_placeholder_and_a_route_and_nothing_else(tmp_path):
    env = AnthropicBackend(credentials=tmp_path / "c.json").client_env()
    assert env == {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        OAUTH_TOKEN_ENV: PLACEHOLDER_KEY,
    }

    assert "placeholder" in PLACEHOLDER_KEY


def test_an_api_key_backend_dresses_the_box_as_an_api_key_client(tmp_path):
    env = AnthropicBackend(api_key="sk-ant-not-real").client_env()
    assert env == {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        API_KEY_ENV: PLACEHOLDER_KEY,
    }

    assert AnthropicBackend(api_key="sk-ant-not-real").credentials == ()


def test_a_backend_holds_exactly_one_credential_kind(tmp_path):
    with pytest.raises(ValueError, match="at most one"):
        AnthropicBackend(credentials=tmp_path / "c.json", api_key="sk-ant-not-real")


def test_the_plan_label_is_read_best_effort_and_never_raises(tmp_path):
    missing = AnthropicBackend(credentials=tmp_path / "absent.json").client_env()
    assert SUBSCRIPTION_ENV not in missing

    junk = tmp_path / "junk.json"
    junk.write_text("not json at all")
    assert SUBSCRIPTION_ENV not in AnthropicBackend(credentials=junk).client_env()

    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({"claudeAiOauth": {"subscriptionType": 7}}))
    assert (
        SUBSCRIPTION_ENV not in AnthropicBackend(credentials=wrong_shape).client_env()
    )

    present = tmp_path / "max.json"
    present.write_text(json.dumps({"claudeAiOauth": {"subscriptionType": "max"}}))
    assert AnthropicBackend(credentials=present).client_env()[SUBSCRIPTION_ENV] == "max"


def test_the_port_is_distinct_from_the_other_backends():
    from aisan.egress.reapi import PORT as REAPI_PORT
    from aisan.egress.vertex import PORT as VERTEX_PORT

    assert len({PORT, REAPI_PORT, VERTEX_PORT}) == 3


async def test_preflight_passes_when_the_credential_and_upstream_are_both_there(
    tmp_path,
):
    up, runner = await _upstream_server(_hello)
    try:
        await AnthropicBackend(
            credentials=_credentials(tmp_path / "c.json"), upstream=up
        ).preflight()
    finally:
        await runner.cleanup()


async def test_a_missing_credential_refuses_with_the_login_command(tmp_path):
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await AnthropicBackend(
                credentials=tmp_path / "absent.json", upstream=up
            ).preflight()
    finally:
        await runner.cleanup()
    assert "claude auth login" in e.value.fix
    assert "absent.json" in e.value.reason


async def test_preflight_refreshes_an_expired_token_with_host_claude(
    tmp_path, monkeypatch
):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=-60)
    calls = []

    async def refresh(command, config_dir):
        calls.append((command, config_dir))
        _credentials(
            credentials,
            token="refreshed-access-token",
            refresh_token="rotated-refresh-token",
        )

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    up, runner = await _upstream_server(_hello)
    try:
        await AnthropicBackend(
            credentials=credentials,
            upstream=up,
            claude_command=("host-claude",),
        ).preflight()
    finally:
        await runner.cleanup()
    assert calls == [(("host-claude",), tmp_path)]
    data = json.loads(credentials.read_text())["claudeAiOauth"]
    assert data["accessToken"] == "refreshed-access-token"
    assert data["refreshToken"] == "rotated-refresh-token"


async def test_a_token_expiring_inside_the_margin_is_refreshed(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=60)

    async def refresh(command, config_dir):
        _credentials(credentials, token="fresh-before-startup")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    up, runner = await _upstream_server(_hello)
    try:
        await AnthropicBackend(credentials=credentials, upstream=up).preflight()
    finally:
        await runner.cleanup()


async def test_a_dead_upstream_says_start_it_not_log_in(tmp_path):
    up, runner = await _upstream_server(_hello)
    await runner.cleanup()
    with pytest.raises(PreflightError) as e:
        await AnthropicBackend(
            credentials=_credentials(tmp_path / "c.json"), upstream=up
        ).preflight()
    assert "does not answer" in e.value.reason
    assert "start the service" in e.value.fix
    assert "login" not in e.value.fix


async def test_no_route_is_reported_as_the_route_not_the_login(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    async def refresh(command, config_dir):
        return

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    up, runner = await _upstream_server(_hello)
    await runner.cleanup()
    with pytest.raises(PreflightError) as e:
        await AnthropicBackend(
            credentials=_credentials(tmp_path / "c.json", ttl_s=60), upstream=up
        ).preflight()
    assert "does not answer" in e.value.reason
    assert "login" not in e.value.fix


async def test_preflight_does_not_spend_a_model_call(tmp_path):
    seen: list[tuple[str, str]] = []

    async def upstream(request: web.Request) -> web.Response:
        seen.append((request.method, request.path))
        return web.json_response({"ok": True})

    up, runner = await _upstream_server(upstream)
    try:
        await AnthropicBackend(
            credentials=_credentials(tmp_path / "c.json"), upstream=up
        ).preflight()
    finally:
        await runner.cleanup()
    assert all(path != "/v1/messages" for _, path in seen), seen


async def test_a_credential_file_that_is_not_json_names_the_path_only(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text('{"claudeAiOauth": {"accessToken": "' + FAKE_TOKEN + '"')
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await AnthropicBackend(credentials=bad, upstream=up).preflight()
    finally:
        await runner.cleanup()
    assert FAKE_TOKEN not in str(e.value)
    assert str(bad) in str(e.value)


async def test_the_backend_serves_and_injects_the_credential_it_read(tmp_path):
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    try:
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            assert sock.exists()
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as s,
                s.post(
                    "http://anthropic.invalid/v1/messages",
                    data=b"{}",
                    headers={"x-api-key": PLACEHOLDER_KEY},
                ) as r,
            ):
                assert r.status == 200

        assert not backend.socket_path(runtime).exists()
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "x-api-key" not in got


async def test_shared_backend_serves_authenticated_tcp_with_per_box_state(tmp_path):
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    try:
        async with backend.serve_shared(tmp_path) as activation:
            endpoint = activation.client_env["ANTHROPIC_BASE_URL"]
            client_token = activation.client_env[OAUTH_TOKEN_ENV]
            assert activation.port == int(endpoint.rsplit(":", 1)[1])
            assert client_token.endswith(PLACEHOLDER_KEY[-20:])
            assert client_token != PLACEHOLDER_KEY

            async with (
                ClientSession() as session,
                session.post(
                    f"{endpoint}/v1/messages",
                    data=b"{}",
                    headers={"authorization": f"Bearer {client_token}"},
                ) as response,
            ):
                assert response.status == 200
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "x-api-key" not in got


async def test_one_backend_can_hold_two_shared_activations(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    try:
        async with (
            backend.serve_shared(tmp_path) as first,
            backend.serve_shared(tmp_path) as second,
        ):
            assert first.port != second.port
            assert (
                first.client_env[OAUTH_TOKEN_ENV] != second.client_env[OAUTH_TOKEN_ENV]
            )
    finally:
        await up_runner.cleanup()


async def test_the_backend_never_writes_the_credential(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    creds = _credentials(tmp_path / "c.json")
    before = creds.read_bytes()
    before_mtime = creds.stat().st_mtime_ns
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = AnthropicBackend(credentials=creds, upstream=up)
    try:
        await backend.preflight()
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as s,
                s.post("http://anthropic.invalid/v1/messages", data=b"{}") as r,
            ):
                assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert creds.read_bytes() == before
    assert creds.stat().st_mtime_ns == before_mtime

    assert creds.stat().st_mode & 0o777 == 0o600


async def test_a_stale_socket_from_a_killed_box_does_not_block_the_next_one(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    backend.socket_path(runtime).write_bytes(b"")
    try:
        async with backend.serve(runtime):
            assert backend.socket_path(runtime).is_socket()
    finally:
        await up_runner.cleanup()


async def test_the_upstream_credential_matches_the_dress(tmp_path):
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    backend = AnthropicBackend(api_key="sk-ant-not-real", upstream=up)
    try:
        async with backend.serve_shared(tmp_path) as activation:
            endpoint = activation.client_env["ANTHROPIC_BASE_URL"]
            client_token = activation.client_env[API_KEY_ENV]
            async with (
                ClientSession() as session,
                session.post(
                    f"{endpoint}/v1/messages",
                    data=b"{}",
                    headers={"x-api-key": client_token},
                ) as response,
            ):
                assert response.status == 200
    finally:
        await up_runner.cleanup()

    assert got["x-api-key"] == "sk-ant-not-real"

    assert got["x-api-key"] != client_token
    assert "authorization" not in got


async def test_either_dress_can_present_the_relay_token(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    try:
        async with backend.serve_shared(tmp_path) as activation:
            endpoint = activation.client_env["ANTHROPIC_BASE_URL"]
            token = activation.client_env[OAUTH_TOKEN_ENV]

            async def post(headers: dict[str, str]) -> int:
                async with (
                    ClientSession() as session,
                    session.post(
                        f"{endpoint}/v1/messages", data=b"{}", headers=headers
                    ) as response,
                ):
                    return response.status

            assert await post({"authorization": f"Bearer {token}"}) == 200
            assert await post({"x-api-key": token}) == 200
            assert await post({"authorization": "Bearer wrong"}) == 401
            assert await post({"x-api-key": "wrong"}) == 401
            assert await post({}) == 401
    finally:
        await up_runner.cleanup()


_WEB_SEARCH = {
    "model": "m",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "foo"}]}],
    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    "tool_choice": {"type": "tool", "name": "web_search"},
}


async def test_the_transport_decides_the_body_policy(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    body = json.dumps(_WEB_SEARCH).encode()
    runtime = tmp_path / "rt"
    runtime.mkdir()
    try:
        async with backend.serve_shared(tmp_path) as activation:
            endpoint = activation.client_env["ANTHROPIC_BASE_URL"]
            token = activation.client_env[OAUTH_TOKEN_ENV]
            async with (
                ClientSession() as session,
                session.post(
                    f"{endpoint}/v1/messages",
                    data=body,
                    headers={"authorization": f"Bearer {token}"},
                ) as response,
            ):
                assert response.status == 200

        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as session,
                session.post(
                    "http://anthropic.invalid/v1/messages", data=body
                ) as response,
            ):
                assert response.status == 403
                assert "tool_choice" in (await response.json())["error"]["message"]
    finally:
        await up_runner.cleanup()


async def test_a_token_that_expires_mid_session_is_refreshed(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append((request.path, request.headers.get("authorization")))
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    credentials = _credentials(tmp_path / "c.json")
    backend = AnthropicBackend(credentials=credentials, upstream=up)

    async def refresh(command, config_dir):
        _credentials(
            credentials,
            token="fresh-mid-session",
            refresh_token="rotated-mid-session",
        )

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    try:
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            async with ClientSession(connector=UnixConnector(path=str(sock))) as s:
                headers = {"x-api-key": PLACEHOLDER_KEY}
                async with s.post(
                    "http://anthropic.invalid/v1/messages", data=b"{}", headers=headers
                ) as r:
                    assert r.status == 200

                _credentials(credentials, ttl_s=-60)

                async with s.post(
                    "http://anthropic.invalid/v1/messages", data=b"{}", headers=headers
                ) as r:
                    assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert reached == [
        ("/v1/messages", f"Bearer {FAKE_TOKEN}"),
        ("/v1/messages", "Bearer fresh-mid-session"),
    ]
    assert (
        json.loads(credentials.read_text())["claudeAiOauth"]["refreshToken"]
        == "rotated-mid-session"
    )


async def test_concurrent_requests_launch_one_refresh(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=-60)
    calls = 0

    async def refresh(command, config_dir):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        _credentials(credentials, token="fresh-after-one-refresh")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)
    headers = await asyncio.gather(*(backend._upstream_auth() for _ in range(8)))

    assert calls == 1
    assert headers == [{"authorization": "Bearer fresh-after-one-refresh"}] * 8


async def test_a_still_valid_token_survives_a_failed_refresh(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", token="still-good", ttl_s=240)

    async def refresh(command, config_dir):
        raise OSError("no such file or directory: 'claude'")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)

    assert await backend._upstream_auth() == {"authorization": "Bearer still-good"}


async def test_an_expired_token_is_still_refused_when_refresh_fails(
    tmp_path, monkeypatch
):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=-60)

    async def refresh(command, config_dir):
        raise OSError("boom")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)

    with pytest.raises(backend_module.CredentialRefreshError, match="boom"):
        await backend._upstream_auth()


async def test_preflight_still_demands_the_whole_margin(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", token="short", ttl_s=240)

    async def refresh(command, config_dir):
        raise OSError("boom")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)

    with pytest.raises(PreflightError):
        await backend._credential.check("anthropic")


async def test_a_refresh_that_wrote_the_token_then_timed_out_counts(
    tmp_path, monkeypatch
):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=-60)

    async def refresh(command, config_dir):
        _credentials(credentials, token="written-before-the-timeout")
        raise TimeoutError("host Claude token refresh timed out")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)

    assert await backend._upstream_auth() == {
        "authorization": "Bearer written-before-the-timeout"
    }


async def test_a_failing_refresh_is_shared_by_the_whole_wave(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", token="still-good", ttl_s=240)
    calls = 0

    async def refresh(command, config_dir):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        raise OSError("host claude is not answering")

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)
    headers = await asyncio.gather(*(backend._upstream_auth() for _ in range(8)))

    assert calls == 1
    assert headers == [{"authorization": "Bearer still-good"}] * 8


async def test_a_refresh_that_changed_nothing_does_not_blame_the_login(
    tmp_path, monkeypatch
):
    import aisan.egress.anthropic as backend_module

    credentials = _credentials(tmp_path / "c.json", ttl_s=-60)

    async def refresh(command, config_dir):
        return None

    monkeypatch.setattr(backend_module, "_refresh_claude_login", refresh)
    backend = AnthropicBackend(credentials=credentials)

    with pytest.raises(backend_module.CredentialRefreshError) as caught:
        await backend._upstream_auth()
    assert "apiKeyHelper" in str(caught.value)


def test_proxy_bypass_keeps_both_spellings_of_no_proxy():
    from aisan.egress.anthropic import _merge_proxy_bypass

    assert _merge_proxy_bypass("", "corp.example,10.0.0.1", "127.0.0.1") == [
        "corp.example",
        "10.0.0.1",
        "127.0.0.1",
    ]

    assert _merge_proxy_bypass("a, b", "b", "a") == ["a", "b"]


async def test_nonzero_claude_status_is_accepted_after_refresh_and_trapped_locally(
    tmp_path, monkeypatch
):
    credentials = _credentials(tmp_path / ".credentials.json", ttl_s=-60)
    marker = tmp_path / "refresh-result.json"
    script = tmp_path / "fake-claude.py"
    script.write_text(
        """\
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

credential = Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".credentials.json"
data = json.loads(credential.read_text())
data["claudeAiOauth"]["accessToken"] = "subprocess-access-token"
data["claudeAiOauth"]["refreshToken"] = "subprocess-refresh-token"
data["claudeAiOauth"]["expiresAt"] = int((time.time() + 3600) * 1000)
replacement = credential.with_suffix(".new")
replacement.write_text(json.dumps(data))
replacement.replace(credential)

request = urllib.request.Request(
    os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
    data=b"not a real model request",
    method="POST",
)
try:
    urllib.request.urlopen(request, timeout=3)
    status = 200
except urllib.error.HTTPError as error:
    status = error.code

Path(os.environ["AISAN_REFRESH_TEST_MARKER"]).write_text(json.dumps({
    "argv": sys.argv[1:],
    "base_url": os.environ["ANTHROPIC_BASE_URL"],
    "config_dir": os.environ["CLAUDE_CONFIG_DIR"],
    "status": status,
    "api_key_present": "ANTHROPIC_API_KEY" in os.environ,
    "oauth_token_present": "CLAUDE_CODE_OAUTH_TOKEN" in os.environ,
    "cwd": os.getcwd(),
    "cwd_entries": sorted(os.listdir(os.getcwd())),
    "path_env": {name: os.environ.get(name) for name in
                 ("PWD", "TMPDIR", "TMP", "TEMP", "OLDPWD", "INIT_CWD")},
}))
raise SystemExit(23)
"""
    )
    monkeypatch.setenv("AISAN_REFRESH_TEST_MARKER", str(marker))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-host-claude")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "must-not-reach-host-claude")
    launcher_cwd = tmp_path / "the-repo-the-agent-edited"
    (launcher_cwd / ".claude").mkdir(parents=True)
    (launcher_cwd / "CLAUDE.md").write_text("steering the agent wrote\n")
    monkeypatch.chdir(launcher_cwd)

    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, runner = await _upstream_server(upstream)
    try:
        await AnthropicBackend(
            credentials=credentials,
            upstream=up,
            claude_command=(sys.executable, str(script)),
        ).preflight()
    finally:
        await runner.cleanup()

    result = json.loads(marker.read_text())
    assert result == {
        "argv": [
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            "haiku",
            "-p",
            "Reply with exactly hello.",
        ],
        "base_url": result["base_url"],
        "config_dir": str(tmp_path),
        "status": 502,
        "api_key_present": False,
        "oauth_token_present": False,
        "cwd": result["cwd"],
        "cwd_entries": [],
        "path_env": {
            "PWD": result["cwd"],
            "TMPDIR": result["cwd"],
            "TMP": result["cwd"],
            "TEMP": result["cwd"],
            "OLDPWD": None,
            "INIT_CWD": None,
        },
    }

    assert result["cwd"] != str(launcher_cwd)
    assert result["base_url"].startswith("http://127.0.0.1:")
    assert reached == ["/api/hello"]
    oauth = json.loads(credentials.read_text())["claudeAiOauth"]
    assert oauth["accessToken"] == "subprocess-access-token"
    assert oauth["refreshToken"] == "subprocess-refresh-token"


async def test_nonzero_claude_status_with_a_stale_credential_fails(tmp_path):
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as error:
            await AnthropicBackend(
                credentials=_credentials(tmp_path / "c.json", ttl_s=-60),
                upstream=up,
                claude_command=(sys.executable, "-c", "raise SystemExit(9)"),
            ).preflight()
    finally:
        await runner.cleanup()
    assert "host Claude ran" in error.value.reason
    assert "expired" in error.value.reason
    assert FAKE_TOKEN not in str(error.value)


async def test_missing_claude_executable_has_a_host_login_fix(tmp_path):
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as error:
            await AnthropicBackend(
                credentials=_credentials(tmp_path / "c.json", ttl_s=-60),
                upstream=up,
                claude_command=("aisan-test-no-such-claude",),
            ).preflight()
    finally:
        await runner.cleanup()
    assert "could not run host Claude" in error.value.reason
    assert "claude auth login" in error.value.fix


async def test_claude_refresh_timeout_has_a_host_login_fix(tmp_path, monkeypatch):
    import aisan.egress.anthropic as backend_module

    monkeypatch.setattr(backend_module, "_REFRESH_TIMEOUT_S", 0.01)
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as error:
            await AnthropicBackend(
                credentials=_credentials(tmp_path / "c.json", ttl_s=-60),
                upstream=up,
                claude_command=(
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ),
            ).preflight()
    finally:
        await runner.cleanup()
    assert "timed out" in error.value.reason
    assert "claude auth login" in error.value.fix


async def test_one_request_reads_the_credential_file_exactly_once(
    tmp_path, monkeypatch
):
    import aisan.egress.anthropic as backend_module

    reads = []
    real = backend_module._read_oauth

    def counted(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(backend_module, "_read_oauth", counted)

    up, up_runner = await _upstream_server(_hello)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    try:
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            reads.clear()
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as s,
                s.post(
                    "http://anthropic.invalid/v1/messages",
                    data=b"{}",
                    headers={"x-api-key": PLACEHOLDER_KEY},
                ) as r,
            ):
                assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert len(reads) == 1
