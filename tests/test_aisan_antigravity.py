# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web
from conftest import needs_bwrap

from aisan import Box
from aisan.egress.antigravity import (
    CLOUD_CODE_URL_ENV,
    PORT,
    USER_INFO_URL_ENV,
    AntigravityBackend,
    box_token_file,
)
from aisan.egress.base import (
    PLACEHOLDER_KEY,
    SHARED_PORT_MARKER,
    PreflightError,
)
from aisan.presets import PRESETS
from aisan.presets.antigravity import (
    TOKEN_FILE,
    agy_argv,
    agy_binary,
    antigravity,
    antigravity_default,
)
from aisan.proxy.antigravity import USERINFO, Profile
from aisan.sandbox import RO, RW, Bind, BindOver
from aisan.spec import BoxSpec, Limits

HOST_TOKEN = "host-token-for-tests"
SEARCH = {
    "project": "aicode-consumers",
    "model": "gemini-3.5-flash-lite",
    "userAgent": "antigravity",
    "requestType": "web_search",
    "request": {
        "contents": [{"role": "user", "parts": [{"text": "q"}]}],
        "tools": [{"googleSearch": {}}],
    },
}


class _Credential:
    files: tuple[Path, ...] = ()

    def __init__(self, failure: PreflightError | None = None) -> None:
        self._failure = failure

    async def token(self) -> str:
        return HOST_TOKEN

    async def check(self, backend: str) -> None:
        if self._failure is not None:
            raise self._failure


async def _upstream_server(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


def _backend(tmp_path: Path, upstream: str = "http://127.0.0.1:1", **kw):
    kw.setdefault("credential", _Credential())
    kw.setdefault("token_file", tmp_path / "state" / "antigravity-oauth-token")
    return AntigravityBackend(profile=Profile(upstream=upstream), **kw)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_the_box_token_file_is_the_shape_the_client_accepted():
    assert json.loads(box_token_file("box-token")) == {
        "token": {
            "access_token": "box-token",
            "token_type": "Bearer",
            "refresh_token": PLACEHOLDER_KEY,
            "expiry": "2099-01-01T00:00:00.000000000+00:00",
            "expires_in": 3599,
        },
        "auth_method": "consumer",
    }


def test_the_client_is_pointed_at_the_loopback_proxy(tmp_path):
    backend = _backend(tmp_path)
    assert backend.client_env() == {
        CLOUD_CODE_URL_ENV: f"http://127.0.0.1:{PORT}",
        USER_INFO_URL_ENV: f"http://127.0.0.1:{PORT}/userinfo",
    }
    assert backend.shared_client_env_description() == {
        CLOUD_CODE_URL_ENV: f"http://127.0.0.1:{SHARED_PORT_MARKER}",
        USER_INFO_URL_ENV: f"http://127.0.0.1:{SHARED_PORT_MARKER}/userinfo",
    }


def test_the_placeholder_login_is_bound_over_the_clients_token_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("aisan.policy_tree.sys.platform", "linux")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    backend = _backend(tmp_path)
    backend.prepare(runtime)

    [bind] = backend.box_binds(runtime)
    assert bind == BindOver(
        runtime / "antigravity-oauth-token",
        tmp_path / "state" / "antigravity-oauth-token",
    )
    assert bind.src.read_text() == box_token_file(PLACEHOLDER_KEY)
    assert bind.src.stat().st_mode & 0o777 == 0o600


def test_the_port_is_distinct_from_every_other_backend():
    from aisan.egress.anthropic import PORT as ANTHROPIC_PORT
    from aisan.egress.openai_compat import PORT as COMPAT_PORT
    from aisan.egress.openai_responses import PORT as RESPONSES_PORT
    from aisan.egress.reapi import PORT as REAPI_PORT
    from aisan.egress.vertex import PORT as VERTEX_PORT

    others = {ANTHROPIC_PORT, COMPAT_PORT, RESPONSES_PORT, REAPI_PORT, VERTEX_PORT}
    assert PORT not in others


async def test_the_isolated_backend_swaps_the_placeholder_and_refuses_search(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append((request.path, request.headers["Authorization"]))
        return web.json_response({})

    url, upstream_runner = await _upstream_server(upstream)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    backend = _backend(tmp_path, url)
    base = "http://antigravity.invalid/v1internal"
    try:
        async with (
            backend.serve(runtime),
            ClientSession(
                connector=UnixConnector(path=str(backend.socket_path(runtime)))
            ) as session,
        ):
            async with session.post(
                f"{base}:loadCodeAssist", data=b"{}", headers=_bearer(PLACEHOLDER_KEY)
            ) as response:
                assert response.status == 200
            async with session.post(
                f"{base}:loadCodeAssist", data=b"{}", headers=_bearer(HOST_TOKEN)
            ) as response:
                assert response.status == 401
            async with session.post(
                f"{base}:generateContent", json=SEARCH, headers=_bearer(PLACEHOLDER_KEY)
            ) as response:
                assert response.status == 403
    finally:
        await upstream_runner.cleanup()
    assert reached == [("/v1internal:loadCodeAssist", f"Bearer {HOST_TOKEN}")]
    assert not backend.socket_path(runtime).exists()


async def test_the_shared_backend_requires_its_activation_token(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append((request.path, request.headers["Authorization"]))
        return web.json_response({})

    url, upstream_runner = await _upstream_server(upstream)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    backend = _backend(tmp_path, url)
    backend.prepare(runtime)
    try:
        async with backend.serve_shared(runtime) as activation:
            base = f"http://127.0.0.1:{activation.port}"
            assert activation.client_env == {
                CLOUD_CODE_URL_ENV: base,
                USER_INFO_URL_ENV: f"{base}/userinfo",
            }
            boxed = json.loads((runtime / "antigravity-oauth-token").read_text())
            client_token = boxed["token"]["access_token"]
            assert client_token != PLACEHOLDER_KEY
            assert boxed == json.loads(box_token_file(client_token))

            async with ClientSession() as session:
                async with session.get(
                    f"{base}/userinfo", headers=_bearer(PLACEHOLDER_KEY)
                ) as response:
                    assert response.status == 401
                async with session.post(
                    f"{base}/v1internal:generateContent",
                    json=SEARCH,
                    headers=_bearer(PLACEHOLDER_KEY),
                ) as response:
                    assert response.status == 401
                async with session.post(
                    f"{base}/v1internal:generateContent",
                    json=SEARCH,
                    headers=_bearer(client_token),
                ) as response:
                    assert response.status == 200

        async with backend.serve_shared(runtime, isolated=True) as isolated_act:
            iso_base = f"http://127.0.0.1:{isolated_act.port}"
            iso_token = json.loads((runtime / "antigravity-oauth-token").read_text())[
                "token"
            ]["access_token"]
            async with (
                ClientSession() as session,
                session.post(
                    f"{iso_base}/v1internal:generateContent",
                    json=SEARCH,
                    headers=_bearer(iso_token),
                ) as response,
            ):
                assert response.status == 403
    finally:
        await upstream_runner.cleanup()
    assert reached == [("/v1internal:generateContent", f"Bearer {HOST_TOKEN}")]


async def test_preflight_reports_an_unusable_credential(tmp_path):
    async def hello(request: web.Request) -> web.Response:
        return web.Response(status=404)

    url, upstream_runner = await _upstream_server(hello)
    failure = PreflightError("antigravity", "no login in the keyring", "agy")
    try:
        await _backend(tmp_path, url).preflight()
        with pytest.raises(PreflightError, match="no login in the keyring"):
            await _backend(tmp_path, url, credential=_Credential(failure)).preflight()
    finally:
        await upstream_runner.cleanup()


async def test_preflight_reports_an_unreachable_upstream_before_the_credential(
    tmp_path,
):
    failure = PreflightError("antigravity", "no login in the keyring", "agy")
    backend = _backend(tmp_path, credential=_Credential(failure))
    with pytest.raises(PreflightError, match="does not answer"):
        await backend.preflight()


@needs_bwrap
async def test_a_real_box_reads_the_placeholder_and_reaches_only_the_proxy(tmp_path):
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append((request.path, request.headers["Authorization"]))
        return web.json_response({"currentTier": {"id": "free-tier"}})

    url, upstream_runner = await _upstream_server(upstream)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    token_file = state / "antigravity-oauth-token"
    spec = BoxSpec(
        root=worktree,
        binds=(Bind(state, RW),),
        tmpfs=(),
        env=(),
        egress=(_backend(tmp_path, url, token_file=token_file),),
        unshare_net=True,
        limits=Limits(use_cgroup=False),
    )
    script = (
        "import json, os, urllib.request\n"
        f"login = json.load(open({str(token_file)!r}))\n"
        "bearer = login['token']['access_token']\n"
        "print('bearer=' + bearer)\n"
        "def call(url, data=None):\n"
        "    request = urllib.request.Request(\n"
        "        url, data=data, headers={'Authorization': 'Bearer ' + bearer})\n"
        "    return urllib.request.urlopen(request, timeout=20).read().decode()\n"
        f"print('userinfo=' + call(os.environ[{USER_INFO_URL_ENV!r}]))\n"
        f"base = os.environ[{CLOUD_CODE_URL_ENV!r}]\n"
        "print('tier=' + call(base + '/v1internal:loadCodeAssist', b'{}'))\n"
        "try:\n"
        f"    urllib.request.urlopen({url!r}, timeout=5)\n"
        "    print('direct=reached')\n"
        "except OSError:\n"
        "    print('direct=unreachable')\n"
    )
    box = Box(spec, box_id="antigravity-test")
    try:
        async with box:
            result = await asyncio.to_thread(
                subprocess.run,
                box.command([sys.executable, "-c", script]),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env={**os.environ, **box.env},
            )
    finally:
        await upstream_runner.cleanup()

    assert result.returncode == 0, result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert lines["bearer"] == PLACEHOLDER_KEY
    assert json.loads(lines["userinfo"]) == json.loads(USERINFO.body)
    assert json.loads(lines["tier"]) == {"currentTier": {"id": "free-tier"}}
    assert lines["direct"] == "unreachable"
    assert reached == [("/v1internal:loadCodeAssist", f"Bearer {HOST_TOKEN}")]
    assert token_file.read_bytes() == b""


def test_the_preset_is_registered_and_uses_isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("BROWSER", "firefox")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    config = tmp_path / "git-config"
    config.mkdir()
    backend = _backend(tmp_path, token_file=state / TOKEN_FILE)
    spec = antigravity(worktree, state=state, egress=(backend,), extra_ro=(config,))

    assert PRESETS["antigravity"] is antigravity_default
    assert [(bind.path, bind.mode) for bind in spec.binds] == [
        (config, RO),
        (state, RW),
    ]
    assert not {
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "BROWSER",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
    } & set(dict(spec.env))
    assert agy_argv(("-p", "hello"), state=state) == [
        "agy",
        "--gemini_dir",
        str(state),
        "--dangerously-skip-permissions",
        "-p",
        "hello",
    ]
    [bind] = backend.box_binds(tmp_path)
    assert bind.dst == state / "antigravity-cli" / "antigravity-oauth-token"


def test_state_is_seeded_past_first_run_with_the_repository_trusted(tmp_path):
    from aisan.cli.antigravity import ONBOARDING, host_settings, seed_state

    state = tmp_path / "state"
    state.mkdir()
    repo = tmp_path / "repo"
    host = tmp_path / "host-settings.json"
    host.write_text(json.dumps({"model": "Host Model", "trustedWorkspaces": ["/h"]}))

    seed_state(state, repo, host_settings(host))
    app = state / "antigravity-cli"
    assert json.loads((app / "cache" / "onboarding.json").read_text()) == ONBOARDING
    assert ONBOARDING == {
        "consumerOnboardingComplete": True,
        "enterpriseOnboardingComplete": False,
        "onboardingComplete": True,
    }
    assert json.loads((app / "settings.json").read_text()) == {
        "model": "Host Model",
        "trustedWorkspaces": [str(repo)],
    }

    # The box's own choices stay, the host's model wins, trust is not repeated.
    (app / "settings.json").write_text(
        json.dumps(
            {
                "model": "Box Model",
                "colorScheme": "dark",
                "trustedWorkspaces": ["/elsewhere", str(repo)],
            }
        )
    )
    seed_state(state, repo, host_settings(host))
    assert json.loads((app / "settings.json").read_text()) == {
        "model": "Host Model",
        "colorScheme": "dark",
        "trustedWorkspaces": ["/elsewhere", str(repo)],
    }


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"model": 3}'])
def test_host_settings_copies_nothing_from_a_mangled_or_absent_file(tmp_path, content):
    from aisan.cli.antigravity import host_settings

    source = tmp_path / "settings.json"
    if content is not None:
        source.write_text(content)
    assert host_settings(source) == {}


def test_host_rules_files_are_mirrored_under_their_own_names(tmp_path):
    from aisan.cli.antigravity import USER_MEMORY, seed_user_memory

    assert [path.name for path in USER_MEMORY] == ["GEMINI.md", "AGENTS.md"]
    assert {path.parent for path in USER_MEMORY} == {Path.home() / ".gemini"}

    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "GEMINI.md"
    source.write_text("host rules\n")

    seed_user_memory(state, source)
    assert (state / "GEMINI.md").read_text() == "host rules\n"

    seed_user_memory(state, tmp_path / "AGENTS.md")
    assert not (state / "AGENTS.md").exists()


def test_seeding_writes_nothing_through_links_planted_in_the_state(tmp_path):
    from aisan.cli.antigravity import seed_state

    victim = tmp_path / "victim"
    (victim / "cache").mkdir(parents=True)
    target = tmp_path / "target.json"
    target.write_text("host file")
    state = tmp_path / "state"
    state.mkdir()

    (state / "antigravity-cli").symlink_to(victim)
    seed_state(state, tmp_path / "repo", {})
    assert list(victim.iterdir()) == [victim / "cache"]
    assert list((victim / "cache").iterdir()) == []

    app = state / "antigravity-cli"
    (app / "settings.json").unlink()
    (app / "settings.json").symlink_to(target)
    seed_state(state, tmp_path / "repo", {})
    assert target.read_text() == "host file"
    assert not (app / "settings.json").is_symlink()


# The proxy policy, the token file, ``CLOUD_CODE_URL`` and ``--gemini_dir``
# were measured against this version. On another, re-measure before trusting
# a pass.
AGY_VERSION = "1.5.11"


def _agy_version() -> str:
    done = subprocess.run(
        ["agy", "--version"], capture_output=True, text=True, timeout=30, check=True
    )
    return done.stdout.strip()


async def _run_agy(tmp_path: Path, backend_for, *payload: str):
    """Run real agy in a real box; ``backend_for`` builds the backend."""
    from aisan.cli.antigravity import seed_state

    assert _agy_version() == AGY_VERSION
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    seed_state(state, worktree, {})
    backend = backend_for(state / TOKEN_FILE)
    base = antigravity(worktree, state=state, egress=(backend,))
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    async with Box(spec, box_id="antigravity-live") as box:
        result = await asyncio.to_thread(
            subprocess.run,
            box.command(agy_argv(payload, state=state)),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env={**os.environ, **box.env},
        )
    assert (state / TOKEN_FILE).read_bytes() == b""
    return result


@pytest.mark.live
@pytest.mark.skipif(agy_binary() is None, reason="agy is not installed")
async def test_real_agy_presents_only_the_placeholder_to_a_stub(tmp_path):
    """The tripwire: a client that rejected the placeholder login would start
    an OAuth flow and send the proxy nothing."""
    seen = []

    async def upstream(request: web.Request) -> web.Response:
        # Preflight probes the root without a credential.
        if request.path != "/":
            seen.append(
                (request.method, request.path, request.headers.get("Authorization"))
            )
        return web.json_response(
            {"error": {"code": 501, "message": "stub", "status": "UNIMPLEMENTED"}},
            status=501,
        )

    url, upstream_runner = await _upstream_server(upstream)
    try:
        result = await _run_agy(
            tmp_path,
            lambda token_file: _backend(tmp_path, url, token_file=token_file),
            "--print-timeout",
            "30s",
            "-p",
            "say hi",
        )
    finally:
        await upstream_runner.cleanup()

    assert result.returncode != 0
    assert ("POST", "/v1internal:loadCodeAssist", f"Bearer {HOST_TOKEN}") in seen
    # The proxy forwards only what arrived with the placeholder, and attaches
    # the host token itself.
    assert {authorization for _, _, authorization in seen} == {f"Bearer {HOST_TOKEN}"}


@pytest.mark.live
@pytest.mark.skipif(agy_binary() is None, reason="agy is not installed")
async def test_real_agy_completes_a_turn_on_the_host_login(tmp_path):
    from aisan.egress.antigravity_keyring import KeyringCredential, KeyringError

    credential = KeyringCredential()
    try:
        await credential.token()
    except KeyringError as e:
        pytest.skip(f"no usable host agy login: {e}")

    result = await _run_agy(
        tmp_path,
        lambda token_file: AntigravityBackend(
            credential=credential, token_file=token_file
        ),
        "--model",
        "claude-opus-5-5-low",
        "-p",
        "Reply with exactly: hi",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hi"


@pytest.mark.live
@pytest.mark.skipif(agy_binary() is None, reason="agy is not installed")
async def test_host_agy_runs_against_the_refresh_sink_and_keeps_its_login():
    from aisan.egress.antigravity_keyring import (
        KeyringError,
        _refresh_agy_login,
        parse_login,
        read_keyring,
    )

    assert _agy_version() == AGY_VERSION
    try:
        before = parse_login(read_keyring())
    except KeyringError as e:
        pytest.skip(f"no usable host agy login: {e}")

    await _refresh_agy_login(("agy",))

    after = parse_login(read_keyring())
    assert after.remaining() > 0
    # agy rewrites the item only near or past the expiry.
    if before.remaining() > 120:
        assert after == before
