# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan import Box
from aisan.egress.base import PLACEHOLDER_KEY, PreflightError
from aisan.egress.openai_compat import (
    DEFAULT_CREDENTIALS,
    PORT,
    OpenAICompatBackend,
)
from aisan.presets import PRESETS
from aisan.presets.opencode import opencode, opencode_binary, opencode_default
from aisan.sandbox import RW

FAKE_KEY = "fake-api-key-for-tests"

PROVIDER = "zai-coding-plan"


def _auth(path: Path, *, provider: str = PROVIDER, key: str = FAKE_KEY) -> Path:
    path.write_text(json.dumps({provider: {"type": "api", "key": key}}))
    path.chmod(0o600)
    return path


def _oauth_auth(path: Path, *, provider: str = PROVIDER) -> Path:
    path.write_text(
        json.dumps({provider: {"type": "oauth", "access": "a", "refresh": "r"}})
    )
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


def _backend(tmp_path: Path, **kw) -> OpenAICompatBackend:
    if "credentials" not in kw:
        kw["credentials"] = _auth(tmp_path / "auth.json")
    return OpenAICompatBackend(provider=PROVIDER, **kw)


def test_the_box_is_told_a_placeholder_a_route_and_a_model():
    b = OpenAICompatBackend(
        provider=PROVIDER,
        model="glm-5.3",
        upstream="https://x/v4",
        credentials=Path("/tmp-does-not-exist/auth.json"),
    )
    env = b.client_env()
    auth = json.loads(env["OPENCODE_AUTH_CONTENT"])
    assert auth == {PROVIDER: {"type": "api", "key": PLACEHOLDER_KEY}}
    cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert cfg["provider"][PROVIDER]["options"]["baseURL"] == f"http://127.0.0.1:{PORT}"
    assert cfg["model"] == f"{PROVIDER}/glm-5.3"

    assert cfg["share"] == "disabled"
    assert "placeholder" in PLACEHOLDER_KEY


def test_a_backend_without_a_model_picks_none(tmp_path):
    b = _backend(tmp_path, upstream="https://x")
    assert "model" not in json.loads(b.client_env()["OPENCODE_CONFIG_CONTENT"])


def test_client_env_reads_no_files(tmp_path):
    b = OpenAICompatBackend(
        provider=PROVIDER,
        model="m",
        upstream="https://x",
        credentials=tmp_path / "absent.json",
        catalog=tmp_path / "absent.json",
    )
    assert "OPENCODE_AUTH_CONTENT" in b.client_env()


def test_the_port_is_distinct_from_the_other_backends():
    from aisan.egress.anthropic import PORT as ANTHROPIC_PORT
    from aisan.egress.reapi import PORT as REAPI_PORT
    from aisan.egress.vertex import PORT as VERTEX_PORT

    assert len({PORT, ANTHROPIC_PORT, REAPI_PORT, VERTEX_PORT}) == 4


def test_two_providers_of_the_family_need_two_ports(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    a = OpenAICompatBackend(
        provider="a",
        upstream="https://x",
        credentials=_auth(tmp_path / "a.json", provider="a"),
    )
    b = OpenAICompatBackend(
        provider="b",
        upstream="https://y",
        credentials=_auth(tmp_path / "b.json", provider="b"),
        port=PORT + 1,
    )
    spec = opencode(wt, state=state, egress=(a, b))
    assert [e.name for e in spec.egress] == ["openai-a", "openai-b"]
    with pytest.raises(ValueError):
        opencode(
            wt,
            state=state,
            egress=(a, OpenAICompatBackend(provider="c", upstream="https://z")),
        )


async def test_preflight_passes_when_the_route_the_key_and_upstream_are_there(
    tmp_path,
):
    up, runner = await _upstream_server(_hello)
    try:
        await _backend(tmp_path, upstream=up).preflight()
    finally:
        await runner.cleanup()


async def test_the_route_resolves_from_the_catalog(tmp_path):
    up, runner = await _upstream_server(_hello)
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps({PROVIDER: {"api": f"{up}/v4", "models": {}}}))
    try:
        backend = _backend(tmp_path, catalog=catalog)
        await backend.preflight()
        assert backend._resolve_api() == f"{up}/v4"
    finally:
        await runner.cleanup()


async def test_an_unresolvable_route_names_the_catalog_fix(tmp_path):
    with pytest.raises(PreflightError) as e:
        await _backend(
            tmp_path, upstream=None, catalog=tmp_path / "absent.json"
        ).preflight()
    assert "cannot resolve the route" in e.value.reason
    assert "opencode once on the HOST" in e.value.fix
    assert "login" not in e.value.fix


async def test_an_explicit_upstream_needs_no_catalog(tmp_path):
    up, runner = await _upstream_server(_hello)
    try:
        await _backend(
            tmp_path, upstream=up, catalog=tmp_path / "absent.json"
        ).preflight()
    finally:
        await runner.cleanup()


async def test_a_missing_credential_refuses_with_the_login_command(tmp_path):
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await _backend(
                tmp_path, upstream=up, credentials=tmp_path / "absent.json"
            ).preflight()
    finally:
        await runner.cleanup()
    assert "opencode providers login" in e.value.fix
    assert "absent.json" in e.value.reason


async def test_a_subscription_credential_is_refused_as_a_shape_not_a_login(
    tmp_path,
):
    up, runner = await _upstream_server(_hello)
    creds = _oauth_auth(tmp_path / "auth.json")
    try:
        with pytest.raises(PreflightError) as e:
            await _backend(tmp_path, upstream=up, credentials=creds).preflight()
    finally:
        await runner.cleanup()
    assert "'oauth'" in e.value.reason
    assert "login" not in e.value.fix
    assert "API key" in e.value.fix


async def test_a_dead_upstream_says_check_the_network_not_log_in(tmp_path):
    up, runner = await _upstream_server(_hello)
    await runner.cleanup()
    with pytest.raises(PreflightError) as e:
        await _backend(tmp_path, upstream=up).preflight()
    assert "does not answer" in e.value.reason
    assert "check the network" in e.value.fix
    assert "login" not in e.value.fix


async def test_preflight_does_not_spend_a_model_call(tmp_path):
    seen: list[tuple[str, str]] = []

    async def upstream(request: web.Request) -> web.Response:
        seen.append((request.method, request.path))
        return web.json_response({"ok": True})

    up, runner = await _upstream_server(upstream)
    try:
        await _backend(tmp_path, upstream=up).preflight()
    finally:
        await runner.cleanup()
    assert all(path != "/chat/completions" for _, path in seen), seen


async def test_a_credential_file_that_is_not_json_names_the_path_only(tmp_path):
    bad = tmp_path / "auth.json"
    bad.write_text('{"p": {"type": "api", "key": "' + FAKE_KEY + '"')
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await _backend(tmp_path, upstream=up, credentials=bad).preflight()
    finally:
        await runner.cleanup()
    assert FAKE_KEY not in str(e.value)
    assert str(bad) in str(e.value)


async def test_the_backend_serves_and_injects_the_credential_it_read(tmp_path):
    got: dict[str, str] = {}
    paths: list[str] = []

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        paths.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = _backend(tmp_path, upstream=f"{up}/v4")
    try:
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            assert sock.exists()
            async with (
                ClientSession(connector=UnixConnector(path=str(sock))) as s,
                s.post(
                    "http://openai.invalid/chat/completions",
                    data=b"{}",
                    headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"},
                ) as r,
            ):
                assert r.status == 200

        assert not backend.socket_path(runtime).exists()
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == f"Bearer {FAKE_KEY}"
    assert paths == ["/v4/chat/completions"]


async def test_shared_backend_serves_the_activated_tcp_endpoint(tmp_path):
    got: dict[str, str] = {}

    async def upstream(request: web.Request) -> web.Response:
        got.update({k.lower(): v for k, v in request.headers.items()})
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    backend = _backend(tmp_path, upstream=f"{up}/v4")
    try:
        async with backend.serve_shared(tmp_path) as activation:
            auth = json.loads(activation.client_env["OPENCODE_AUTH_CONTENT"])
            config = json.loads(activation.client_env["OPENCODE_CONFIG_CONTENT"])
            client_token = auth[PROVIDER]["key"]
            endpoint = config["provider"][PROVIDER]["options"]["baseURL"]
            async with (
                ClientSession() as session,
                session.post(
                    f"{endpoint}/chat/completions",
                    data=b"{}",
                    headers={"Authorization": f"Bearer {client_token}"},
                ) as response,
            ):
                assert response.status == 200
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == f"Bearer {FAKE_KEY}"


async def test_the_backend_never_writes_the_credential(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    creds = _auth(tmp_path / "auth.json")
    before = creds.read_bytes()
    before_mtime = creds.stat().st_mtime_ns
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = _backend(tmp_path, upstream=up, credentials=creds)
    try:
        await backend.preflight()
        async with (
            backend.serve(runtime),
            ClientSession(
                connector=UnixConnector(path=str(backend.socket_path(runtime)))
            ) as s,
            s.post("http://openai.invalid/chat/completions", data=b"{}") as r,
        ):
            assert r.status == 200
    finally:
        await up_runner.cleanup()

    assert creds.read_bytes() == before
    assert creds.stat().st_mtime_ns == before_mtime

    assert creds.stat().st_mode & 0o777 == 0o600


async def test_a_stale_socket_from_a_killed_box_does_not_block_the_next_one(
    tmp_path,
):
    up, up_runner = await _upstream_server(_hello)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = _backend(tmp_path, upstream=up)
    backend.socket_path(runtime).write_bytes(b"")
    try:
        async with backend.serve(runtime):
            assert backend.socket_path(runtime).is_socket()
    finally:
        await up_runner.cleanup()


def _spec(tmp_path, **kw):
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    catalog = tmp_path / "models.json"
    catalog.write_text("{}")
    spec = opencode(wt, state=state, models=catalog, **kw)
    return dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )


def test_the_preset_is_registered_under_its_own_name():
    assert PRESETS["opencode"] is opencode_default


def test_no_bind_source_contains_the_credential(tmp_path):
    for m in Box(_spec(tmp_path), box_id="t").mounts():
        if m.src is None:
            continue
        assert not DEFAULT_CREDENTIALS.is_relative_to(m.src), f"{m.src} would expose it"


def test_the_state_dir_is_rw_and_the_catalog_substitutes_at_the_box_read_path(tmp_path):
    from aisan.sandbox import BindOver

    spec = _spec(tmp_path)
    modes = {
        Path(b.path): (b.mode, b.optional) for b in spec.binds if hasattr(b, "mode")
    }
    assert modes[tmp_path / "state"] == (RW, False)
    box_dst = Path.home() / ".cache" / "opencode" / "models.json"
    overs = [b for b in spec.binds if isinstance(b, BindOver)]
    assert any(b.src == tmp_path / "models.json" and b.dst == box_dst for b in overs)


def test_an_absent_catalog_binds_nothing(tmp_path):

    from aisan.sandbox import BindOver

    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    spec = opencode(wt, state=state, models=tmp_path / "does-not-exist.json")
    assert not any(
        isinstance(b, BindOver) and b.dst.name == "models.json" for b in spec.binds
    )


def test_a_state_dir_inside_the_root_needs_no_bind(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = opencode(wt, state=wt / ".aisan-opencode-state", models=tmp_path / "m.json")
    assert all(Path(b.path) != wt / ".aisan-opencode-state" for b in spec.binds)


def test_the_environment_names_the_state_dir_and_disables_the_fetchers(tmp_path):
    spec = _spec(tmp_path)
    env = dict(spec.env)
    assert env["XDG_DATA_HOME"] == str(tmp_path / "state")
    assert env["HOME"] == str(Path.home())
    for flag in (
        "OPENCODE_DISABLE_AUTOUPDATE",
        "OPENCODE_DISABLE_MODELS_FETCH",
        "OPENCODE_DISABLE_LSP_DOWNLOAD",
    ):
        assert env[flag] == "1"


def test_xdg_config_home_is_left_unset(tmp_path):
    assert "XDG_CONFIG_HOME" not in dict(_spec(tmp_path).env)


def test_the_state_dir_has_no_default(tmp_path):
    with pytest.raises(TypeError):
        opencode(tmp_path / "wt")  # type: ignore[call-arg]


def test_egress_requires_the_network_namespace(tmp_path):
    assert _spec(tmp_path).unshare_net is True


def test_the_default_entry_builds_without_a_deployment(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = opencode_default(wt)
    assert spec.root == wt
    assert spec.egress == ()


def test_the_binary_is_found_on_the_usr_surface_or_reported_absent():
    found = opencode_binary()
    if found is None:
        pytest.skip("opencode is not installed on this host")
    assert found.exists()


def test_the_real_backend_and_the_preset_agree_on_the_port(tmp_path):
    spec = _spec(tmp_path, egress=(_backend(tmp_path, upstream="https://x"),))
    cfg = json.loads(spec.egress[0].client_env()["OPENCODE_CONFIG_CONTENT"])
    assert cfg["provider"][PROVIDER]["options"]["baseURL"].endswith(f":{PORT}")


async def _run_in_box(spec, script: str) -> subprocess.CompletedProcess:
    box = Box(spec, box_id="opencode-test")
    async with box:
        argv = box.command([sys.executable, "-c", script])
        env = {**os.environ, **box.env}
        return await asyncio.to_thread(
            subprocess.run,
            argv,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )


@pytest.mark.skipif(
    not DEFAULT_CREDENTIALS.exists(), reason="no host credential, so nothing to hide"
)
async def test_the_credential_is_not_in_the_box(tmp_path):
    up, up_runner = await _upstream_server(_hello)
    try:
        spec = _spec(tmp_path, egress=(_backend(tmp_path, upstream=up),))
        script = (
            "import os\n"
            f"p = {str(DEFAULT_CREDENTIALS)!r}\n"
            "print('exists=%s' % os.path.exists(p))\n"
            "print('share_exists=%s' % os.path.exists(os.path.dirname(p)))\n"
        )
        r = await _run_in_box(spec, script)
    finally:
        await up_runner.cleanup()
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    assert "exists=False" in r.stdout

    assert "share_exists=False" in r.stdout, r.stdout


async def test_the_box_reaches_the_model_only_through_the_relay(tmp_path):

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host_port = srv.getsockname()[1]
    spec = _spec(tmp_path, egress=(_backend(tmp_path, upstream=up),))
    script = (
        "import http.client, json, os, socket\n"
        f"c = http.client.HTTPConnection('127.0.0.1', {PORT}, timeout=10)\n"
        "c.request('POST', '/chat/completions', body='{}', headers={"
        "'Authorization': 'Bearer ' + json.loads("
        "os.environ['OPENCODE_AUTH_CONTENT'])['" + PROVIDER + "']['key']})\n"
        "print('body=' + c.getresponse().read().decode())\n"
        "n = socket.socket(); n.settimeout(3)\n"
        f"print('host=' + str(n.connect_ex(('127.0.0.1', {host_port}))))\n"
    )
    try:
        r = await _run_in_box(spec, script)
    finally:
        srv.close()
        await up_runner.cleanup()
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    assert "body=" in r.stdout and '"ok"' in r.stdout.replace(" ", "")

    assert "host=111" in r.stdout


@pytest.mark.live
@pytest.mark.skipif(opencode_binary() is None, reason="opencode is not installed")
async def test_a_real_opencode_turn_through_the_whole_chain(tmp_path):
    marker = "aisan-e2e-refusal-marker"

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"message": marker, "type": "invalid_request_error"}},
            status=400,
        )

    up, up_runner = await _upstream_server(upstream)
    state = tmp_path / "state"
    state.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    backend = OpenAICompatBackend(
        provider=PROVIDER,
        model="glm-5.2",
        upstream=up,
        credentials=_auth(tmp_path / "auth.json"),
    )

    absent_catalog = tmp_path / "no-host-catalog.json"
    spec = dataclasses.replace(
        opencode(wt, state=state, egress=(backend,), models=absent_catalog),
        limits=dataclasses.replace(
            opencode(wt, state=state, egress=(backend,)).limits, use_cgroup=False
        ),
    )
    box = Box(spec, box_id="opencode-e2e")
    async with box:
        argv = box.command(["opencode", "run", "say hi"])
        r = await asyncio.to_thread(
            subprocess.run,
            argv,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, **box.env},
        )
    await up_runner.cleanup()
    assert r.returncode != 0
    assert marker in r.stdout + r.stderr, f"OUT={r.stdout!r} ERR={r.stderr!r}"
