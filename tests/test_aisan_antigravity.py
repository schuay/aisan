# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web

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
from aisan.proxy.antigravity import USERINFO, Profile
from aisan.sandbox import RW, Bind, BindOver
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


def test_the_placeholder_login_is_bound_over_the_clients_token_file(tmp_path):
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
