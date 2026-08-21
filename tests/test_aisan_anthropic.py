# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Anthropic backend: what it reads, what it refuses, and what it never does.

`test_proxy_anthropic.py` covers the transport (the two allowlists, the injected
bearer). This covers the backend -- the credential file, the preflight decisions,
and the one thing the whole design rests on: **aisan reads this file and never
writes it.**

That last claim needs a test rather than a docstring, because the failure it
guards against is silent and expensive. The host's own Claude Code owns
`~/.credentials.json`, rewrites it on refresh, and holds no lock; an aisan that
refreshed would race it, and losing that race logs the user out of the tool aisan
exists to sandbox. So `test_the_backend_never_writes_the_credential` runs a real
box lifecycle against a real file and asserts the bytes and the mtime are
untouched.

Every fixture credential here is fake by construction -- a literal token string
with no upstream that would accept it. Nothing in this file reads the host's own.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from aiohttp import ClientSession, UnixConnector, web

from aisan.egress.anthropic import PLACEHOLDER_KEY, PORT, AnthropicBackend
from aisan.egress.base import PreflightError

# Not a credential: a fixed string, in a file this test wrote, against an
# upstream that is a local aiohttp handler. Named so a reader does not have to
# work that out.
FAKE_TOKEN = "fake-access-token-for-tests"


def _credentials(path: Path, *, token: str = FAKE_TOKEN, ttl_s: float = 3600) -> Path:
    """A credential file in the real shape, with an expiry `ttl_s` from now.

    Epoch MILLISECONDS, which is what Claude Code writes -- a test that used
    seconds would pass against a backend that also used seconds and fail against
    the real file, which is the wrong way round.
    """
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": token,
                    "refreshToken": "fake-refresh-token",
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
    """Both variables matter, for different reasons.

    The base URL is the route. The KEY is what stops the client looking for a
    credential the box does not have: with none set, Claude Code reads
    `~/.claude/.credentials.json`, which is absent in the box, and tries to log
    in interactively rather than failing. Measured against 2.1.219.
    """
    env = AnthropicBackend(credentials=tmp_path / "c.json").client_env()
    assert env == {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        "ANTHROPIC_API_KEY": PLACEHOLDER_KEY,
    }
    # A placeholder that looked like a key would be one somebody eventually
    # believed. It says what it is.
    assert "placeholder" in PLACEHOLDER_KEY


def test_the_port_is_distinct_from_the_other_backends():
    """`BoxSpec.__post_init__` asserts injectivity across a spec's backends, but
    only for the backends that spec actually has. A collision between two
    backends nobody composed yet is found here or when a box mysteriously
    reaches the wrong upstream."""
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
    """Not logged in HERE, so the fix is a login -- and the message has to
    carry it. A PreflightError whose fix is wrong is worse than none: it sends
    an operator to run something that cannot help."""
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


async def test_an_expired_token_says_expired_rather_than_missing(tmp_path):
    """Same command, different reason -- and the reason is the point.

    An operator who IS logged in, reading "no credential", goes looking for a
    file that exists. So the expiry case names the expiry, and says that aisan
    does not refresh, which is why the host has to.
    """
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await AnthropicBackend(
                credentials=_credentials(tmp_path / "c.json", ttl_s=-60), upstream=up
            ).preflight()
    finally:
        await runner.cleanup()
    assert "expires in" in e.value.reason
    assert "never refreshes" in e.value.reason
    assert "claude auth login" in e.value.fix


async def test_a_token_expiring_inside_the_margin_is_refused_too(tmp_path):
    """Still valid, and refused anyway. The margin is the difference between
    failing at preflight -- where there is a terminal and a fix command -- and
    failing three turns in with an opaque 401."""
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError):
            await AnthropicBackend(
                credentials=_credentials(tmp_path / "c.json", ttl_s=60), upstream=up
            ).preflight()
    finally:
        await runner.cleanup()


async def test_a_dead_upstream_says_start_it_not_log_in(tmp_path):
    """The user's local proxy is not running. There is nothing to log into, and
    telling them to log in would be the wrong fix confidently stated."""
    up, runner = await _upstream_server(_hello)
    await runner.cleanup()  # dead address, deliberately
    with pytest.raises(PreflightError) as e:
        await AnthropicBackend(
            credentials=_credentials(tmp_path / "c.json"), upstream=up
        ).preflight()
    assert "does not answer" in e.value.reason
    assert "start the service" in e.value.fix
    assert "login" not in e.value.fix


async def test_preflight_does_not_spend_a_model_call(tmp_path):
    """A connect, not a turn. Asking the upstream something real would spend
    quota to learn that the hop exists -- and on a metered subscription that is
    a cost per box start."""
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
    """The message must never quote the document. Everything under
    `claudeAiOauth` is a secret or a timestamp, so a parse error that echoed
    what it failed to parse would put the token wherever that message is
    logged."""
    bad = tmp_path / "c.json"
    bad.write_text(
        '{"claudeAiOauth": {"accessToken": "' + FAKE_TOKEN + '"'
    )  # truncated
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await AnthropicBackend(credentials=bad, upstream=up).preflight()
    finally:
        await runner.cleanup()
    assert FAKE_TOKEN not in str(e.value)
    assert str(bad) in str(e.value)


async def test_the_backend_serves_and_injects_the_credential_it_read(tmp_path):
    """The two halves joined: `serve` puts the transport on the socket the Box
    binds, and the bearer that arrives upstream is the one in the file."""
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
        # Torn down with the block: a proxy surviving its box is an
        # unauthenticated capability to the credential behind it.
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
            client_token = activation.client_env["ANTHROPIC_API_KEY"]
            assert activation.port == int(endpoint.rsplit(":", 1)[1])
            assert client_token.endswith(PLACEHOLDER_KEY[-20:])
            assert client_token != PLACEHOLDER_KEY
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
                first.client_env["ANTHROPIC_API_KEY"]
                != second.client_env["ANTHROPIC_API_KEY"]
            )
    finally:
        await up_runner.cleanup()


async def test_the_backend_never_writes_the_credential(tmp_path):
    """The settled decision, asserted rather than intended.

    A writeback would race the host's own Claude Code over a 0600 file neither
    side locks, and losing that race logs the user out. Checked over a full
    serve-and-request cycle -- the moment a refresh would plausibly be attempted
    -- by bytes and mtime, because a rewrite with identical content is still a
    rewrite and still a race.
    """
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
    # Still only readable by its owner: nothing here relaxed the mode either.
    assert creds.stat().st_mode & 0o777 == 0o600


async def test_a_stale_socket_from_a_killed_box_does_not_block_the_next_one(tmp_path):
    """bind() fails EADDRINUSE forever on a leftover socket file, so a box
    killed with SIGKILL would poison its own runtime dir. The path is derived
    from this box, so nothing else owns it and unlinking is safe."""
    up, up_runner = await _upstream_server(_hello)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = AnthropicBackend(
        credentials=_credentials(tmp_path / "c.json"), upstream=up
    )
    backend.socket_path(runtime).write_bytes(b"")  # stale socket of a killed box
    try:
        async with backend.serve(runtime):
            assert backend.socket_path(runtime).is_socket()
    finally:
        await up_runner.cleanup()
