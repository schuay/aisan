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

from aisan.egress.anthropic import (
    API_KEY_ENV,
    OAUTH_TOKEN_ENV,
    PLACEHOLDER_KEY,
    PORT,
    SUBSCRIPTION_ENV,
    AnthropicBackend,
)
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

    The base URL is the route. The TOKEN is what stops the client looking for a
    credential the box does not have: with none set, Claude Code reads
    `~/.claude/.credentials.json`, which is absent in the box, and tries to log
    in interactively rather than failing. Measured against 2.1.219 for the key
    dress and 2.1.246 for the subscription one.

    The variable NAME tracks the credential kind, and the credential file here
    does not exist -- which is the other half of the assertion: a description
    computed on a host with nothing logged in is still complete, because
    `explain` and the snapshots run there.
    """
    env = AnthropicBackend(credentials=tmp_path / "c.json").client_env()
    assert env == {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        OAUTH_TOKEN_ENV: PLACEHOLDER_KEY,
    }
    # A placeholder that looked like a credential would be one somebody
    # eventually believed. It says what it is.
    assert "placeholder" in PLACEHOLDER_KEY


def test_an_api_key_backend_dresses_the_box_as_an_api_key_client(tmp_path):
    """The kind picks BOTH halves, and the halves must not drift apart.

    Measured: the client's `anthropic-beta` set differs by dress, and the proxy
    forwards that header -- so a box dressed for one credential while the proxy
    attaches the other announces a protocol its credential does not match. This
    pins the box-facing half of each kind; `test_the_upstream_credential_matches
    _the_dress` pins the other.
    """
    env = AnthropicBackend(api_key="sk-ant-not-real").client_env()
    assert env == {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        API_KEY_ENV: PLACEHOLDER_KEY,
    }
    # Nothing on disk to protect: the key lives in memory, so there is no path
    # for the Box's credential-exposure refusal to be given.
    assert AnthropicBackend(api_key="sk-ant-not-real").credentials == ()


def test_a_backend_holds_exactly_one_credential_kind(tmp_path):
    """Both would mean an implicit precedence, which is how credentials get mixed up."""
    with pytest.raises(ValueError, match="at most one"):
        AnthropicBackend(credentials=tmp_path / "c.json", api_key="sk-ant-not-real")


def test_the_plan_label_is_read_best_effort_and_never_raises(tmp_path):
    """A display-only label must not be able to break a dry run.

    `client_env` is called by `explain` and by the snapshot tests on hosts with
    no credential at all. Absent, unreadable and malformed all have to omit the
    variable rather than raise -- and a present one has to appear, or the label
    silently stops tracking the account.
    """
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
            client_token = activation.client_env[OAUTH_TOKEN_ENV]
            assert activation.port == int(endpoint.rsplit(":", 1)[1])
            assert client_token.endswith(PLACEHOLDER_KEY[-20:])
            assert client_token != PLACEHOLDER_KEY
            # Presented the way this dress presents it. The other location is
            # covered by `test_either_dress_can_present_the_relay_token`.
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


async def test_the_upstream_credential_matches_the_dress(tmp_path):
    """The other half of the pairing: an API-key kind attaches `x-api-key`.

    The subscription kind's bearer is pinned by
    `test_the_backend_serves_and_injects_the_credential_it_read`. This one pins
    the second kind, and pins that the box-facing dress and the upstream header
    are chosen together rather than by two independent settings.

    The host half of this kind is otherwise unmeasured -- no real key was
    available to answer upstream -- so what is asserted here is the whole of
    what aisan controls: the header it attaches and the one it never forwards.
    """
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
    # The box's own token stopped at the proxy, and no bearer was invented.
    assert got["x-api-key"] != client_token
    assert "authorization" not in got


async def test_either_dress_can_present_the_relay_token(tmp_path):
    """The relay authenticates a VALUE, not a location.

    A box dressed for a subscription presents the token in `authorization`, one
    dressed for a key presents it in `x-api-key` (both measured). The proxy
    accepts either, so the dress can change without the check having to be told;
    what it must never accept is a request carrying neither, or the wrong value
    in either.
    """
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
    """Which policy a box gets is derived from its network mode, not passed.

    `serve` is what the Box calls under `unshare_net` and `serve_shared` is what
    it calls without it, so the transport already encodes the one fact the
    relaxation rests on: whether the box can reach the network itself. This pins
    both directions at the backend, because the policy objects agreeing in
    isolation would not catch `serve_shared` being wired to the strict one.
    """
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

        # Same backend, same body, isolated transport: refused.
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


async def test_a_token_that_expires_mid_session_is_refused_with_the_host_fix(tmp_path):
    """The failure this exists for: a session outliving the token it launched
    with, on a machine where the only Claude Code running is the one in the box,
    reading a placeholder. Nothing refreshes the file, so the bearer goes stale
    under a session preflight passed.

    Forwarding it got the agent Anthropic's own 401, which its client renders as
    "Please run /login" -- impossible in a box with no credential file and no
    route to the OAuth endpoints. The refusal has to name the HOST-side fix, and
    it has to happen here rather than upstream.
    """
    reached = []

    async def upstream(request: web.Request) -> web.Response:
        reached.append(request.path)
        return web.json_response({"ok": True})

    up, up_runner = await _upstream_server(upstream)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    credentials = _credentials(tmp_path / "c.json")
    backend = AnthropicBackend(credentials=credentials, upstream=up)
    try:
        async with backend.serve(runtime):
            sock = backend.socket_path(runtime)
            async with ClientSession(connector=UnixConnector(path=str(sock))) as s:
                headers = {"x-api-key": PLACEHOLDER_KEY}
                async with s.post(
                    "http://anthropic.invalid/v1/messages", data=b"{}", headers=headers
                ) as r:
                    assert r.status == 200

                # The token ages out under the running box. The file is intact
                # and the login is valid -- only the access token is past due.
                _credentials(credentials, ttl_s=-60)

                async with s.post(
                    "http://anthropic.invalid/v1/messages", data=b"{}", headers=headers
                ) as r:
                    assert r.status == 503
                    body = await r.json()
    finally:
        await up_runner.cleanup()

    # Only the first request was worth forwarding: a bearer aisan knows is
    # expired buys an upstream round trip and a worse message.
    assert reached == ["/v1/messages"]
    assert body["error"]["type"] == "authentication_error"
    message = body["error"]["message"]
    assert "expired" in message
    assert "claude auth login" in message and "HOST" in message
    assert str(credentials) in message
    # The number, never the token -- this message reaches the agent's transcript.
    assert FAKE_TOKEN not in message


async def test_one_request_reads_the_credential_file_exactly_once(
    tmp_path, monkeypatch
):
    """Both values out of one read. Two reads would let the host rewrite the
    file in between and pair a fresh expiry with the token from before it --
    which is precisely an expired bearer passing the expiry check."""
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
