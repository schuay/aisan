# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The OpenAI-compatible backend and preset: what they read, refuse, never do.

`test_proxy_openai_compat.py` covers the transport (the route, the body
policy, the injected bearer). This covers the backend -- the credential file,
the catalog-driven route, the preflight decisions -- and the preset, in the
shape of `test_aisan_claude_code.py`, because the two agent profiles carry the
same safety claim and it deserves the same assertions:

- `test_the_credential_is_not_in_the_box` runs a REAL box and looks for the
  host's `auth.json` from inside it. The claim is not "the preset does not
  list that bind" -- it is that the file is not there.
- `test_the_box_reaches_the_model_only_through_the_relay` asserts the network
  half: a host loopback port is refused, and the backend's port answers.

One test here has no claude-code equivalent and is the strongest in the file:
`test_a_real_opencode_turn_through_the_whole_chain` runs the REAL binary in a
real box, through the relay, the proxy and a stub upstream, and asserts the
stub's error message comes back out of the client. It is skipped on hosts
without the binary; nothing here needs a real credential or spends quota --
the upstream is a local handler and the key is fake by construction.
"""

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
from aisan.sandbox import RO, RW

# Not a credential: a fixed string, in a file this test wrote, against an
# upstream that is a local aiohttp handler. Named so a reader does not have to
# work that out.
FAKE_KEY = "fake-api-key-for-tests"

PROVIDER = "zai-coding-plan"


def _auth(path: Path, *, provider: str = PROVIDER, key: str = FAKE_KEY) -> Path:
    """An auth.json in the real shape: a static API key for one provider."""
    path.write_text(json.dumps({provider: {"type": "api", "key": key}}))
    path.chmod(0o600)
    return path


def _oauth_auth(path: Path, *, provider: str = PROVIDER) -> Path:
    """The shape this backend does NOT carry: a subscription login."""
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
    """The backend with fake credentials under this run's tmp dir.

    The default is filled only when absent -- a `setdefault(..., _auth(...))`
    would run `_auth` eagerly and overwrite whatever file the test just
    corrupted on purpose.
    """
    if "credentials" not in kw:
        kw["credentials"] = _auth(tmp_path / "auth.json")
    return OpenAICompatBackend(provider=PROVIDER, **kw)


# --- what the box is told ----------------------------------------------------


def test_the_box_is_told_a_placeholder_a_route_and_a_model():
    """All three client facts ride two env vars, and none is real.

    The auth content is a COMPLETE in-memory store for the provider (measured:
    when set, opencode does not read auth.json at all), the base URL points at
    the relay's bare origin -- the provider's path prefix is host-side
    knowledge -- and the model is composed here so caller and config cannot
    desynchronise the two halves of `provider/model`.
    """
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
    # Sharing uploads the transcript off the machine; it is off where project
    # config cannot override it.
    assert cfg["share"] == "disabled"
    assert "placeholder" in PLACEHOLDER_KEY


def test_a_backend_without_a_model_picks_none(tmp_path):
    """No model argument means no model in the config content -- the client's
    own default or the user's /models choice, not a silent pin."""
    b = _backend(tmp_path, upstream="https://x")
    assert "model" not in json.loads(b.client_env()["OPENCODE_CONFIG_CONTENT"])


def test_client_env_reads_no_files(tmp_path):
    """`client_env` is pure: neither the credential nor the catalog is read.

    This is what lets `explain`, a dry run and a snapshot describe the profile
    on a host with no login and no catalog -- the wire facts are resolved at
    preflight and serve, where failing is the point."""
    b = OpenAICompatBackend(
        provider=PROVIDER,
        model="m",
        upstream="https://x",
        credentials=tmp_path / "absent.json",
        catalog=tmp_path / "absent.json",
    )
    assert "OPENCODE_AUTH_CONTENT" in b.client_env()


def test_the_port_is_distinct_from_the_other_backends():
    """`BoxSpec.__post_init__` asserts injectivity across a spec's backends,
    but only for the backends that spec actually has. A collision between two
    backends nobody composed yet is found here or when a box mysteriously
    reaches the wrong upstream."""
    from aisan.egress.anthropic import PORT as ANTHROPIC_PORT
    from aisan.egress.reapi import PORT as REAPI_PORT
    from aisan.egress.vertex import PORT as VERTEX_PORT

    assert len({PORT, ANTHROPIC_PORT, REAPI_PORT, VERTEX_PORT}) == 4


def test_two_providers_of_the_family_need_two_ports(tmp_path):
    """One route per box is the common shape, not the only expressible one:
    a second provider is a name and a port override, and the spec composes.
    Without the override the collision is refused -- see BoxSpec."""
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


# --- preflight: the route, the credential, the upstream ----------------------


async def test_preflight_passes_when_the_route_the_key_and_upstream_are_there(
    tmp_path,
):
    up, runner = await _upstream_server(_hello)
    try:
        await _backend(tmp_path, upstream=up).preflight()
    finally:
        await runner.cleanup()


async def test_the_route_resolves_from_the_catalog(tmp_path):
    """The data-driven path: no upstream given, and the provider's api base --
    prefix included -- comes out of the catalog the host's opencode fetched.
    The upstream check runs against the RESOLVED route, so a passing preflight
    here is also evidence the resolution produced a real URL."""
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
    """No catalog on this host, or the provider is unknown to it. The fix is
    to fetch one or to name the upstream -- not to log in, which could not
    help. Resolution is checked before anything dials, so no upstream is
    needed to see it fail."""
    with pytest.raises(PreflightError) as e:
        await _backend(
            tmp_path, upstream=None, catalog=tmp_path / "absent.json"
        ).preflight()
    assert "cannot resolve the route" in e.value.reason
    assert "opencode once on the HOST" in e.value.fix
    assert "login" not in e.value.fix


async def test_an_explicit_upstream_needs_no_catalog(tmp_path):
    """The constructor escape hatch, and the reason a host that has never run
    opencode can still run a box: name the api base and the catalog is never
    consulted."""
    up, runner = await _upstream_server(_hello)
    try:
        await _backend(
            tmp_path, upstream=up, catalog=tmp_path / "absent.json"
        ).preflight()
    finally:
        await runner.cleanup()


async def test_a_missing_credential_refuses_with_the_login_command(tmp_path):
    """Not connected HERE, so the fix is a login -- and the message has to
    carry it. A PreflightError whose fix is wrong is worse than none."""
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
    """An oauth entry exists and is unreadable BY POLICY, not by accident:
    those refresh, and a refresh from here would race the host's opencode over
    a file neither side locks. No login fixes that, so the fix must not say
    login."""
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
    """Nothing to log into: the fix is reachability, stated as that."""
    up, runner = await _upstream_server(_hello)
    await runner.cleanup()  # dead address, deliberately
    with pytest.raises(PreflightError) as e:
        await _backend(tmp_path, upstream=up).preflight()
    assert "does not answer" in e.value.reason
    assert "check the network" in e.value.fix
    assert "login" not in e.value.fix


async def test_preflight_does_not_spend_a_model_call(tmp_path):
    """A connect, not a turn. Asking the upstream something real would spend
    quota to learn that the hop exists."""
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
    """The message must never quote the document: a parse error that echoed
    what it failed to parse would put the key wherever that message is
    logged."""
    bad = tmp_path / "auth.json"
    bad.write_text('{"p": {"type": "api", "key": "' + FAKE_KEY + '"')  # truncated
    up, runner = await _upstream_server(_hello)
    try:
        with pytest.raises(PreflightError) as e:
            await _backend(tmp_path, upstream=up, credentials=bad).preflight()
    finally:
        await runner.cleanup()
    assert FAKE_KEY not in str(e.value)
    assert str(bad) in str(e.value)


# --- serve: the two halves joined --------------------------------------------


async def test_the_backend_serves_and_injects_the_credential_it_read(tmp_path):
    """`serve` puts the transport on the socket the Box binds, and the bearer
    that arrives upstream is the one in the file -- with the provider's path
    prefix preserved, because the box's requests arrive prefix-less and the
    prefix lives in the upstream the proxy dials."""
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
        # Torn down with the block: a proxy surviving its box is an
        # unauthenticated capability to the credential behind it.
        assert not backend.socket_path(runtime).exists()
    finally:
        await up_runner.cleanup()

    assert got["authorization"] == f"Bearer {FAKE_KEY}"
    assert paths == ["/v4/chat/completions"]


async def test_the_backend_never_writes_the_credential(tmp_path):
    """The settled rule from the Anthropic backend, asserted the same way:
    over a full preflight-serve-request cycle, by bytes and mtime -- a
    rewrite with identical content is still a rewrite."""
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
    # Still only readable by its owner: nothing here relaxed the mode either.
    assert creds.stat().st_mode & 0o777 == 0o600


async def test_a_stale_socket_from_a_killed_box_does_not_block_the_next_one(
    tmp_path,
):
    """bind() fails EADDRINUSE forever on a leftover socket file, so a box
    killed with SIGKILL would poison its own runtime dir. The path is derived
    from this box, so nothing else owns it and unlinking is safe."""
    up, up_runner = await _upstream_server(_hello)
    runtime = tmp_path / "rt"
    runtime.mkdir()
    backend = _backend(tmp_path, upstream=up)
    backend.socket_path(runtime).write_bytes(b"")  # stale socket of a killed box
    try:
        async with backend.serve(runtime):
            assert backend.socket_path(runtime).is_socket()
    finally:
        await up_runner.cleanup()


# --- the preset ---------------------------------------------------------------


def _spec(tmp_path, **kw):
    """The preset with a worktree, a state dir, a catalog and cgroups off.

    A transient systemd scope is not available in every test env and what is
    under test is the mount policy -- the same reason the claude_code boxed
    tests do it. The catalog is pinned to a file this test wrote, so the spec
    does not depend on whether this host has run opencode.
    """
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
    """The spec-level half of the safety claim, stated where a diff would show.

    Containment, not equality: binding `~/.local/share` exposes the credential
    just as surely as binding the file, and so does binding $HOME. The real
    assertion is the boxed one below; this one exists so that a change which
    adds the bind fails at the spec too, with a message naming the credential.
    """
    for m in Box(_spec(tmp_path), box_id="t").mounts():
        if m.src is None:
            continue
        assert not DEFAULT_CREDENTIALS.is_relative_to(m.src), f"{m.src} would expose it"


def test_the_state_dir_is_rw_and_the_catalog_is_ro_and_optional(tmp_path):
    """Session history is the agent's to write; the catalog is a host fact it
    reads. Optional, so a host without one still builds the profile."""
    spec = _spec(tmp_path)
    modes = {
        Path(b.path): (b.mode, b.optional) for b in spec.binds if hasattr(b, "mode")
    }
    assert modes[tmp_path / "state"] == (RW, False)
    assert modes[tmp_path / "models.json"] == (RO, True)


def test_a_state_dir_inside_the_root_needs_no_bind(tmp_path):
    """Writable there by the root's own mount; binding it again would require
    it to EXIST at assembly time, which a caller expecting the client to
    create it has no reason to arrange. Same rule as the claude_code preset's."""
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = opencode(wt, state=wt / ".aisan-opencode-state", models=tmp_path / "m.json")
    assert all(Path(b.path) != wt / ".aisan-opencode-state" for b in spec.binds)


def test_the_environment_names_the_state_dir_and_disables_the_fetchers(tmp_path):
    """`XDG_DATA_HOME` is the one redirect that persists (sessions, logs, the
    db); the three disable flags are legibility knobs, not security controls --
    each stops a fetcher that would otherwise hang on a network that is not
    coming back and look like a working box."""
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
    """The trap the preset exists to avoid: git reads
    `$XDG_CONFIG_HOME/git/config` INSTEAD of `~/.config/git/config` when it is
    set, so redirecting it would silently bypass the ro bind a launcher puts
    at `~/.config/git`. Unset, the config root lands on the HOME tmpfs."""
    assert "XDG_CONFIG_HOME" not in dict(_spec(tmp_path).env)


def test_the_state_dir_has_no_default(tmp_path):
    """It belongs to the JOB. A preset that picked one would have every box in
    a fleet writing its session history into the same directory."""
    with pytest.raises(TypeError):
        opencode(tmp_path / "wt")  # type: ignore[call-arg]


def test_egress_requires_the_network_namespace(tmp_path):
    assert _spec(tmp_path).unshare_net is True


def test_the_default_entry_builds_without_a_deployment(tmp_path):
    """The registry contract: `PRESETS[name](root)` on a host with no config,
    no credential and no deployment."""
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = opencode_default(wt)
    assert spec.root == wt
    assert spec.egress == ()


def test_the_binary_is_found_on_the_usr_surface_or_reported_absent():
    """No binary bind, because `/usr` is already bound ro. Returned rather
    than raised on: a host without the binary can still describe the
    profile."""
    found = opencode_binary()
    if found is None:
        pytest.skip("opencode is not installed on this host")
    assert found.exists()


def test_the_real_backend_and_the_preset_agree_on_the_port(tmp_path):
    """The preset does not name the port -- the backend does, and the client
    env comes from it. Pinned so a port change cannot leave the box dialling a
    relay nobody is serving."""
    spec = _spec(tmp_path, egress=(_backend(tmp_path, upstream="https://x"),))
    cfg = json.loads(spec.egress[0].client_env()["OPENCODE_CONFIG_CONTENT"])
    assert cfg["provider"][PROVIDER]["options"]["baseURL"].endswith(f":{PORT}")


# --- the claims, asserted from inside a real box ------------------------------


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
    """The whole safety claim, asserted from INSIDE a real box.

    Not "the preset does not list that bind" -- that is a fact about a tuple,
    and a one-line diff restoring it would pass every spec-level test above.
    This looks for the file, on the path the client would look on, from the
    only vantage point that can be wrong.

    The path is PRINTED and its contents never are, for the same reason as the
    claude_code suite's twin: a test that dumped what it found would put the
    credential in CI output on the day it regressed.
    """
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
    # The credential's whole directory is absent, not merely the file -- so
    # nothing arrived by a bind of an ancestor that happens to include it.
    # NOT asserted over the home listing: `.local` legitimately exists in a
    # box on a host whose aisan runtime lives under ~/.local/share/uv (the
    # launcher binds the interpreter at its real path), which is a fact about
    # the install rather than the profile.
    assert "share_exists=False" in r.stdout, r.stdout


async def test_the_box_reaches_the_model_only_through_the_relay(tmp_path):
    """Both halves of the network claim, which are misleading apart -- and,
    unlike the stub the claude suite needs, through the REAL backend: its
    credentials argument makes a fake key enough, so the client env, the
    socket and the proxy in this test are the real implementations.

    The in-box script speaks HTTP, not a raw ping: the bytes it gets back have
    crossed the relay, the unix socket, the proxy and the stub upstream, and
    the body it prints came from the far end.
    """

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
    # ECONNREFUSED on the host's own loopback port: the relay is necessary,
    # not merely convenient.
    assert "host=111" in r.stdout


@pytest.mark.skipif(opencode_binary() is None, reason="opencode is not installed")
async def test_a_real_opencode_turn_through_the_whole_chain(tmp_path):
    """The strongest claim in the file: the REAL client, in a REAL box, turns
    against a stub upstream, and the stub's error message comes back out of
    the client's own output.

    That message is the proof of the whole chain at once -- it exists only on
    the far side of the relay, the unix socket, the proxy and the client's
    error handling -- and it is also the error-envelope claim in code: an
    openai-shaped body surfaces verbatim (measured; any other shape collapses
    to a generic "Unexpected server error").

    glm-5.2 because it is in the binary's EMBEDDED catalog: no fetched catalog
    needed, so the box under test needs nothing from this host's cache. No
    real credential is read -- the backend's is fake, and the client's key is
    the placeholder.
    """
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
    # models= an absent path, not the default: the default binds the HOST's
    # fetched catalog, whose inode the host's own opencode can unlink and
    # replace mid-run -- a bind of the dead inode stats fine but cannot be
    # re-bound by a nested bwrap, so the box under test would depend on live
    # host state this docstring already disclaims. Absent -> optional drop ->
    # the embedded catalog the model below was picked for.
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
