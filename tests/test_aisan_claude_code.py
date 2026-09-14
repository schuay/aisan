# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from aisan import Box
from aisan.egress.anthropic import (
    OAUTH_TOKEN_ENV,
    PLACEHOLDER_KEY,
    PORT,
    AnthropicBackend,
)
from aisan.egress.base import Backend
from aisan.presets import PRESETS
from aisan.presets.claude_code import (
    claude_code,
    claude_code_argv,
    claude_code_binary,
    claude_code_default,
)
from aisan.sandbox import RO, RW

CREDENTIALS = Path.home() / ".claude" / ".credentials.json"


def _spec(tmp_path, **kw):
    import dataclasses

    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    spec = claude_code(wt, state=state, **kw)
    return dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )


def test_the_preset_is_registered_under_its_own_name():
    assert PRESETS["claude_code"] is claude_code_default


def test_no_bind_source_contains_the_credential(tmp_path):
    for m in Box(_spec(tmp_path), box_id="t").mounts():
        if m.src is None:
            continue
        assert not CREDENTIALS.is_relative_to(m.src), f"{m.src} would expose it"


def test_the_state_dir_is_writable_and_the_config_dir_is_not(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    spec = _spec(tmp_path, config=config)
    modes = {Path(b.path): b.mode for b in spec.binds if hasattr(b, "mode")}
    assert modes[tmp_path / "state"] == RW
    assert modes[config] == RO


def test_the_config_dir_is_optional(tmp_path):
    assert all(Path(b.path) != tmp_path / "config" for b in _spec(tmp_path).binds)


def test_host_skills_mount_inside_the_redirected_config_dir(tmp_path):
    """Claude Code reads `$CLAUDE_CONFIG_DIR/skills`, never `~/.claude/skills`."""
    skills = tmp_path / "host-skills"
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n")
    state = tmp_path / "state"

    mounts = Box(_spec(tmp_path, skills=skills), box_id="t").mounts()
    at_state = [m for m in mounts if m.dst == state]
    at_skills = [m for m in mounts if m.dst == state / "skills"]

    assert [(m.op, m.src) for m in at_skills] == [("ro", skills)]
    # The state bind is writable, so the skills mount must come after it.
    assert mounts.index(at_skills[0]) > mounts.index(at_state[-1])


def test_the_skills_mount_point_is_created_on_the_host(tmp_path):
    skills = tmp_path / "host-skills"
    skills.mkdir()
    ensured = {e.path: e.is_dir for e in _spec(tmp_path, skills=skills).ensure}
    assert ensured[tmp_path / "state" / "skills"] is True


def test_skills_are_optional(tmp_path):
    spec = _spec(tmp_path)
    assert all(
        getattr(b, "dst", None) != tmp_path / "state" / "skills" for b in spec.binds
    )
    assert all(e.path != tmp_path / "state" / "skills" for e in spec.ensure)


def test_the_box_is_told_the_state_dir_through_claude_config_dir(tmp_path):
    env = dict(_spec(tmp_path).env)
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "state")
    assert env["HOME"] == str(Path.home())


def test_the_box_keeps_its_flags_off_disk_with_the_traffic_off(tmp_path):
    env = dict(_spec(tmp_path).env)
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["CLAUDE_CODE_GB_DISK_CACHE_WHEN_TELEMETRY_OFF"] == "1"


def test_the_state_dir_has_no_default(tmp_path):
    with pytest.raises(TypeError):
        claude_code(tmp_path / "wt")  # type: ignore[call-arg]


def test_egress_requires_the_network_namespace(tmp_path):
    assert _spec(tmp_path).unshare_net is True


def test_the_argv_helper_puts_settings_before_the_prompt(tmp_path):
    settings = tmp_path / "settings.json"
    argv = claude_code_argv("hello", settings=settings)
    assert argv == ["claude", "--settings", str(settings), "-p", "hello"]
    assert claude_code_argv("hello") == ["claude", "-p", "hello"]


def test_the_preset_does_not_set_new_session(tmp_path):
    assert "--new-session" not in Box(_spec(tmp_path), box_id="t").wrapper()


def test_the_binary_is_found_on_the_usr_surface_or_reported_absent():
    found = claude_code_binary()
    if found is None:
        pytest.skip("claude is not installed on this host")
    assert found.exists()


class _StubBackend(Backend):
    name = "anthropic"
    port = PORT

    def client_env(self):

        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.port}",
            OAUTH_TOKEN_ENV: PLACEHOLDER_KEY,
        }

    @contextlib.asynccontextmanager
    async def serve(self, runtime_dir: Path):
        async def handle(reader, writer):
            await reader.read(16)
            writer.write(b"from-the-host")
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(
            handle, str(self.socket_path(runtime_dir))
        )
        try:
            yield
        finally:
            server.close()


async def _run_in_box(spec, script: str, extra_env=None) -> subprocess.CompletedProcess:
    box = Box(spec, box_id="claude-code-test")
    async with box:
        argv = box.command([sys.executable, "-c", script])
        env = {**os.environ, **box.env, **(extra_env or {})}
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
    not CREDENTIALS.exists(), reason="no host credential, so nothing to hide"
)
async def test_the_credential_is_not_in_the_box(tmp_path):
    spec = _spec(tmp_path, egress=(_StubBackend(),))
    script = (
        "import os\n"
        f"p = {str(CREDENTIALS)!r}\n"
        "print('exists=%s' % os.path.exists(p))\n"
        "print('home_listing=%s' % sorted(os.listdir(os.path.expanduser('~'))))\n"
    )
    r = await _run_in_box(spec, script)
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    assert "exists=False" in r.stdout

    assert ".claude" not in r.stdout, r.stdout


async def test_the_box_reaches_the_model_only_through_the_relay(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host_port = srv.getsockname()[1]
    spec = _spec(tmp_path, egress=(_StubBackend(),))
    script = (
        "import os, socket\n"
        f"s = socket.create_connection(('127.0.0.1', {PORT}), 5)\n"
        "s.sendall(b'ping'); print('relay=' + s.recv(32).decode()); s.close()\n"
        "n = socket.socket(); n.settimeout(3)\n"
        f"print('host=' + str(n.connect_ex(('127.0.0.1', {host_port}))))\n"
        f"print('token=' + os.environ[{OAUTH_TOKEN_ENV!r}])\n"
    )
    try:
        r = await _run_in_box(spec, script)
    finally:
        srv.close()
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    assert "relay=from-the-host" in r.stdout

    assert "host=111" in r.stdout

    assert f"token={PLACEHOLDER_KEY}" in r.stdout


def test_the_default_entry_builds_without_a_deployment(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = claude_code_default(wt)
    assert spec.root == wt
    assert spec.egress == ()


def test_the_real_backend_and_the_preset_agree_on_the_port(tmp_path):
    spec = _spec(tmp_path, egress=(AnthropicBackend(credentials=tmp_path / "c"),))
    assert spec.egress[0].client_env()["ANTHROPIC_BASE_URL"].endswith(f":{PORT}")


def test_the_real_backend_and_the_preset_agree_on_the_client_env(tmp_path):
    real = AnthropicBackend(credentials=tmp_path / "c").client_env()
    assert set(_StubBackend().client_env()) == set(real)
