# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Claude Code preset: the profile, and the one claim the design rests on.

`claude_code` is a pure function from arguments to a `BoxSpec`, so most of this
reads the spec. Two tests do not, and they are the point of the file:

- `test_the_credential_is_not_in_the_box` runs a REAL box and looks for
  `~/.claude/.credentials.json` from inside it. The claim is not "the preset does
  not list that bind" -- which is a fact about a tuple somebody could restore in
  a one-line diff and no test would notice -- it is that the file is not there.
- `test_the_box_reaches_the_model_only_through_the_relay` asserts the network
  half the same way: a host loopback port is refused, and the backend's port is
  the only one that answers.

Nothing here reads the host's real credential, and no test in this file needs a
model call: the boxed tests use a stub backend whose "upstream" is a local
handler, so the suite is offline and spends no quota. The end-to-end turn against
the real API is validated separately rather than run here -- it
needs a live credential, and a suite that fails when someone's token expires is
a suite people learn to ignore.
"""

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
from aisan.egress.anthropic import PLACEHOLDER_KEY, PORT, AnthropicBackend
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
    """The preset with a worktree, a state dir, and cgroups off.

    A transient systemd scope is not available in every test env and what is
    under test is the mount policy, so `use_cgroup=False` throughout -- the same
    reason the v8_job boxed tests do it.
    """
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
    """The spec-level half of the safety claim, stated where a diff would show.

    Containment, not equality: binding `~/.claude` exposes the credential just
    as surely as binding the file, and so does binding $HOME. So the assertion
    is that the credential is not INSIDE any bind source -- which is the
    property, where "the path is not in the list" is only a spelling of it.

    Bind sources only. The HOME tmpfs's mount point is an ancestor of the
    credential and is the very thing that hides it, so a check over destinations
    would fail on the mount that makes the claim true.

    The real assertion is the boxed one below; this one exists so that a change
    which adds the bind fails at the spec too, with a message naming the
    credential, rather than only in a test that needs bubblewrap to run.
    """
    for m in Box(_spec(tmp_path), box_id="t").mounts():
        if m.src is None:
            continue
        assert not CREDENTIALS.is_relative_to(m.src), f"{m.src} would expose it"


def test_the_state_dir_is_writable_and_the_config_dir_is_not(tmp_path):
    """The split, asserted as the two bindings it is.

    `~/.claude` on the host is one directory doing both jobs. Here the agent's
    own session state is rw because the CLI cannot run otherwise (it writes
    `.claude.json`, `projects/`, `sessions/` within a single turn), and the
    operator's settings are ro because the agent reads policy rather than
    editing it.
    """
    config = tmp_path / "config"
    config.mkdir()
    spec = _spec(tmp_path, config=config)
    modes = {Path(b.path): b.mode for b in spec.binds if hasattr(b, "mode")}
    assert modes[tmp_path / "state"] == RW
    assert modes[config] == RO


def test_the_config_dir_is_optional(tmp_path):
    """A box with no settings file runs the CLI's defaults, which is a real
    configuration rather than a broken one. A preset that required one would make
    every caller invent a file to pass."""
    assert all(Path(b.path) != tmp_path / "config" for b in _spec(tmp_path).binds)


def test_the_box_is_told_the_state_dir_through_claude_config_dir(tmp_path):
    """`CLAUDE_CONFIG_DIR` is the rw half and nothing else names it.

    Measured: a `CLAUDE.md` placed in this directory is obeyed by the turn, so
    the variable carries CONFIG as well as state -- which is exactly why the ro
    half had to be a separate mechanism (`--settings`) and not this.
    """
    env = dict(_spec(tmp_path).env)
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "state")
    assert env["HOME"] == str(Path.home())


def test_the_state_dir_has_no_default(tmp_path):
    """It belongs to the JOB. A preset that picked one would have every box in a
    fleet writing its session history into the same directory."""
    with pytest.raises(TypeError):
        claude_code(tmp_path / "wt")  # type: ignore[call-arg]


def test_egress_requires_the_network_namespace(tmp_path):
    """`unshare_net` defaults to True here, unlike v8_job.

    An agent CLI with a route off the machine has no use for a relay: the box
    would reach the API directly, with whatever the environment gave it, and the
    proxy's path allowlist would be decoration. BoxSpec refuses the combination,
    so this pins the DEFAULT rather than re-asserting the check.
    """
    assert _spec(tmp_path).unshare_net is True


def test_the_argv_helper_puts_settings_before_the_prompt(tmp_path):
    """`--settings` is a flag, not an environment variable -- measured: a
    settings file on a ro bind whose `env` block named a dead port made the turn
    fail to connect, which an unread file could not have done."""
    settings = tmp_path / "settings.json"
    argv = claude_code_argv("hello", settings=settings)
    assert argv == ["claude", "--settings", str(settings), "-p", "hello"]
    assert claude_code_argv("hello") == ["claude", "-p", "hello"]


def test_the_preset_does_not_set_new_session(tmp_path):
    """Deliberately unset, and deliberately untested beyond this.

    `--new-session` blocks TIOCSTI injection back into the launching terminal,
    and also detaches the controlling terminal; whether an INTERACTIVE Claude
    Code survives that is unvalidated -- every measurement behind this preset was
    headless (`-p`), where it does not arise. So this asserts only what the
    preset currently does, and is not evidence that leaving it unset is right.
    """
    assert "--new-session" not in Box(_spec(tmp_path), box_id="t").wrapper()


def test_the_binary_is_found_on_the_usr_surface_or_reported_absent():
    """No node bind, no CLI bind, because `/usr` is already bound ro.

    Returned rather than raised on: a host without the CLI can still describe
    the profile. If this is None the preset still builds -- what breaks is
    running it, and the caller is the one who knows whether it intends to.
    """
    found = claude_code_binary()
    if found is None:
        pytest.skip("claude is not installed on this host")
    assert found.exists()


class _StubBackend(Backend):
    """The Anthropic backend's SHAPE without its credential or its upstream.

    A real `AnthropicBackend` would read the host's own credential file, and a
    test that did that would be one refresh away from failing for a reason
    nobody caused. What the boxed tests below need from a backend is a port and
    a socket that answers, which is all of it that confinement depends on.
    """

    name = "anthropic"
    port = PORT

    def client_env(self):
        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.port}",
            "ANTHROPIC_API_KEY": PLACEHOLDER_KEY,
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
    """The whole safety claim, asserted from INSIDE a real box.

    Not "the preset does not list that bind" -- that is a fact about a tuple,
    and a one-line diff restoring it would pass every spec-level test in this
    file. This looks for the file, on the path the CLI would look on, from the
    only vantage point that can be wrong.

    Skipped when the host has no credential: there the absence proves nothing,
    since the file is missing everywhere.

    The path is PRINTED and its contents never are. A test that dumped what it
    found to prove the box could not read it would put the credential in CI
    output on the day it regressed -- which is the one day it matters.
    """
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
    # And the directory it lives in is not there either, so nothing arrives by a
    # sibling bind that happens to include it.
    assert ".claude" not in r.stdout, r.stdout


async def test_the_box_reaches_the_model_only_through_the_relay(tmp_path):
    """Both halves of the network claim, which are misleading apart.

    The box really has no network -- a host loopback port is refused -- AND the
    backend's port still answers, because a UNIX socket bound in from the host
    crosses the namespace where a port does not. Asserting only the first
    describes a box that cannot work; only the second, a box that is not
    confined.
    """
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
        "print('key=' + os.environ['ANTHROPIC_API_KEY'])\n"
    )
    try:
        r = await _run_in_box(spec, script)
    finally:
        srv.close()
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    assert "relay=from-the-host" in r.stdout
    # ECONNREFUSED on the host's own loopback port: the relay is necessary, not
    # merely convenient.
    assert "host=111" in r.stdout
    # And what the box holds is the placeholder, which is the other half of why
    # the credential's absence is survivable.
    assert f"key={PLACEHOLDER_KEY}" in r.stdout


def test_the_default_entry_builds_without_a_deployment(tmp_path):
    """The registry contract: `PRESETS[name](root)` on a host with no config,
    no credential and no deployment. A dry run that needed one would not be a
    dry run."""
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = claude_code_default(wt)
    assert spec.root == wt
    assert spec.egress == ()


def test_the_real_backend_and_the_preset_agree_on_the_port(tmp_path):
    """The preset does not name the port -- the backend does, and the client env
    comes from it. Pinned so a port change cannot leave the box dialling a relay
    nobody is serving."""
    spec = _spec(tmp_path, egress=(AnthropicBackend(credentials=tmp_path / "c"),))
    assert spec.egress[0].client_env()["ANTHROPIC_BASE_URL"].endswith(f":{PORT}")
