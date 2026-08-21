# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Tests for the manifest-driven in-box launcher.

The manifest is the single description of relay wiring. These tests assert that
N backends work with no launcher change and that the payload's exit status
survives.

The last test is the whole seam end to end -- a real box, a real network
namespace, a real relay -- because that is precisely what a unit test cannot
see: the reason the previous transport failed silently under isolation was that
every part looked right alone.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from aisan import launch as launch_mod
from aisan.launch import (
    interpreter_roots,
    launch_prefix,
    launcher_binds,
    own_source_root,
)
from aisan.runtime import write_client_env, write_manifest
from aisan.sandbox import RO


def _launch(runtime_dir: Path, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run the launcher out of process, the way a box does."""
    return subprocess.run(
        [*launch_prefix(runtime_dir), *cmd],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        **kw,
    )


def test_the_launcher_is_named_by_path_not_by_a_path_lookup(tmp_path):
    # A box's PATH deliberately holds several venvs' bin dirs, so picking the
    # launcher by name would make which aisan runs a function of bind order.
    # `python -m` at an absolute interpreter path names exactly the interpreter
    # launcher_binds bound, with no shim to resolve.
    prefix = launch_prefix(tmp_path)
    assert prefix[0] == sys.executable
    assert prefix[1] == "-m" and prefix[2].endswith("aisan.launch")
    assert prefix[-2:] == [str(tmp_path), "--"]


async def test_shared_mode_injects_client_env_then_execs(tmp_path, monkeypatch):
    write_client_env(tmp_path, {"AISAN_TEST_TOKEN": "secret"})
    seen = {}

    def execvpe(file, argv, env):
        seen.update(file=file, argv=argv, env=env)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(launch_mod.os, "execvpe", execvpe)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        await launch_mod._run(tmp_path, ["payload", "arg"])
    assert seen["file"] == "payload"
    assert seen["argv"] == ["payload", "arg"]
    assert seen["env"]["AISAN_TEST_TOKEN"] == "secret"


def test_launcher_binds_cover_the_venv_and_every_interpreter_hop():
    # uv keeps two names for one interpreter: a versioned dir and an unversioned
    # symlink to it, and a venv's bin/python points at the UNVERSIONED one. So
    # binding only .resolve() leaves the name the venv actually names absent,
    # and bwrap fails the exec with "No such file or directory" -- which reads
    # as a missing venv rather than a missing hop.
    binds = launcher_binds()
    paths = [Path(str(b.path)) for b in binds]
    venv = Path(sys.executable).parent.parent
    assert paths[0] == venv
    assert all(b.mode is RO for b in binds)  # nothing in here is ours to write
    for root in interpreter_roots(Path(sys.executable)):
        assert root in paths


def test_launcher_binds_carry_this_package_and_no_other_editable_tree():
    # A package installed editable beside us must not end up mounted into a box
    # that only needs a relay.
    #
    # Asserted against the venv's own dist metadata rather than a fixed list, so
    # this fails in exactly the workspace where a regression would matter and
    # says nothing on a standalone install that has no other editable trees.
    import json
    from importlib.metadata import distributions

    binds = {Path(str(b.path)) for b in launcher_binds()}
    venv = Path(sys.executable).parent.parent
    foreign = []
    for dist in distributions():
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        info = json.loads(raw)
        if not info.get("dir_info", {}).get("editable"):
            continue
        root = Path(info["url"].removeprefix("file://"))
        # Ours is the tree this module is imported from; every other editable
        # root belongs to a consumer and has no business in the box.
        if root.exists() and Path(__file__).resolve().is_relative_to(root):
            continue
        if not root.is_relative_to(venv):
            foreign.append(root)

    assert not (binds & set(foreign)), (
        f"launcher_binds leaked a consumer's source tree: {binds & set(foreign)}"
    )
    # Not vacuous only where there IS a foreign editable dist; where there is
    # not, the assertion above is trivially true and this records why.
    assert own_source_root() is None or own_source_root() in binds


def test_own_source_root_is_none_for_a_non_editable_install(tmp_path, monkeypatch):
    # Installed normally the package lives in site-packages, which the venv bind
    # already covers -- binding its parent again would mount the same tree twice
    # under a different name. Simulated by relocating the module, since the test
    # venv is necessarily an editable one.
    fake = tmp_path / "lib" / "python3.13" / "site-packages" / "aisan"
    fake.mkdir(parents=True)
    monkeypatch.setattr(launch_mod, "__file__", str(fake / "launch.py"))
    assert own_source_root() is None


def test_interpreter_roots_walks_a_symlink_chain(tmp_path):
    # Built by hand rather than asserted on this host's layout: the chain that
    # matters (venv -> unversioned alias -> versioned dir) only exists on a uv
    # managed interpreter, and a test that skipped elsewhere would not guard it.
    real = tmp_path / "cpython-3.12.12" / "bin"
    real.mkdir(parents=True)
    (real / "python3").write_text("#!/bin/true\n")
    alias = tmp_path / "cpython-3.12"
    alias.symlink_to(tmp_path / "cpython-3.12.12")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(alias / "bin" / "python3")
    roots = interpreter_roots(venv / "python")
    assert tmp_path / "venv" in roots  # the venv the exec names
    assert tmp_path / "cpython-3.12.12" in roots  # where it really lands


def test_an_empty_manifest_still_runs_the_payload(tmp_path):
    # Degenerate but reachable: a spec with backends whose manifest is empty is
    # not a broken box, it is a box with nothing to relay. The launcher must not
    # treat "no relays" as "nothing to do".
    write_manifest(tmp_path, [])
    r = _launch(tmp_path, ["echo", "hello"])
    assert r.returncode == 0
    assert r.stdout.strip() == "hello"


def test_the_payloads_exit_status_is_the_launchers(tmp_path):
    # The launcher is in the middle of the status path: a job's caller reads the
    # exit code to decide whether the build failed, and a launcher that returned
    # its own would make every payload look successful.
    write_manifest(tmp_path, [])
    assert _launch(tmp_path, ["sh", "-c", "exit 7"]).returncode == 7


def test_a_payload_killed_by_a_signal_reports_the_shell_convention(tmp_path):
    # A signalled payload has no exit code. Reporting it as a shell does (128+N)
    # means a caller reading the status sees the same number it would have seen
    # without the launcher in the middle.
    write_manifest(tmp_path, [])
    r = _launch(tmp_path, ["sh", "-c", "kill -TERM $$"])
    assert r.returncode == 128 + 15


async def test_the_launcher_serves_every_relay_in_the_manifest(tmp_path):
    """N backends, no launcher change -- the property the manifest buys.

    Two host-side listeners on two UNIX sockets, two ports in the manifest, one
    launcher: the payload dials both ports and each answer identifies which
    socket it came from. A launcher that served only the first would pass any
    test with one backend in it, which is why there are two here.
    """
    sockets = {}
    servers = []

    async def _echo(name):
        async def handle(reader, writer):
            await reader.read(16)
            writer.write(name.encode())
            await writer.drain()
            writer.close()

        return handle

    for name, port in (("a", 18801), ("b", 18802)):
        path = tmp_path / f"{name}.sock"
        sockets[name] = (path, port)
        servers.append(await asyncio.start_unix_server(await _echo(name), str(path)))
    try:
        write_manifest(
            tmp_path,
            [
                {"socket": str(p), "port": port, "name": name}
                for name, (p, port) in sockets.items()
            ],
        )
        script = (
            "import socket\n"
            "for port in (18801, 18802):\n"
            "    s = socket.create_connection(('127.0.0.1', port), 5)\n"
            "    s.sendall(b'ping'); print(s.recv(16).decode()); s.close()\n"
        )
        r = await asyncio.to_thread(_launch, tmp_path, [sys.executable, "-c", script])
        assert r.returncode == 0, r.stderr
        assert r.stdout.split() == ["a", "b"]
    finally:
        for s in servers:
            s.close()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
async def test_an_isolated_box_reaches_the_host_only_through_the_relay(tmp_path):
    """The whole point of the socket transport, end to end.

    A box with its OWN network namespace cannot reach a host loopback port, so
    everything here works the other way round: the host half listens on a UNIX
    socket bound into the box, the launcher serves the port the client dials
    inside the box, and the two are spliced. Both halves of the claim are
    asserted together, because either alone is misleading -- the box really has
    no network, AND it can still reach the host half.

    Through a real Box rather than a hand-built argv: what is under test is the
    composition (wrapper, then launcher, then payload) that `command()` owns.
    """
    import contextlib

    from aisan import Box
    from aisan.egress.base import Backend
    from aisan.sandbox import RO, Bind
    from aisan.spec import BoxSpec, Limits

    seen: list[bytes] = []

    class _EchoBackend(Backend):
        name = "echo"
        port = 18811

        def client_env(self):
            return {"ECHO_ENDPOINT": f"127.0.0.1:{self.port}"}

        @contextlib.asynccontextmanager
        async def serve(self, runtime_dir: Path):
            async def handle(reader, writer):
                seen.append(await reader.read(16))
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

    # A host listener the box must NOT be able to reach: the negative half.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host_port = srv.getsockname()[1]

    root = tmp_path / "root"
    root.mkdir()
    script = root / "probe.py"
    script.write_text(
        "import os, socket\n"
        "host, port = os.environ['ECHO_ENDPOINT'].split(':')\n"
        "s = socket.create_connection((host, int(port)), 5)\n"
        "s.sendall(b'ping'); print('relay=' + s.recv(32).decode()); s.close()\n"
        "n = socket.socket(); n.settimeout(3)\n"
        f"print('host=%d' % n.connect_ex(('127.0.0.1', {host_port})))\n"
    )
    spec = BoxSpec(
        root=root,
        # The interpreter the payload runs; the launcher's own binds are added
        # by the Box, but the PAYLOAD's interpreter is the consumer's business.
        binds=(Bind(Path(sys.executable).parent.parent, RO),),
        tmpfs=(),
        env=(("PATH", "/usr/bin:/bin"),),
        egress=(_EchoBackend(),),
        unshare_net=True,
        limits=Limits(use_cgroup=False),  # no systemd scope in a test env
    )
    box = Box(spec, box_id=str(tmp_path / "job"))
    try:
        async with box:
            argv = box.command([sys.executable, str(script)])
            env = {**os.environ, **box.env}
            r = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=env,
            )
    finally:
        srv.close()
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    # The positive half: the request crossed the namespace and the host half saw
    # it. Asserted on the HOST's view, so a listener that merely accepts cannot
    # pass.
    assert "relay=from-the-host" in r.stdout
    assert seen == [b"ping"]
    # The negative half: ECONNREFUSED (111) on the host's own loopback port,
    # which is what makes the relay necessary rather than merely convenient.
    assert "host=111" in r.stdout
