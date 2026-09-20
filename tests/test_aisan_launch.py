# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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
    interpreter_chain_dirs,
    interpreter_roots,
    launch_prefix,
    launcher_binds,
    own_source_root,
)
from aisan.runtime import (
    CLIENT_ENV_NAME,
    cleanup_runtime_dir,
    prepare_runtime_dir,
    write_client_env,
    write_manifest,
)
from aisan.sandbox import RO


def _launch(runtime_dir: Path, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*launch_prefix(runtime_dir), *cmd],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        **kw,
    )


def test_the_launcher_is_named_by_path_not_by_a_path_lookup(tmp_path):

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


def test_launcher_binds_cover_both_interpreter_prefixes_and_the_link_chain():

    binds = launcher_binds()
    paths = [Path(str(b.path)) for b in binds]
    assert Path(sys.prefix) in paths
    assert Path(sys.base_prefix) in paths
    assert all(b.mode is RO for b in binds)
    for hop in interpreter_chain_dirs(Path(sys.executable)):
        assert hop in paths


@pytest.mark.skipif(
    not Path("/bin/true").exists() or not Path("/bin").is_symlink(),
    reason="needs a merged-/usr host",
)
def test_launcher_binds_name_a_system_symlink_dir_by_its_target(tmp_path):
    """`/bin` is a symlink inside every box, and the sandbox refuses a
    destination through a link, so the chain directory is renamed."""
    binds = [Path(str(b.path)) for b in launcher_binds(Path("/bin/true"))]
    assert Path("/bin") not in binds
    assert Path("/usr/bin") in binds


def test_launcher_binds_omit_paths_the_system_surface_already_mounts(tmp_path):

    usr = tmp_path / "usr" / "bin"
    usr.mkdir(parents=True)
    py = usr / "python3"
    py.write_text("")
    # A system interpreter reports the system prefix for both values.
    binds = [
        Path(str(b.path)) for b in launcher_binds(py, (Path("/usr"), Path("/usr")))
    ]
    assert Path("/usr") not in binds
    assert usr in binds, "the link chain still needs its own directory"
    assert len(binds) == len(set(binds))


def test_launcher_binds_reach_source_masked_by_a_later_tmpfs(tmp_path, monkeypatch):

    # A checkout whose path lies under an interpreter prefix by name only. The
    # box masks the tree between them, so prefix containment cannot stand in for
    # readability and the source bind must be emitted regardless.
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    py = prefix / "bin" / "python3"
    py.write_text("")
    own = prefix / "nested" / "checkout" / "src"
    own.mkdir(parents=True)
    monkeypatch.setattr(launch_mod, "own_source_root", lambda: own)
    binds = [Path(str(b.path)) for b in launcher_binds(py, (prefix, prefix))]
    assert own in binds


def test_launcher_binds_do_not_claim_the_home_directory_as_a_prefix(tmp_path):

    # An interpreter reached through a personal link farm sits at <home>/bin/x,
    # whose parent-of-parent is the home directory. Binding that would shadow
    # the box's home tmpfs and every profile would be refused.
    farm = tmp_path / "bin"
    farm.mkdir()
    py = farm / "python3"
    py.write_text("")
    binds = [
        Path(str(b.path)) for b in launcher_binds(py, (Path("/usr"), Path("/usr")))
    ]
    assert tmp_path not in binds
    assert farm in binds


def test_launcher_binds_carry_this_package_and_no_other_editable_tree():

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

        if root.exists() and Path(__file__).resolve().is_relative_to(root):
            continue
        if not root.is_relative_to(venv):
            foreign.append(root)

    assert not (binds & set(foreign)), (
        f"launcher_binds leaked a consumer's source tree: {binds & set(foreign)}"
    )

    assert own_source_root() is None or own_source_root() in binds


def test_own_source_root_is_none_for_a_non_editable_install(tmp_path, monkeypatch):

    fake = tmp_path / "lib" / "python3.13" / "site-packages" / "aisan"
    fake.mkdir(parents=True)
    monkeypatch.setattr(launch_mod, "__file__", str(fake / "launch.py"))
    assert own_source_root() is None


def test_interpreter_roots_walks_a_symlink_chain(tmp_path):

    real = tmp_path / "cpython-3.12.12" / "bin"
    real.mkdir(parents=True)
    (real / "python3").write_text("#!/bin/true\n")
    alias = tmp_path / "cpython-3.12"
    alias.symlink_to(tmp_path / "cpython-3.12.12")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(alias / "bin" / "python3")
    roots = interpreter_roots(venv / "python")
    assert tmp_path / "venv" in roots
    assert tmp_path / "cpython-3.12.12" in roots


def test_an_empty_manifest_still_runs_the_payload(tmp_path):

    write_manifest(tmp_path, [])
    r = _launch(tmp_path, ["echo", "hello"])
    assert r.returncode == 0
    assert r.stdout.strip() == "hello"


def test_the_payloads_exit_status_is_the_launchers(tmp_path):

    write_manifest(tmp_path, [])
    assert _launch(tmp_path, ["sh", "-c", "exit 7"]).returncode == 7


def test_a_payload_killed_by_a_signal_reports_the_shell_convention(tmp_path):

    write_manifest(tmp_path, [])
    r = _launch(tmp_path, ["sh", "-c", "kill -TERM $$"])
    assert r.returncode == 128 + 15


async def test_the_launcher_serves_every_relay_in_the_manifest(tmp_path):
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
        binds=(Bind(Path(sys.executable).parent.parent, RO),),
        tmpfs=(),
        env=(("PATH", "/usr/bin:/bin"),),
        egress=(_EchoBackend(),),
        unshare_net=True,
        limits=Limits(use_cgroup=False),
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

    assert "relay=from-the-host" in r.stdout
    assert seen == [b"ping"]

    assert "host=111" in r.stdout


def test_prepare_runtime_dir_clears_a_stale_client_env(monkeypatch, tmp_path):

    d = prepare_runtime_dir("box-l6")
    try:
        write_client_env(d, {"AISAN_DEAD": "1"})
        assert (d / CLIENT_ENV_NAME).exists()

        d2 = prepare_runtime_dir("box-l6")
        assert d2 == d
        assert not (d2 / CLIENT_ENV_NAME).exists()
    finally:
        cleanup_runtime_dir("box-l6")
