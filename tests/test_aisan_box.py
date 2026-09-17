# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import contextlib
import os
import shutil
import socket as _socket
import subprocess
import sys
from pathlib import Path

import pytest

from aisan import Box, PreflightError
from aisan import private as private_mod
from aisan import sandbox as sandbox_mod
from aisan.egress.base import Backend, BackendActivation
from aisan.hostproc import neutral_child
from aisan.runtime import (
    CLIENT_ENV_NAME,
    MANIFEST_NAME,
    cleanup_runtime_dir,
    prepare_runtime_dir,
    read_client_env,
    read_manifest,
    runtime_dir,
)
from aisan.sandbox import RO, RW, Bind, BindOver, BindSpec, Mount
from aisan.spec import BoxSpec, Limits

_SUN_PATH_MAX = 107
needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)


def _spec(root: Path, *, egress=(), **kw) -> BoxSpec:
    return BoxSpec(
        root=root,
        binds=kw.get("binds", ()),
        tmpfs=kw.get("tmpfs", ()),
        env=kw.get("env", ()),
        egress=tuple(egress),
        unshare_net=kw.get("unshare_net", bool(egress)),
        limits=kw.get("limits", Limits(use_cgroup=False)),
    )


class _FakeBackend(Backend):
    name = "fake"
    port = 8799

    def __init__(self, *, name: str = "fake", port: int = 8799, fail: str = "") -> None:
        self.name = name
        self.port = port
        self._fail = fail
        self.order: list[str] = []

    def client_env(self) -> dict[str, str]:
        return {"FAKE_ENDPOINT": f"http://127.0.0.1:{self.port}"}

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        return [BindOver(runtime_dir / f"{self.name}.conf", Path("/etc/fake.conf"))]

    def prepare(self, runtime_dir: Path) -> None:
        self.order.append("prepare")
        (runtime_dir / f"{self.name}.conf").write_text("x\n")

    async def preflight(self) -> None:
        self.order.append("preflight")
        if self._fail:
            raise PreflightError(self.name, self._fail, "run-this-to-fix")

    @contextlib.asynccontextmanager
    async def serve(self, runtime_dir: Path):
        self.order.append("serve")
        sock = self.socket_path(runtime_dir)
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.bind(str(sock))
        try:
            yield
        finally:
            s.close()
            sock.unlink(missing_ok=True)


class _SharedFakeBackend(_FakeBackend):
    supports_shared_net = True

    def shared_client_env_description(self) -> dict[str, str]:
        return {
            "FAKE_ENDPOINT": "http://127.0.0.1:(assigned at launch)",
            "FAKE_TOKEN": "<per-box proxy token>",
        }

    @contextlib.asynccontextmanager
    async def serve_shared(self, runtime_dir: Path):
        self.order.append("serve_shared")
        yield BackendActivation(
            23456,
            {
                "FAKE_ENDPOINT": "http://127.0.0.1:23456",
                "FAKE_TOKEN": "private-token",
            },
        )


def test_the_runtime_dir_is_per_box_and_derived_from_the_id():

    assert runtime_dir("a") != runtime_dir("b")
    assert runtime_dir("a") == runtime_dir("a")


def test_a_named_root_that_leaves_no_room_for_a_socket_is_refused(
    tmp_path, monkeypatch
):
    import aisan.private as private_mod

    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", tmp_path / ("r" * 90))
    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        _FakeBackend().socket_path(runtime_dir("box"))


def test_a_caller_named_backend_cannot_overrun_the_socket_path(tmp_path):

    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        _FakeBackend(name="a" * 64).socket_path(runtime_dir("box"))


def test_socket_path_fits_in_sun_path(tmp_path):
    deep = tmp_path / ("d" * 90) / ("e" * 90) / ("f" * 90) / "control"
    box_id = str(deep / ("v8-a817808e4ad2-src-debug-debug-scopes-cc-1434" * 3))
    assert len(box_id) > _SUN_PATH_MAX
    sock = _FakeBackend().socket_path(runtime_dir(box_id))
    assert len(str(sock)) <= _SUN_PATH_MAX, sock

    prepare_runtime_dir(box_id)
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.bind(str(sock))
    finally:
        s.close()
        sock.unlink(missing_ok=True)
        cleanup_runtime_dir(box_id)


def test_the_runtime_dir_is_private_to_this_user(tmp_path):

    box_id = str(tmp_path / "job1")
    d = prepare_runtime_dir(box_id)
    try:
        assert d.stat().st_mode & 0o777 == 0o700
    finally:
        cleanup_runtime_dir(box_id)


def test_runtime_dir_rejects_symlink_and_permissive_existing_path(tmp_path):
    box_id = str(tmp_path / "job-hardening")
    d = runtime_dir(box_id)
    d.symlink_to(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not a directory"):
            prepare_runtime_dir(box_id)
    finally:
        d.unlink(missing_ok=True)

    d.mkdir()
    d.chmod(0o755)  # mkdir's mode is trimmed by the umask
    try:
        with pytest.raises(PermissionError, match="too permissive"):
            prepare_runtime_dir(box_id)
    finally:
        d.rmdir()


def test_box_rejects_unsafe_or_duplicate_backend_names(tmp_path):
    unsafe = _FakeBackend(name="../escape")
    with pytest.raises(ValueError, match="safe unique socket-file"):
        _spec(tmp_path, egress=(unsafe,))

    first = _FakeBackend(name="same", port=8799)
    second = _FakeBackend(name="same", port=8800)
    with pytest.raises(ValueError, match="names collide"):
        _spec(tmp_path, egress=(first, second))


def test_cleanup_leaves_a_non_empty_directory_alone(tmp_path):

    box_id = str(tmp_path / "job1")
    prepare_runtime_dir(box_id)
    stray = runtime_dir(box_id) / "still-here"
    stray.write_text("x")
    cleanup_runtime_dir(box_id)
    assert runtime_dir(box_id).is_dir()
    stray.unlink()
    cleanup_runtime_dir(box_id)
    assert not runtime_dir(box_id).exists()


async def test_the_bracket_preflights_then_stages_then_serves(tmp_path):

    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert b.order == ["preflight", "prepare", "serve"]
        assert (box.runtime_dir / MANIFEST_NAME).exists()


async def test_a_preflight_failure_starts_nothing_and_leaves_nothing(tmp_path):

    good, bad = (
        _FakeBackend(name="good", port=8801),
        _FakeBackend(name="bad", port=8802, fail="no credential"),
    )
    box = Box(_spec(tmp_path, egress=(good, bad)), box_id=str(tmp_path / "j"))
    with pytest.raises(PreflightError) as exc:
        async with box:
            pass
    assert exc.value.fix == "run-this-to-fix"
    assert good.order == ["preflight"]
    assert not runtime_dir(box.box_id).exists()


async def test_the_bracket_removes_the_runtime_dir_on_the_way_out(tmp_path):

    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert box.runtime_dir.exists()
    assert not box.runtime_dir.exists()


async def test_the_manifest_names_every_backends_socket_and_port(tmp_path):

    a, b = _FakeBackend(name="a", port=8801), _FakeBackend(name="b", port=8802)
    box = Box(_spec(tmp_path, egress=(a, b)), box_id=str(tmp_path / "j"))
    async with box:
        entries = read_manifest(box.runtime_dir)
    assert [e["name"] for e in entries] == ["a", "b"]
    assert [e["port"] for e in entries] == [8801, 8802]
    assert [Path(str(e["socket"])).parent for e in entries] == [box.runtime_dir] * 2


async def test_shared_network_activation_uses_a_private_environment_file(tmp_path):
    backend = _SharedFakeBackend()
    box = Box(
        _spec(tmp_path, egress=(backend,), unshare_net=False),
        box_id=str(tmp_path / "shared"),
    )
    async with box:
        path = box.runtime_dir / CLIENT_ENV_NAME
        assert path.stat().st_mode & 0o777 == 0o600
        assert read_client_env(box.runtime_dir)["FAKE_TOKEN"] == "private-token"
        assert not (box.runtime_dir / MANIFEST_NAME).exists()
        assert "FAKE_TOKEN" not in box.env
        assert "private-token" not in "\0".join(box.command(["true"]))
        assert box.activation(backend).port == 23456
    assert not box.runtime_dir.exists()


@needs_bwrap
async def test_shared_client_environment_reaches_a_real_box_without_argv_leak(
    tmp_path, monkeypatch
):
    class _NoFilesShared(_SharedFakeBackend):
        def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
            return []

        def prepare(self, runtime_dir: Path) -> None:
            self.order.append("prepare")

    private = tmp_path / "private"
    sibling = private / "another-box" / "client-env.json"
    private.mkdir(mode=0o700)
    sibling.parent.mkdir(mode=0o700)
    sibling.write_text("must stay hidden\n")
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    root = tmp_path
    backend = _NoFilesShared()
    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    port = listener.getsockname()[1]
    box = Box(
        _spec(root, egress=(backend,), unshare_net=False),
        box_id=str(tmp_path / "shared-real"),
    )
    async with box:
        argv = box.command(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, socket; "
                    f"assert not pathlib.Path({str(sibling)!r}).exists(); "
                    f"s = socket.create_connection(('127.0.0.1', {port})); "
                    "s.sendall(b'from-box'); print(os.environ['FAKE_TOKEN'])"
                ),
            ]
        )
        assert "private-token" not in "\0".join(argv)
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **box.env},
        )
    assert result.returncode == 0, result.stderr
    connection, _ = listener.accept()
    try:
        assert connection.recv(64) == b"from-box"
    finally:
        connection.close()
        listener.close()
    assert result.stdout.strip() == "private-token"


def test_shared_network_explain_markers_are_not_live_activation_values(tmp_path):
    backend = _SharedFakeBackend()
    box = Box(
        _spec(tmp_path, egress=(backend,), unshare_net=False),
        box_id=str(tmp_path / "shared-explain"),
    )
    from aisan.explain import explain

    with box.staged():
        report = explain(box)
    assert "shared host namespace" in report
    assert "authenticated host-loopback TCP" in report
    assert "<per-box proxy token>" in report
    assert "private-token" not in report


async def test_the_endpoint_the_client_reads_is_the_port_the_relay_serves(tmp_path):

    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        served = {e["port"] for e in read_manifest(box.runtime_dir)}
    assert box.env["FAKE_ENDPOINT"] == f"http://127.0.0.1:{b.port}"
    assert served == {b.port}


async def test_wrapper_refuses_a_box_whose_runtime_dir_is_not_mounted(tmp_path):

    class _NoFiles(_FakeBackend):
        def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
            return []

    box = Box(_spec(tmp_path, egress=(_NoFiles(),)), box_id=str(tmp_path / "j"))
    with pytest.raises(ValueError, match="runtime dir"):
        box.wrapper()


def test_egress_without_a_private_loopback_is_refused(tmp_path):

    with pytest.raises(ValueError, match="unshare_net"):
        _spec(tmp_path, egress=(_FakeBackend(),), unshare_net=False)


def test_two_backends_on_one_port_are_refused(tmp_path):

    a, b = _FakeBackend(name="a", port=8800), _FakeBackend(name="b", port=8800)
    with pytest.raises(ValueError, match="8800"):
        _spec(tmp_path, egress=(a, b))


def test_shared_backends_do_not_reserve_their_isolated_ports(tmp_path):
    a = _SharedFakeBackend(name="a", port=8800)
    b = _SharedFakeBackend(name="b", port=8800)
    spec = _spec(tmp_path, egress=(a, b), unshare_net=False)
    assert spec.egress == (a, b)


def test_a_spec_bind_containing_a_credential_is_refused(tmp_path):

    b = _FakeBackend()
    b.credentials = (tmp_path / "creds" / "k.json",)
    creds_dir = tmp_path / "creds"
    creds_dir.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = _spec(wt, egress=(b,), binds=(Bind(creds_dir, RO),))
    box = Box(spec, box_id=str(tmp_path / "j"))

    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_a_root_containing_a_credential_is_refused(tmp_path):

    b = _FakeBackend()
    b.credentials = (tmp_path / "home" / ".secret" / "key.json",)
    root = tmp_path / "home"
    root.mkdir()
    box = Box(_spec(root, egress=(b,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def _home_under_a_system_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    system_root = tmp_path / "usr"
    home = system_root / "local" / "google" / "home" / "u"
    worktree = home / "src" / "wt"
    (home / ".config" / "chrome_infra").mkdir(parents=True)
    worktree.mkdir(parents=True)
    return system_root, home, worktree


def test_binding_a_system_root_that_holds_home_is_not_an_exposure(
    tmp_path, monkeypatch
):
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_BINDS", ((system_root, system_root),))
    b = _FakeBackend()
    b.credentials = (home / ".config" / "chrome_infra",)
    spec = _spec(
        worktree,
        egress=(b,),
        binds=(Bind(system_root, RO),),
        tmpfs=((str(home), 1 << 20),),
    )
    box = Box(spec, box_id=str(tmp_path / "j"))
    with box.staged():
        mounts = box.mounts()
        box.wrapper()
    assert not [m for m in mounts if m.dst == system_root]


def test_a_credential_reachable_through_the_system_surface_is_refused(
    tmp_path, monkeypatch
):
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_BINDS", ((system_root, system_root),))
    b = _FakeBackend()
    b.credentials = (home / ".config" / "chrome_infra",)
    box = Box(_spec(worktree, egress=(b,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_a_home_tmpfs_does_not_excuse_a_spec_bind_of_the_credential(
    tmp_path, monkeypatch
):
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_BINDS", ((system_root, system_root),))
    cred = home / ".config" / "chrome_infra"
    b = _FakeBackend()
    b.credentials = (cred,)
    spec = _spec(
        worktree, egress=(b,), binds=(Bind(cred, RO),), tmpfs=((str(home), 1 << 20),)
    )
    box = Box(spec, box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_binding_one_file_inside_a_credential_store_is_refused(tmp_path, monkeypatch):
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_BINDS", ((system_root, system_root),))
    store = home / ".config" / "chrome_infra"
    token = store / "luci_context"
    token.write_text("{}\n")
    b = _FakeBackend()
    b.credentials = (store,)
    spec = _spec(
        worktree, egress=(b,), binds=(Bind(token, RO),), tmpfs=((str(home), 1 << 20),)
    )
    box = Box(spec, box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_the_check_models_the_system_surface_by_source_not_by_name(
    tmp_path, monkeypatch
):
    host = tmp_path / "hostsys"
    (host / "creds").mkdir(parents=True)
    box_side = tmp_path / "boxsys"
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_BINDS", ((host, box_side),))
    wt = tmp_path / "wt"
    wt.mkdir()
    b = _FakeBackend()
    b.credentials = (host / "creds" / "k.json",)
    box = Box(_spec(wt, egress=(b,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_masking_follows_the_destination_not_the_source(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    token = store / "token"
    token.write_text("t\n")
    d = tmp_path / "d"
    d.mkdir()
    (d / "t").touch()
    wt = tmp_path / "wt"
    wt.mkdir()
    b = _FakeBackend()
    b.credentials = (token,)
    published = BindOver(token, d / "t")

    box = Box(_spec(wt, egress=(b,), binds=(published,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()

    masked = Box(
        _spec(wt, egress=(b,), binds=(published, Bind(d, RO))),
        box_id=str(tmp_path / "j"),
    )
    with masked.staged():
        masked.wrapper()


def test_a_bind_of_the_adc_store_is_refused_for_a_vertex_box(tmp_path, monkeypatch):
    from aisan.egress.vertex import VertexBackend

    gcloud = tmp_path / "gcloud"
    gcloud.mkdir()
    (gcloud / "application_default_credentials.json").write_text("{}")
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(gcloud))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    wt = tmp_path / "wt"
    wt.mkdir()
    b = VertexBackend(rpm=1, project="p", location="l", models=("m",), fetch=object())
    spec = _spec(wt, egress=(b,), binds=(Bind(gcloud, RO),))
    box = Box(spec, box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the vertex"):
        box.wrapper()


def test_the_vertex_backend_points_both_sdks_at_its_proxy():
    from aisan.egress.vertex import PORT, VertexBackend

    b = VertexBackend(
        rpm=1,
        project="p",
        location="global",
        models=("gemini-3",),
        anthropic_models=("claude-4",),
        fetch=object(),
    )
    assert b.client_env() == {
        "AISAN_VERTEX_PROXY_ENDPOINT": f"http://127.0.0.1:{PORT}",
        "ANTHROPIC_VERTEX_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
    }

    assert (
        VertexBackend(
            rpm=1, project="p", location="global", models=("g",), fetch=object()
        ).client_env()
        == b.client_env()
    )


def test_a_root_beside_a_credential_is_allowed(tmp_path):
    b = _FakeBackend()
    b.credentials = (tmp_path / "credentials" / "key.json",)
    root = tmp_path / "worktree"
    root.mkdir()
    box = Box(_spec(root, egress=(b,)), box_id=str(tmp_path / "j"))

    with box.staged():
        box.wrapper()


def test_every_box_seals_the_private_host_root_before_its_runtime(
    tmp_path, monkeypatch
):
    private = tmp_path / "private"
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    _launcher(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()
    box = Box(_spec(wt, egress=(_FakeBackend(),)), box_id="private-order")

    with box.staged():
        mounts = box.mounts()

    seal = mounts.index(Mount("tmpfs", private))
    own_runtime = mounts.index(Mount("ro", box.runtime_dir, box.runtime_dir))
    assert seal < own_runtime
    assert Mount("seal-ro", private) == mounts[-1]


def test_private_root_alias_is_refused_even_when_own_runtime_is_allowed(
    tmp_path, monkeypatch
):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    _launcher(monkeypatch)
    alias = tmp_path / "alias"
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = _spec(
        wt,
        egress=(_FakeBackend(),),
        binds=(BindOver(private, alias),),
    )
    box = Box(spec, box_id="private-alias")

    with box.staged(), pytest.raises(ValueError, match="private host-control root"):
        box.wrapper()


def test_private_root_at_its_normal_name_is_hidden_by_the_seal(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    _launcher(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()
    box = Box(
        _spec(wt, binds=(Bind(private, RW),)),
        box_id="private-canonical",
    )

    box.wrapper()


def test_runtime_paths_ignore_ambient_tmpdir(tmp_path, monkeypatch):
    private = tmp_path / "private"
    hostile_tmp = tmp_path / "box-writable"
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    monkeypatch.setenv("TMPDIR", str(hostile_tmp))

    path = runtime_dir("not-in-ambient-tmpdir")
    assert path.parent == private
    assert not path.is_relative_to(hostile_tmp)


@needs_bwrap
def test_a_live_host_child_directory_is_absent_from_a_real_box(tmp_path, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    box = Box(_spec(tmp_path), box_id="live-host-child")

    with neutral_child() as child:
        marker = child.cwd / "agent-controlled.py"
        marker.write_text("must not be visible in the box\n")
        script = (
            f"from pathlib import Path; assert not Path({str(child.cwd)!r}).exists()"
        )
        result = subprocess.run(
            box.command([sys.executable, "-c", script]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    assert result.returncode == 0, result.stderr


async def test_backend_binds_and_launcher_binds_do_not_trip_the_guard(tmp_path):

    b = _FakeBackend()
    b.credentials = (Path("/definitely/not/mounted/anywhere"),)
    wt = tmp_path / "wt"
    wt.mkdir()
    box = Box(_spec(wt, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        box.command(["true"])


async def test_refused_surfaces_the_first_backend_refusal(tmp_path):

    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert box.refused is None
        b.__class__.refused = property(lambda self: RuntimeError("no token"))
        try:
            assert isinstance(box.refused, RuntimeError)
        finally:
            del b.__class__.refused


def test_the_argv_never_asks_bwrap_to_be_pid_1(tmp_path):
    box = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    assert "--as-pid-1" not in box.wrapper()
    assert "--as-pid-1" not in box.command(["true"])


async def test_command_composes_wrapper_then_launcher_then_payload(tmp_path):

    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        argv = box.command(["echo", "hi"])
        wrapper, prefix = box.wrapper(), box.launch_prefix()
    assert argv == [*wrapper, *prefix, "echo", "hi"]

    assert prefix[-2:] == [str(box.runtime_dir), "--"]

    assert prefix[0].startswith("/") and prefix[1] == "-m"
    assert prefix[2].endswith(".launch")


async def test_command_returns_argv_that_no_shell_reinterprets(tmp_path):
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    hostile = ["sh", "-c", "$(id -u) > /tmp/pwned; *; `whoami`", "a b\tc", "$HOME"]
    async with box:
        argv = box.command(hostile)
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    assert argv[-len(hostile) :] == hostile, "the payload must arrive unmangled"

    prefix = argv[: -len(hostile)]
    assert not any(a.endswith("sh") or a == "-c" for a in prefix), prefix


def test_a_box_without_egress_has_no_launcher(tmp_path):

    box = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    assert box.launch_prefix() == []
    assert box.command(["/bin/sh"]) == [*box.wrapper(), "/bin/sh"]


async def test_egress_adds_only_its_own_plumbing_to_the_mounts(tmp_path):
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    plain = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    async with box:
        added = {Path(m.dst) for m in box.mounts()} - {
            Path(m.dst) for m in plain.mounts()
        }
    assert added == {Path("/etc/fake.conf"), box.runtime_dir}


def test_the_box_env_is_the_spec_plus_the_backends_client_vars(tmp_path):

    b = _FakeBackend()
    spec = _spec(tmp_path, egress=(b,), env=(("HOME", "/h"),))
    box = Box(spec, box_id=str(tmp_path / "j"))
    assert box.env == {"HOME": "/h", "FAKE_ENDPOINT": f"http://127.0.0.1:{b.port}"}


def _launcher(monkeypatch, *binds):
    from aisan import launch

    monkeypatch.setattr(launch, "launcher_binds", lambda: list(binds))
    return launch


def test_a_writable_root_wins_over_the_launcher_defaults(
    tmp_path, tmp_path_factory, monkeypatch
):
    (tmp_path / "src").mkdir()
    (tmp_path / ".venv").mkdir()
    outside = tmp_path_factory.mktemp("uv") / "python"
    outside.mkdir()
    _launcher(
        monkeypatch,
        Bind(tmp_path / "src", RO),
        Bind(tmp_path / ".venv", RO),
        Bind(outside, RO),
    )

    mounts = Box(_spec(tmp_path), box_id=str(tmp_path / "j")).mounts()
    ro = {m.dst for m in mounts if m.op == "ro"}

    assert tmp_path / "src" not in ro
    assert tmp_path / ".venv" not in ro

    assert outside in ro


def test_a_writable_spec_bind_covers_a_launcher_path_the_root_does_not(
    tmp_path, tmp_path_factory, monkeypatch
):
    tools = tmp_path_factory.mktemp("tools")
    (tools / "venv").mkdir()
    _launcher(monkeypatch, Bind(tools / "venv", RO))

    spec = _spec(tmp_path, binds=(Bind(tools, RW),))
    mounts = Box(spec, box_id=str(tmp_path / "j")).mounts()

    assert tools / "venv" not in {m.dst for m in mounts if m.op == "ro"}
    assert tools in {m.dst for m in mounts if m.op == "rw"}


def test_a_writable_symlink_does_not_cover_its_target(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    _launcher(monkeypatch, Bind(target, RO))

    spec = _spec(tmp_path / "root", binds=(Bind(alias, RW),))
    spec.root.mkdir()
    mounts = Box(spec, box_id=str(tmp_path / "j")).mounts()

    assert target in {m.dst for m in mounts if m.op == "ro"}


def test_consumer_ro_pins_inside_the_root_are_untouched(tmp_path, monkeypatch):
    _launcher(monkeypatch)
    pin = tmp_path / ".git"
    pin.mkdir()

    spec = _spec(tmp_path, binds=(Bind(pin, RO),))
    mounts = Box(spec, box_id=str(tmp_path / "j")).mounts()

    ops = [(m.op, m.dst) for m in mounts]
    assert ("ro", pin) in ops

    assert ops.index(("rw", tmp_path)) < ops.index(("ro", pin))


@needs_bwrap
async def test_the_launcher_interpreter_imports_aisan_inside_a_real_box(tmp_path):
    # The launcher binds are a prediction about the box. Only a launch confirms
    # it: a home tmpfs can mask a bound ancestor, so a bind list that looks
    # complete can still leave the package unimportable.
    root = tmp_path / "work"
    root.mkdir()
    spec = _spec(root, tmpfs=((str(Path.home()), 64 << 20),))
    box = Box(spec, box_id=str(tmp_path / "launcher-import"))
    async with box:
        argv = box.command(
            [
                sys.executable,
                "-c",
                "import aisan, aisan.launch; print(aisan.launch.__file__)",
            ]
        )
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("launch.py")
