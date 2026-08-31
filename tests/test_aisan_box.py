# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The one bracket: runtime dir, backends, manifest, argv.

What was four nested lifecycle brackets in four places is now `async with Box`,
so the properties that used to be spread across four test files are asserted
here against the thing that owns them. Three of them are load-bearing enough to
name:

- the runtime dir's LENGTH (a 129-byte socket path failed at bind() with
  sun_path) and its mode (0700: these sockets are unauthenticated capabilities
  onto the daemon's credentials);
- the ORDER inside `__aenter__` -- preflight before anything is written, every
  bind-over source on disk before `wrapper()` resolves;
- that the argv never carries `--as-pid-1`, whose name is the inverse of what it
  does.
"""

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
from aisan import sandbox as sandbox_mod
from aisan.egress.base import Backend, BackendActivation
from aisan.runtime import (
    CLIENT_ENV_NAME,
    MANIFEST_NAME,
    cleanup_runtime_dir,
    prepare_runtime_dir,
    read_client_env,
    read_manifest,
    runtime_dir,
)
from aisan.sandbox import RO, RW, Bind, BindOver, BindSpec
from aisan.spec import BoxSpec, Limits

# sizeof(struct sockaddr_un.sun_path) on Linux, minus the NUL.
_SUN_PATH_MAX = 107
needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)


def _spec(root: Path, *, egress=(), **kw) -> BoxSpec:
    """A minimal spec. Non-defaulting by design, so a test states every field --
    which is the property under test as much as anything else here: a reader of
    a call site knows what is mounted without simulating default resolution."""
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
    """A backend that records its lifecycle and needs no credential."""

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


# -- the runtime directory ----------------------------------------------------


def test_the_runtime_dir_is_per_box_and_derived_from_the_id():
    # Per box by construction: two boxes must get two directories, because the
    # sockets in there are unauthenticated capabilities and a shared directory
    # would splice one box's client to the other box's proxy. Derived (not
    # passed) so the host half and the launcher name the same path without one
    # telling the other.
    assert runtime_dir("a") != runtime_dir("b")
    assert runtime_dir("a") == runtime_dir("a")


def test_socket_path_fits_in_sun_path(tmp_path):
    """The bug this shape exists for: bind() died "AF_UNIX path too long" on a
    observed job at 129 bytes, killing it outright.

    The old path's two terms were both unbounded and neither was ours to shorten
    -- control_root is the operator's, and job_id carries repo-hash-file-line so
    several findings on one commit stay distinct. So the assertion is that the
    length no longer depends on either: a pathological box_id must still fit.
    """
    deep = tmp_path / ("d" * 90) / ("e" * 90) / ("f" * 90) / "control"
    box_id = str(deep / ("v8-a817808e4ad2-src-debug-debug-scopes-cc-1434" * 3))
    assert len(box_id) > _SUN_PATH_MAX  # the input really is huge
    sock = _FakeBackend().socket_path(runtime_dir(box_id))
    assert len(str(sock)) <= _SUN_PATH_MAX, sock
    # Not merely short: actually bindable, which is the property that failed.
    prepare_runtime_dir(box_id)
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.bind(str(sock))
    finally:
        s.close()
        sock.unlink(missing_ok=True)
        cleanup_runtime_dir(box_id)


def test_the_runtime_dir_is_private_to_this_user(tmp_path):
    # The system temp dir is world-writable on a shared host, and behind these
    # sockets sit the daemon's credentials: anyone who can connect gets model
    # calls and RBE calls on somebody else's identity. The mode is the only
    # thing standing in the way.
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

    d.mkdir(mode=0o755)
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
    # The directory is ours (its name is this box's digest), but a non-empty one
    # means something we do not own is in there, and deleting a stranger's file
    # is worse than leaving litter.
    box_id = str(tmp_path / "job1")
    prepare_runtime_dir(box_id)
    stray = runtime_dir(box_id) / "still-here"
    stray.write_text("x")
    cleanup_runtime_dir(box_id)
    assert runtime_dir(box_id).is_dir()
    stray.unlink()
    cleanup_runtime_dir(box_id)
    assert not runtime_dir(box_id).exists()


# -- the bracket --------------------------------------------------------------


async def test_the_bracket_preflights_then_stages_then_serves(tmp_path):
    # The order is a constraint, not a preference. Preflight first because luci
    # reauth is interactive and the box owns the TTY once bwrap is up; prepare
    # before wrapper() because a bind-over source that does not exist fails the
    # whole profile; serve last because that is when the socket appears.
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert b.order == ["preflight", "prepare", "serve"]
        assert (box.runtime_dir / MANIFEST_NAME).exists()


async def test_a_preflight_failure_starts_nothing_and_leaves_nothing(tmp_path):
    # A backend that cannot mint must fail BEFORE bwrap and before the second
    # backend is started: a box whose second backend has no credential is just
    # as dead as one whose first does not, and starting half of them only makes
    # the teardown ambiguous.
    good, bad = (
        _FakeBackend(name="good", port=8801),
        _FakeBackend(name="bad", port=8802, fail="no credential"),
    )
    box = Box(_spec(tmp_path, egress=(good, bad)), box_id=str(tmp_path / "j"))
    with pytest.raises(PreflightError) as exc:
        async with box:
            pass
    assert exc.value.fix == "run-this-to-fix"  # the fix command is the payload
    assert good.order == ["preflight"]  # started nothing
    assert not runtime_dir(box.box_id).exists()  # and wrote nothing


async def test_the_bracket_removes_the_runtime_dir_on_the_way_out(tmp_path):
    # Every file in there is ours to remove, and one left behind means the rmdir
    # never succeeds -- which is a directory leaked per box, under a system temp
    # dir nobody is watching.
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert box.runtime_dir.exists()
    assert not box.runtime_dir.exists()


async def test_the_manifest_names_every_backends_socket_and_port(tmp_path):
    # The launcher's whole input. A file rather than argv or env because N
    # backends means N sockets, and every other encoding either fixes N at the
    # shape of a command line or reinvents a list format inside a string.
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
async def test_shared_client_environment_reaches_a_real_box_without_argv_leak(tmp_path):
    class _NoFilesShared(_SharedFakeBackend):
        def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
            return []

        def prepare(self, runtime_dir: Path) -> None:
            self.order.append("prepare")

    root = tmp_path / "root"
    root.mkdir()
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
                    "import os, socket; "
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
    # The drift this closes: the code that stamps the endpoint and the code that
    # serves it used to be different code, and a mismatch presents as every call
    # failing to connect from inside a box that otherwise looks correct. One
    # `port` attribute now feeds both, so the test asserts they agree rather
    # than asserting a literal that could match neither.
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        served = {e["port"] for e in read_manifest(box.runtime_dir)}
    assert box.env["FAKE_ENDPOINT"] == f"http://127.0.0.1:{b.port}"
    assert served == {b.port}


async def test_wrapper_refuses_a_box_whose_runtime_dir_is_not_mounted(tmp_path):
    # The bind is optional (a box being explained has no runtime dir yet), so a
    # wrapper() called outside the bracket silently drops it and produces a box
    # whose relays have nothing to splice to. That failure presents as every
    # model call and every RBE call refusing to connect from inside a box that
    # otherwise looks entirely correct -- a long way from the missing
    # `async with` that caused it.
    # A backend with no box_binds, so the refusal under test is the one about
    # the runtime dir and not the (equally correct, but different) refusal of a
    # bind-over whose source has not been written yet.
    class _NoFiles(_FakeBackend):
        def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
            return []

    box = Box(_spec(tmp_path, egress=(_NoFiles(),)), box_id=str(tmp_path / "j"))
    with pytest.raises(ValueError, match="runtime dir"):
        box.wrapper()


def test_egress_without_a_private_loopback_is_refused(tmp_path):
    # The in-box ports are FIXED (8711, 8712), which is only safe because the
    # box's loopback is private to it. Without unshare_net those ports are the
    # host's, so the relay either collides with whatever is already there or --
    # worse -- the box's client reaches a listener that is not ours.
    with pytest.raises(ValueError, match="unshare_net"):
        _spec(tmp_path, egress=(_FakeBackend(),), unshare_net=False)


def test_two_backends_on_one_port_are_refused(tmp_path):
    # Same reasoning from the other side: one loopback, one listener per port.
    # Caught at construction because the alternative is one relay winning the
    # bind and the other's traffic going somewhere plausible but wrong.
    a, b = _FakeBackend(name="a", port=8800), _FakeBackend(name="b", port=8800)
    with pytest.raises(ValueError, match="8800"):
        _spec(tmp_path, egress=(a, b))


def test_shared_backends_do_not_reserve_their_isolated_ports(tmp_path):
    a = _SharedFakeBackend(name="a", port=8800)
    b = _SharedFakeBackend(name="b", port=8800)
    spec = _spec(tmp_path, egress=(a, b), unshare_net=False)
    assert spec.egress == (a, b)


def test_a_spec_bind_containing_a_credential_is_refused(tmp_path):
    # The credential-absence invariant, at the same choke point as the
    # mount-order leak check: a spec is caller input, and the one thing it
    # must not be able to state is "and also mount the credential". A
    # hand-composed caller who never touches userbinds.load cannot route
    # around this -- which is why the rule lives here and not only in the
    # loader.
    #
    # Containment, not equality: the credential's DIRECTORY exposes it as
    # surely as the file.
    b = _FakeBackend()
    b.credentials = (tmp_path / "creds" / "k.json",)
    creds_dir = tmp_path / "creds"
    creds_dir.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    spec = _spec(wt, egress=(b,), binds=(Bind(creds_dir, RO),))
    box = Box(spec, box_id=str(tmp_path / "j"))
    # Staged, because the rule is now decided over the RESOLVED mounts and
    # resolution needs the backend's bind-over source on disk.
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_a_root_containing_a_credential_is_refused(tmp_path):
    # Sandbox mounts the root rw outside the explicit bind list. It is still a
    # host mount and must pass through the same credential exposure guard.
    b = _FakeBackend()
    b.credentials = (tmp_path / "home" / ".secret" / "key.json",)
    root = tmp_path / "home"
    root.mkdir()
    box = Box(_spec(root, egress=(b,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def _home_under_a_system_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A host with $HOME inside a system root, and a credential under it.

    Some enterprise Linux images put home directories under `/usr/local`, which
    places a credential store at `~/.config/<name>` INSIDE `/usr` -- and
    `_SYSTEM_ARGS` binds `/usr` ro into every box unconditionally. That single
    fact is what the tests below pin down, and it is worth building rather than
    mocking: the layout has now broken this library three times (the RO-ancestor
    hoisting and the system-root dedup in `sandbox.resolve` are the other two).

    Returns (system root, home, worktree).
    """
    system_root = tmp_path / "usr"
    home = system_root / "local" / "google" / "home" / "u"
    worktree = home / "src" / "wt"
    (home / ".config" / "chrome_infra").mkdir(parents=True)
    worktree.mkdir(parents=True)
    return system_root, home, worktree


def test_binding_a_system_root_that_holds_home_is_not_an_exposure(
    tmp_path, monkeypatch
):
    """A bind of `/usr` on a host whose home is under it must build.

    It reads like an exposure and is not one: the box already has `/usr` ro from
    the fixed system surface, so `resolve()` drops the duplicate and the bind
    compiles to NO mount -- while the tmpfs over $HOME, mounted after that
    surface, is what actually keeps the credential out. The old rule asked
    whether a bind's source contained the credential, answered yes, and refused
    a box that mounts nothing extra; this is the prod failure that produced.

    `/usr` gets into a spec honestly: an interpreter root resolved for a system
    Python is exactly it (see `launch.interpreter_roots`).
    """
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_ROOTS", frozenset({system_root}))
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
    """The same layout with the $HOME tmpfs removed, which is a real leak.

    Nothing in the spec names the credential, so a check over bind sources
    cannot see this -- and on this host the credential is nonetheless readable
    in the box, through the unconditional `--ro-bind /usr /usr`. Asking the
    finished mount list instead is what turns the invariant from a claim about
    the spec into one about the box.
    """
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_ROOTS", frozenset({system_root}))
    b = _FakeBackend()
    b.credentials = (home / ".config" / "chrome_infra",)
    box = Box(_spec(worktree, egress=(b,)), box_id=str(tmp_path / "j"))
    with box.staged(), pytest.raises(ValueError, match="would expose the fake backend"):
        box.wrapper()


def test_a_home_tmpfs_does_not_excuse_a_spec_bind_of_the_credential(
    tmp_path, monkeypatch
):
    """Masking is not a loophole: spec binds land AFTER the tmpfs.

    The check reads the list in order, so a bind naming the credential is the
    last word on that path and the box is refused -- the same answer the old
    containment rule gave, kept for the case it was right about.
    """
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_ROOTS", frozenset({system_root}))
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
    """Overlap, not containment one way round.

    A credential is a store, and the bind that hands the box the token file
    under it names a path the store CONTAINS rather than one containing the
    store. The box can read the token either way, so the guard has to look both
    directions -- a rule reading only "does this bind's source hold the
    credential" answers no here.
    """
    system_root, home, worktree = _home_under_a_system_root(tmp_path)
    monkeypatch.setattr(sandbox_mod, "_SYSTEM_RO_ROOTS", frozenset({system_root}))
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


def test_a_root_beside_a_credential_is_allowed(tmp_path):
    b = _FakeBackend()
    b.credentials = (tmp_path / "credentials" / "key.json",)
    root = tmp_path / "worktree"
    root.mkdir()
    box = Box(_spec(root, egress=(b,)), box_id=str(tmp_path / "j"))
    # Staging supplies the optional runtime bind needed by wrapper().
    with box.staged():
        box.wrapper()


async def test_backend_binds_and_launcher_binds_do_not_trip_the_guard(tmp_path):
    # The guard is over the whole finished mount list, library-authored entries
    # included -- the runtime dir, aisan's launcher binds, a backend's own
    # bind-overs -- because what it decides is whether the BOX can read the
    # file, and the box cannot tell who wrote the mount. They pass because none
    # of them publishes a credential: the launcher binds (the interpreter under
    # ~/.local) routinely sit BESIDE one without containing it. Asserted by a
    # full bracket that does not raise while the backend declares a credential
    # that nothing mounts.
    b = _FakeBackend()
    b.credentials = (Path("/definitely/not/mounted/anywhere"),)
    wt = tmp_path / "wt"
    wt.mkdir()
    box = Box(_spec(wt, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        box.command(["true"])


async def test_refused_surfaces_the_first_backend_refusal(tmp_path):
    # "Did this build fail for lack of a credential" has a caller: the offline
    # build fallback. Its only alternative is matching someone else's stderr for
    # phrases like "code = Unauthenticated", which fails silently in the worst
    # direction the moment Google rewords a message.
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        assert box.refused is None
        b.__class__.refused = property(lambda self: RuntimeError("no token"))
        try:
            assert isinstance(box.refused, RuntimeError)
        finally:
            del b.__class__.refused


# -- the argv -----------------------------------------------------------------


def test_the_argv_never_asks_bwrap_to_be_pid_1(tmp_path):
    """`--as-pid-1` means "do NOT install a reaper" -- the inverse of its name.

    bwrap's own pid 1 exists to reap: the box unshares its pid namespace, and in
    a pid namespace an orphan is reparented to pid 1, so without a reaper every
    process the agent's build leaves behind (siso workers, a ninja that outlived
    its parent, a d8 that hung) becomes a permanent zombie in that namespace.
    Passing the flag hands pid 1 to the payload, which has no idea it is
    supposed to reap. The name reads like "run the payload as pid 1", which is
    exactly what it does and exactly the thing that must not happen -- hence a
    test rather than a comment, on every argv this app can produce.
    """
    box = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    assert "--as-pid-1" not in box.wrapper()
    assert "--as-pid-1" not in box.command(["true"])


async def test_command_composes_wrapper_then_launcher_then_payload(tmp_path):
    # The one call site that knows these three compose in this order. The
    # launcher must sit INSIDE the wrapper (it serves the box's loopback, which
    # only exists inside the netns) and outside the payload (it spawns it).
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    async with box:
        argv = box.command(["echo", "hi"])
        wrapper, prefix = box.wrapper(), box.launch_prefix()
    assert argv == [*wrapper, *prefix, "echo", "hi"]
    # The runtime dir reaches the launcher as an argument, and `--` ends its
    # own options: everything after it is the payload, verbatim.
    assert prefix[-2:] == [str(box.runtime_dir), "--"]
    # The launcher is named by an absolute interpreter path and a module, never
    # by a PATH lookup: a box's PATH deliberately holds several venvs' bin dirs,
    # and picking the launcher by name there makes which aisan runs a function
    # of bind order.
    assert prefix[0].startswith("/") and prefix[1] == "-m"
    assert prefix[2].endswith(".launch")


async def test_command_returns_argv_that_no_shell_reinterprets(tmp_path):
    """The argv invariant: a payload's bytes are arguments, never shell input.

    A caller handed a command STRING has to spawn it through a shell, and that
    shell is the host's -- so the payload gets word splitting, globbing and
    substitution applied outside the box, before bwrap has run anything. A
    `$(...)` in a payload would execute unconfined. Returning a list keeps every
    caller on `shell=False`, and this is the assertion that stops a future
    convenience overload from quietly reintroducing the string form.

    Written on a payload made of exactly the metacharacters that would prove it:
    each survives as one element, byte for byte.
    """
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    hostile = ["sh", "-c", "$(id -u) > /tmp/pwned; *; `whoami`", "a b\tc", "$HOME"]
    async with box:
        argv = box.command(hostile)
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    assert argv[-len(hostile) :] == hostile, "the payload must arrive unmangled"
    # Nothing joined the payload into a single word, and no element of the
    # wrapper or launcher is a shell either -- which is what a `-c` in the prefix
    # would mean.
    prefix = argv[: -len(hostile)]
    assert not any(a.endswith("sh") or a == "-c" for a in prefix), prefix


def test_a_box_without_egress_has_no_launcher(tmp_path):
    # No backends means no relays, which means no launcher -- and therefore a
    # box whose payload need not be Python at all. Worth a test because the
    # opposite (a launcher that is always prepended) would quietly make aisan's
    # interpreter a requirement for every consumer.
    box = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    assert box.launch_prefix() == []
    assert box.command(["/bin/sh"]) == [*box.wrapper(), "/bin/sh"]


async def test_egress_adds_only_its_own_plumbing_to_the_mounts(tmp_path):
    """Turning egress on must add the plumbing and nothing else.

    Asserted as a DIFF against the same spec without backends, rather than as
    the absence of a named field: a credential bind added under any other name
    shows up here, which is the property that actually matters. Two things, and
    only two: the backend's own bind-over destination and the runtime dir the
    sockets live in. Aisan's own runtime is NOT in the diff, because it is in
    both boxes -- see `_sandbox`, where it is unconditional so that adding a
    backend to a working box cannot change what is mounted in it.
    """
    b = _FakeBackend()
    box = Box(_spec(tmp_path, egress=(b,)), box_id=str(tmp_path / "j"))
    plain = Box(_spec(tmp_path), box_id=str(tmp_path / "j"))
    async with box:
        added = {Path(m.dst) for m in box.mounts()} - {
            Path(m.dst) for m in plain.mounts()
        }
    assert added == {Path("/etc/fake.conf"), box.runtime_dir}


def test_the_box_env_is_the_spec_plus_the_backends_client_vars(tmp_path):
    # The one thing a spec cannot state itself: the values name a port table
    # only the backends know. Computed rather than stored so there is no second
    # copy to fall out of date with the spec it came from.
    b = _FakeBackend()
    spec = _spec(tmp_path, egress=(b,), env=(("HOME", "/h"),))
    box = Box(spec, box_id=str(tmp_path / "j"))
    assert box.env == {"HOME": "/h", "FAKE_ENDPOINT": f"http://127.0.0.1:{b.port}"}


# --- launcher defaults vs explicit grants ------------------------------------
#
# The launcher's runtime binds are a REQUIREMENT (aisan importable, the
# interpreter exec'able) met at least privilege (RO), not a policy over the
# consumer's tree. The rule these tests pin: where the spec already grants a
# launcher path writable -- the root, or an RW bind -- the grant wins and the
# default is dropped. The box that forced it was rooted at aisan's own
# checkout, but the rule is stated for any root (or RW bind) that contains a
# launcher path: a vendored monorepo, a box rooted at a parent directory.


def _launcher(monkeypatch, *binds):
    """Stand in for the host-dependent launcher_binds(), deterministically."""
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

    # The root's grant stands: no RO shadow over either path it contains.
    assert tmp_path / "src" not in ro
    assert tmp_path / ".venv" not in ro
    # The requirement itself still holds where the root does NOT supply it:
    # the launcher path outside every grant is bound, read-only as before.
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

    # The alias and target are distinct destinations in the box. Supplying the
    # alias does not make the target visible under its own name.
    assert target in {m.dst for m in mounts if m.op == "ro"}


def test_consumer_ro_pins_inside_the_root_are_untouched(tmp_path, monkeypatch):
    """The elision disarms the LIBRARY's defaults only. A pin the spec itself
    wrote inside the root is the consumer overriding the consumer -- the .git
    dance -- and later-wins stays theirs to use."""
    _launcher(monkeypatch)  # none, so the pin is the only bind in play
    pin = tmp_path / ".git"
    pin.mkdir()

    spec = _spec(tmp_path, binds=(Bind(pin, RO),))
    mounts = Box(spec, box_id=str(tmp_path / "j")).mounts()

    ops = [(m.op, m.dst) for m in mounts]
    assert ("ro", pin) in ops
    # After the root, which is the whole point of a pin.
    assert ops.index(("rw", tmp_path)) < ops.index(("ro", pin))
