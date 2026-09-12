# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


import dataclasses
import shutil
import socket
import threading
from pathlib import Path

import pytest
from conftest import run_boxed

from aisan.sandbox import (
    RO,
    RW,
    Bind,
    BindOver,
    Mount,
    Overlay,
    Sandbox,
    Seal,
)
from aisan.spec import DEFANG_ENV


def _dest_at(argv: list[str], flag: str, dst: str) -> int:
    for i in range(len(argv)):
        if argv[i] != flag:
            continue

        n = 1 if flag in ("--tmpfs", "--remount-ro", "--tmp-overlay") else 2
        if argv[i + n] == dst:
            return i
    raise AssertionError(f"no {flag} at {dst} in argv")


def _ops(sb: Sandbox) -> list[tuple[str, str]]:
    return [(m.op, str(m.dst)) for m in sb.resolve()]


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "wt"
    casefile = tmp_path / "casefile"
    dep = tmp_path / "dep"
    for d in (root, casefile, dep):
        d.mkdir()
    return Sandbox(
        root=root,
        binds=(Bind(dep, RO, optional=True), Bind(casefile, RW, optional=True)),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        memory_max="1G",
        tasks_max=64,
        use_cgroup=False,
    )


def test_ai_agent_is_in_the_defang_set(profile):

    assert DEFANG_ENV.get("AI_AGENT")
    assert "--setenv AI_AGENT" in " ".join(profile.wrapper())


def test_the_profile_env_is_the_whole_env(tmp_path):

    root = tmp_path / "root"
    root.mkdir()
    argv = Sandbox(root=root, env=(("ONLY", "1"),), use_cgroup=False).wrapper()
    setenv = [argv[i + 1] for i, a in enumerate(argv) if a == "--setenv"]
    assert setenv == ["ONLY"]


def test_wrapper_shape(profile):
    argv = profile.wrapper()

    assert argv[0] == "bwrap"
    s = " ".join(argv)
    dep, casefile = profile.binds[0].path, profile.binds[1].path
    assert f"--bind {profile.root} {profile.root}" in s
    assert f"--ro-bind {dep} {dep}" in s
    assert f"--bind {casefile} {casefile}" in s
    assert "--cap-drop ALL" in s
    assert "--die-with-parent" in s
    assert "--clearenv" in s

    assert "--setenv PAGER cat" in s
    assert f"--setenv HOME {profile.env[0][1]}" in s
    assert f"--chdir {profile.root}" in s


def test_tmpfs_size_precedes_mount(profile):
    argv = profile.wrapper()
    i = argv.index("--tmpfs")
    assert argv[i - 2 : i + 2] == ["--size", str(1 << 20), "--tmpfs", "/tmp"]


def test_journal_socket_bound_when_present(profile, monkeypatch, tmp_path):

    import socket as _socket
    from pathlib import Path

    from aisan import sandbox as sb

    sock_path = tmp_path / "journal-stdout"
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.bind(str(sock_path))
    try:
        monkeypatch.setattr(sb, "_JOURNAL_STDOUT_SOCK", Path(sock_path))
        assert f"--ro-bind {sock_path} {sock_path}" in " ".join(profile.wrapper())

        missing = tmp_path / "nope"
        monkeypatch.setattr(sb, "_JOURNAL_STDOUT_SOCK", missing)
        assert str(missing) not in " ".join(profile.wrapper())
    finally:
        s.close()


def test_missing_optional_binds_skipped(profile):
    dep, casefile = profile.binds[0].path, profile.binds[1].path
    shutil.rmtree(dep)
    shutil.rmtree(casefile)
    s = " ".join(profile.wrapper())
    assert str(dep) not in s
    assert str(casefile) not in s


def test_unreachable_optional_binds_skipped(profile, monkeypatch):

    import errno

    gated = profile.binds[0].path
    casefile = profile.binds[1].path
    real_stat = Path.stat

    def fake_stat(self, *a, **kw):
        if self == gated:
            raise OSError(errno.ENOKEY, "Required key not available", str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", fake_stat)
    s = " ".join(profile.wrapper())
    assert str(gated) not in s

    assert f"--bind {profile.root} {profile.root}" in s
    assert f"--bind {casefile} {casefile}" in s


def test_unreachable_mandatory_bind_still_raises(profile, monkeypatch):

    import errno

    guard = profile.root / "guarded"
    guard.touch()
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(guard, RO),),
        use_cgroup=False,
    )
    real_stat = Path.stat

    def fake_stat(self, *a, **kw):
        if self == guard:
            raise OSError(errno.ENOKEY, "Required key not available", str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(OSError):
        boxed.wrapper()


def test_missing_root_raises(profile):
    shutil.rmtree(profile.root)
    with pytest.raises(FileNotFoundError):
        profile.wrapper()


def test_cgroup_prefix(profile, monkeypatch):
    boxed = Sandbox(
        root=profile.root,
        memory_max="1G",
        cpu_quota="200%",
        tasks_max=64,
        use_cgroup=True,
    )
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/systemd-run")
    argv = boxed.wrapper()
    assert argv[:5] == ["systemd-run", "--user", "--scope", "-q", "--collect"]
    s = " ".join(argv[: argv.index("--")])
    assert "-p MemoryMax=1G" in s
    assert "-p CPUQuota=200%" in s
    assert "-p TasksMax=64" in s

    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert boxed.wrapper()[0] == "bwrap"


needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)


@needs_bwrap
async def test_a_boxed_command_lands_in_the_root_with_the_stated_env(profile):
    (profile.root / "marker").write_text("x")
    out = await run_boxed("ls; echo home=$HOME; ls /tmp | wc -l", sandbox=profile)
    assert "marker" in out
    assert f"home={profile.env[0][1]}" in out
    assert "exit 0" in out


@needs_bwrap
async def test_a_ro_bind_is_enforced_by_the_kernel(profile):
    out = await run_boxed(
        f"touch {profile.binds[0].path}/probe 2>&1; ls /home/*/.* 2>&1 | head -1",
        sandbox=profile,
    )
    assert "Read-only file system" in out


def test_ro_ancestor_of_root_is_hoisted_above_it(profile, tmp_path):

    casefile = profile.binds[1].path
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(tmp_path, RO), Bind(casefile, RW, optional=True)),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    ro_at = _dest_at(argv, "--ro-bind", str(tmp_path))
    root_at = _dest_at(argv, "--bind", str(profile.root))
    assert ro_at < root_at

    assert _dest_at(argv, "--bind", str(casefile)) > ro_at


@needs_bwrap
async def test_root_stays_writable_under_ro_ancestor_end_to_end(profile, tmp_path):
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(tmp_path, RO),),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_boxed(
        f"touch probe && echo wrote; touch {tmp_path}/probe2 2>&1", sandbox=boxed
    )
    assert "wrote" in out
    assert "Read-only file system" in out


def _pinned_box(tmp_path, use_cgroup=False):
    root = tmp_path / "wt"
    gitdir = tmp_path / "gitdir"
    (gitdir / "hooks").mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n")
    (gitdir / "refs").mkdir()
    root.mkdir()
    return gitdir, Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(gitdir / "config", RO),
            Bind(gitdir / "hooks", RO),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=use_cgroup,
    )


def test_a_pin_written_after_the_rw_parent_is_emitted_after_it(tmp_path):

    gitdir, boxed = _pinned_box(tmp_path)
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(gitdir / "config")) > _dest_at(
        argv, "--bind", str(gitdir)
    )


def test_a_pin_written_before_the_rw_parent_does_not_win(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n")
    boxed = Sandbox(
        root=root,
        binds=(Bind(gitdir / "config", RO), Bind(gitdir, RW)),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(gitdir / "config")) < _dest_at(
        argv, "--bind", str(gitdir)
    )


@needs_bwrap
async def test_pin_over_an_rw_parent_is_read_only_end_to_end(tmp_path):
    gitdir, boxed = _pinned_box(tmp_path)
    out = await run_boxed(
        f"touch {gitdir}/refs/probe && echo wrote_refs; "
        f"echo x >> {gitdir}/config 2>&1; "
        f"echo x > {gitdir}/hooks/post-commit 2>&1",
        sandbox=boxed,
    )
    assert "wrote_refs" in out
    assert out.count("Read-only file system") >= 2


def test_wrapper_refuses_a_missing_mandatory_bind_source(tmp_path):

    gitdir, boxed = _pinned_box(tmp_path)
    (gitdir / "config").unlink()
    with pytest.raises(FileNotFoundError, match="config"):
        boxed.wrapper()


def _sealed_box(tmp_path, use_cgroup=False):
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    wts = gitdir / "worktrees"
    private = wts / "mine"
    private.mkdir(parents=True)

    (wts / "OTHERWT").mkdir()
    (wts / "OTHERWT" / "config.worktree").write_text("[core]\n")
    return (
        gitdir,
        private,
        Sandbox(
            root=root,
            binds=(Bind(gitdir, RW), Seal(wts), Bind(private, RW)),
            tmpfs=(("/tmp", 1 << 20),),
            env=(("HOME", str(tmp_path)),),
            use_cgroup=use_cgroup,
        ),
    )


def test_a_hole_through_a_seal_is_mounted_after_it(tmp_path):

    gitdir, private, boxed = _sealed_box(tmp_path)
    argv = boxed.wrapper()
    assert _dest_at(argv, "--bind", str(private)) > _dest_at(
        argv, "--tmpfs", str(gitdir / "worktrees")
    )


def test_a_seal_closes_only_after_every_later_mount(tmp_path):

    gitdir, _, boxed = _sealed_box(tmp_path)
    ops = _ops(boxed)
    assert ops[-1] == ("seal-ro", str(gitdir / "worktrees"))
    argv = boxed.wrapper()
    binds = [i for i, a in enumerate(argv) if a in ("--ro-bind", "--bind", "--tmpfs")]
    assert _dest_at(argv, "--remount-ro", str(gitdir / "worktrees")) > max(binds)


@needs_bwrap
async def test_seal_hides_siblings_and_forbids_creation_end_to_end(tmp_path):

    gitdir, private, boxed = _sealed_box(tmp_path)
    wts = gitdir / "worktrees"
    out = await run_boxed(
        f"ls {wts}; mkdir {wts}/planted 2>&1; "
        f"touch {private}/index.lock && echo wrote_private",
        sandbox=boxed,
    )
    assert "OTHERWT" not in out
    assert "mine" in out
    assert "Read-only file system" in out
    assert "wrote_private" in out

    assert (wts / "OTHERWT" / "config.worktree").read_text() == "[core]\n"
    assert not (wts / "planted").exists()


@needs_bwrap
async def test_a_pin_inside_a_hole_holds_when_written_after_it(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    wts = gitdir / "worktrees"
    private = wts / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    (private / "config.worktree").write_text("")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Seal(wts),
            Bind(private, RW),
            Bind(private / "commondir", RO),
            Bind(private / "config.worktree", RO),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_boxed(
        f"touch {private}/index.lock && echo wrote_lock; "
        f"echo x > {private}/config.worktree 2>&1; "
        f"echo x > {private}/commondir 2>&1",
        sandbox=boxed,
    )
    assert "wrote_lock" in out
    assert out.count("Read-only file system") >= 2


@needs_bwrap
async def test_a_pin_written_before_its_hole_is_re_opened_by_it(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    private = gitdir / "worktrees" / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(private / "commondir", RO),
            Bind(private, RW),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_boxed(
        f"echo x > {private}/commondir 2>&1 && echo REOPENED", sandbox=boxed
    )
    assert "REOPENED" in out
    assert "Read-only file system" not in out


def test_wrapper_refuses_a_missing_hole_source(tmp_path):

    _, private, boxed = _sealed_box(tmp_path)
    private.rmdir()
    with pytest.raises(FileNotFoundError, match="mine"):
        boxed.wrapper()


def test_seal_source_must_exist(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(root=root, binds=(Seal(tmp_path / "gone"),), use_cgroup=False)
    with pytest.raises(FileNotFoundError, match="seal source missing"):
        boxed.wrapper()
    assert not (tmp_path / "gone").exists()


def test_an_internal_seal_may_hide_a_path_before_it_exists(tmp_path):
    root = tmp_path / "wt"
    root.mkdir()
    absent = tmp_path / "reserved" / "future"
    boxed = Sandbox(
        root=root,
        binds=(Seal(absent, allow_missing=True),),
        use_cgroup=False,
    )

    mounts = boxed.resolve()
    assert Mount("tmpfs", absent) in mounts
    assert Mount("seal-ro", absent) in mounts
    assert not absent.exists()


def test_an_internal_seal_still_refuses_an_existing_non_directory(tmp_path):
    root = tmp_path / "wt"
    root.mkdir()
    invalid = tmp_path / "reserved"
    invalid.write_text("not a directory\n")
    boxed = Sandbox(
        root=root,
        binds=(Seal(invalid, allow_missing=True),),
        use_cgroup=False,
    )

    with pytest.raises(NotADirectoryError, match="seal path is not a directory"):
        boxed.resolve()


def test_ro_ancestor_of_tmpfs_phases_before_it(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO),),
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    ro_at = _dest_at(argv, "--ro-bind", str(tmp_path))
    assert ro_at < _dest_at(argv, "--tmpfs", str(home))


def test_descendant_of_a_tmpfs_is_not_hoisted(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    depot = home / "depot_tools"
    depot.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(depot, RO),),
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(depot)) > _dest_at(
        argv, "--tmpfs", str(home)
    )


@needs_bwrap
async def test_ro_ancestor_of_tmpfs_tmpfs_wins_end_to_end(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO),),
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)), ("PATH", "/usr/bin:/bin")),
        use_cgroup=False,
    )
    out = await run_boxed(
        'mkdir -p "$HOME/.cache" && touch "$HOME/.cache/_p" && echo HOME_CACHE_WRITABLE;'
        f"touch {tmp_path}/probe 2>&1",
        sandbox=boxed,
    )
    assert "HOME_CACHE_WRITABLE" in out
    assert "Read-only file system" in out


def test_system_ro_root_is_dropped_not_reemitted(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(root=root, binds=(Bind(Path("/usr"), RO),), use_cgroup=False)
    argv = boxed.wrapper()
    usr_binds = sum(
        1 for i in range(len(argv)) if argv[i] == "--ro-bind" and argv[i + 1] == "/usr"
    )
    assert usr_binds == 1


def test_symlinked_alias_and_its_target_are_both_bound(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    versioned = tmp_path / "cpython-3.12.12"
    versioned.mkdir()
    alias = tmp_path / "cpython-3.12"
    alias.symlink_to(versioned)
    boxed = Sandbox(
        root=root,
        binds=(Bind(alias, RO), Bind(versioned, RO)),
        use_cgroup=False,
    )
    dests = {d for op, d in _ops(boxed) if op == "ro"}
    assert str(alias) in dests
    assert str(versioned) in dests


def test_a_repeated_bind_is_emitted_once(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    dep = tmp_path / "dep"
    dep.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(dep, RO), Bind(dep, RO)),
        use_cgroup=False,
    )
    assert [d for op, d in _ops(boxed) if d == str(dep)] == [str(dep)]


def test_a_pin_restated_after_a_hole_is_not_deduped_away(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    private = gitdir / "worktrees" / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(private / "commondir", RO),
            Bind(private, RW),
            Bind(private / "commondir", RO),
        ),
        use_cgroup=False,
    )
    emitted = [d for op, d in _ops(boxed) if d == str(private / "commondir")]
    assert len(emitted) == 2


def test_assert_no_leak_raises_on_a_later_ancestor(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(BindOver(src, tmp_path),),
        tmpfs=((str(home), 1 << 20),),
        use_cgroup=False,
    )
    with pytest.raises(ValueError, match="leaks"):
        boxed.wrapper()


def test_assert_no_leak_raises_on_a_later_bind_of_the_exact_path(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(BindOver(home, home),),
        tmpfs=((str(home), 1 << 20),),
        use_cgroup=False,
    )
    with pytest.raises(ValueError, match="leaks"):
        boxed.wrapper()

    with pytest.raises(ValueError, match="leaks"):
        dataclasses.replace(boxed, binds=(BindOver(home, root),)).wrapper()


def test_a_seal_does_not_count_as_covering_its_own_remount(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    sealed = root / "worktrees"
    sealed.mkdir()
    Sandbox(root=root, binds=(Seal(sealed),), use_cgroup=False).wrapper()


def test_bind_over_places_a_file_at_a_different_path(tmp_path):
    src = tmp_path / "my-hosts"
    src.write_text("127.0.0.1 example\n")
    root = tmp_path / "wt"
    root.mkdir()
    argv = Sandbox(root=root, binds=(BindOver(src, Path("/etc/hosts")),)).wrapper()
    i = argv.index(str(src))
    assert argv[i - 1] == "--ro-bind"
    assert argv[i + 1] == "/etc/hosts"


def test_bind_over_written_last_lands_after_the_system_binds(tmp_path):
    src = tmp_path / "my-hosts"
    src.write_text("x\n")
    root = tmp_path / "wt"
    root.mkdir()

    ro = tmp_path / "dep"
    ro.mkdir()
    argv = Sandbox(
        root=root,
        binds=(Bind(ro, RO), BindOver(src, Path("/etc/hosts"))),
    ).wrapper()
    binds = [i for i, a in enumerate(argv) if a in ("--ro-bind", "--bind", "--tmpfs")]
    assert argv.index(str(src)) > max(binds[:-1])


def test_bind_over_source_must_exist(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(BindOver(tmp_path / "gone", Path("/etc/x")),))
    with pytest.raises(FileNotFoundError, match="bind-over source missing"):
        box.wrapper()


@pytest.mark.parametrize("destination", [Path("relative"), Path("/safe/../escape")])
def test_mount_destinations_cannot_depend_on_cwd_or_parent_traversal(
    tmp_path, destination
):
    src = tmp_path / "source"
    src.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(BindOver(src, destination),))

    with pytest.raises(
        ValueError, match="mount destination must be absolute and contain no"
    ):
        box.resolve()


def test_the_rw_root_cannot_contain_parent_traversal(tmp_path):
    real_root = tmp_path / "wt"
    real_root.mkdir()
    written_root = real_root / ".." / "wt"
    box = Sandbox(root=written_root)

    with pytest.raises(
        ValueError, match="mount destination must be absolute and contain no"
    ):
        box.resolve()


def test_bind_over_cannot_shadow_the_rw_root(tmp_path):
    src = tmp_path / "f"
    src.write_text("x\n")
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(BindOver(src, root),))
    with pytest.raises(ValueError, match="leaks"):
        box.wrapper()


def test_unshare_net_is_off_by_default(profile):
    assert "--unshare-net" not in profile.wrapper()


def test_unshare_net_is_passed_when_set(profile):
    boxed = dataclasses.replace(profile, unshare_net=True)
    assert "--unshare-net" in boxed.wrapper()


@needs_bwrap
async def test_unshare_net_gives_the_box_its_own_loopback(profile):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    boxed = dataclasses.replace(profile, unshare_net=True)
    try:
        out = await run_boxed(
            f'python3 -c "'
            f"import socket;s=socket.socket();s.settimeout(3);"
            f"print('rc=%d' % s.connect_ex(('127.0.0.1',{port})))\"",
            sandbox=boxed,
        )

        assert "rc=111" in out, out

        out = await run_boxed(
            'python3 -c "'
            "import socket,threading;"
            "srv=socket.socket();srv.bind(('127.0.0.1',0));srv.listen(1);"
            "threading.Thread(target=lambda: srv.accept().sendall(b'ok'),daemon=True).start();"
            "c=socket.socket();c.settimeout(3);c.connect(srv.getsockname());"
            'print(c.recv(8).decode())"',
            sandbox=boxed,
        )
        assert "ok" in out
    finally:
        srv.close()


@needs_bwrap
async def test_a_bound_unix_socket_crosses_the_network_namespace(profile, tmp_path):
    sock_dir = tmp_path / "sock"
    sock_dir.mkdir()
    sock = sock_dir / "s.sock"
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(sock))
    srv.listen(1)

    def _serve():
        conn, _ = srv.accept()
        conn.sendall(b"HOST")
        conn.close()

    threading.Thread(target=_serve, daemon=True).start()
    boxed = dataclasses.replace(
        profile,
        unshare_net=True,
        binds=(*profile.binds, Bind(sock_dir, RO)),
    )
    try:
        out = await run_boxed(
            f'python3 -c "'
            f"import socket;c=socket.socket(socket.AF_UNIX);c.settimeout(3);"
            f"c.connect('{sock}');print(c.recv(8).decode())\"",
            sandbox=boxed,
        )
        assert "HOST" in out
    finally:
        srv.close()


def test_overlay_source_must_exist(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(Overlay(tmp_path / "gone"),))
    with pytest.raises(FileNotFoundError, match="tmp-overlay source missing"):
        box.wrapper()


def test_overlay_cannot_shadow_a_tmpfs(tmp_path):

    src = tmp_path / "cache"
    src.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, tmpfs=((str(src), 1 << 20),), binds=(Overlay(src),))
    with pytest.raises(ValueError, match="leaks"):
        box.wrapper()


@needs_bwrap
async def test_overlay_is_warm_to_read_and_writes_go_nowhere(profile, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "warm.txt").write_text("from the host\n")
    boxed = dataclasses.replace(profile, binds=(*profile.binds, Overlay(cache)))
    out = await run_boxed(
        f"cat {cache}/warm.txt; echo poison > {cache}/evil.txt && echo WROTE",
        sandbox=boxed,
    )
    assert "from the host" in out
    assert "WROTE" in out

    assert not (cache / "evil.txt").exists()
    assert (cache / "warm.txt").read_text() == "from the host\n"
