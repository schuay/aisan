# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from pathlib import Path

from aisan import Box
from aisan.explain import explain, normalise, parse_wrapper
from aisan.sandbox import RO, RW, Bind, BindOver, Sandbox, Seal
from aisan.spec import BoxSpec, Limits


def _idx(prof, path: Path) -> int:
    for m in prof.mounts:
        if Path(m.path).resolve() == path.resolve():
            return m.idx
    raise AssertionError(f"{path} is not mounted at all")


def test_wrapper_phases_ro_ancestor_before_the_tmpfs(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO, guard=False),),
        tmpfs=((str(home), 64 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    prof = parse_wrapper(sb.wrapper(), sb)
    assert _idx(prof, tmp_path) < _idx(prof, home)


def test_descendant_ro_under_home_lands_after_the_tmpfs(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    depot = home / "depot_tools"
    depot.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(
        root=root,
        binds=(Bind(depot, RO),),
        tmpfs=((str(home), 64 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    prof = parse_wrapper(sb.wrapper(), sb)
    assert _idx(prof, home) < _idx(prof, depot)


def test_parses_the_tmpfs_size_cap(tmp_path):

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(
        root=root,
        tmpfs=((str(home), 64 << 20),),
        use_cgroup=False,
    )
    prof = parse_wrapper(sb.wrapper(), sb)
    assert [m.size for m in prof.tmpfs] == [str(64 << 20)]


def test_classifies_a_pin_by_its_position_below_a_writable_mount(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    ctrl = tmp_path / "ctrl"
    ctrl.mkdir()
    dep = tmp_path / "dep"
    dep.mkdir()
    gitdir = ctrl / "gitdir"
    gitdir.mkdir()
    gitcfg = gitdir / "config"
    gitcfg.write_text("")
    sb = Sandbox(
        root=root,
        binds=(
            Bind(dep, RO),
            Bind(ctrl, RW, guard=False),
            Bind(gitdir, RW, guard=False),
            Bind(gitcfg, RO),
        ),
        tmpfs=(("/tmp", 64 << 20),),
        use_cgroup=False,
    )
    kinds = {Path(m.path).name: m.kind for m in parse_wrapper(sb.wrapper(), sb).mounts}
    assert kinds["config"] == "ro-pin"
    assert kinds["dep"] == "ro"
    assert kinds["wt"] == "rw-root"
    assert kinds["ctrl"] == "rw"
    assert kinds["usr"] == "system"


def test_a_pin_below_a_plain_rw_keeps_its_guard_marker(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    gitcfg = gitdir / "config"
    gitcfg.write_text("")
    sb = Sandbox(
        root=root,
        binds=(Bind(gitcfg, RO), Bind(gitdir, RW, guard=False)),
        use_cgroup=False,
    )
    prof = parse_wrapper(sb.wrapper(), sb)
    lines = {Path(m.path).name: m for m in prof.mounts}
    assert lines["config"].kind == "ro-pin"
    assert lines["config"].guard
    assert not lines["gitdir"].guard
    assert _idx(prof, gitdir) < _idx(prof, gitcfg)


def test_a_bind_over_is_labelled_a_substitution_not_a_plain_ro(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    src_file = tmp_path / "box-hosts"
    src_file.write_text("127.0.0.1 example\n")
    dst = tmp_path / "etc-hosts"
    dst.write_text("real\n")
    sb = Sandbox(
        root=root,
        binds=(BindOver(src_file, dst),),
        use_cgroup=False,
    )
    kinds = {Path(m.path): m.kind for m in parse_wrapper(sb.wrapper(), sb).mounts}
    assert kinds[dst] == "ro-sub"


def test_the_fixed_system_surface_is_surfaced_not_dropped(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(root=root, binds=(), use_cgroup=False)
    mounts = parse_wrapper(sb.wrapper(), sb).mounts
    by_kind = {m.kind for m in mounts}
    assert "proc" in by_kind
    assert "dev" in by_kind
    assert "symlink" in by_kind

    symlinks = {m.path for m in mounts if m.kind == "symlink"}
    assert "/bin" in symlinks and "/lib" in symlinks
    procs = {m.path for m in mounts if m.kind == "proc"}
    assert "/proc" in procs


def test_a_seal_is_reported_as_its_own_kind(tmp_path):

    root = tmp_path / "wt"
    root.mkdir()
    sealed = tmp_path / "worktrees"
    sealed.mkdir()
    sb = Sandbox(root=root, binds=(Seal(sealed),), use_cgroup=False)
    mounts = parse_wrapper(sb.wrapper(), sb).mounts
    at_sealed = [m.kind for m in mounts if Path(m.path) == sealed]
    assert at_sealed == ["tmpfs", "seal-ro"]

    assert mounts[-1].kind == "seal-ro"


def test_normalise_replaces_the_longest_host_prefix_first(tmp_path, monkeypatch):

    home = tmp_path / "t" / "home" / "u"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "t"))
    out = normalise(f"a {home}/x b {tmp_path / 't'}/y\n")
    assert out == "a <HOME>/x b <TMP>/y\n"


def test_normalise_hides_the_uid_in_the_nested_root(monkeypatch):
    import aisan.private as private_mod
    from aisan.spec import NESTING_ENV

    nested = NESTING_ENV["AISAN_PRIVATE_ROOT"]
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", Path("/tmp/aisan-99999"))

    out = normalise(f"env AISAN_PRIVATE_ROOT={nested} here")

    assert out == "env AISAN_PRIVATE_ROOT=<AISAN NESTED ROOT> here"
    assert nested not in out


def test_normalise_leaves_the_policy_alone(tmp_path):

    text = "  ro-pin  /x/config\n  --unshare-net\n  ( 4294967296)\n"
    assert normalise(text) == text


def test_normalise_blanks_the_mount_index_but_not_the_order(tmp_path):

    text = "  [ 44] rw-root  /a\n  [131] seal-ro  /b\n"
    assert normalise(text) == "  [..] rw-root  /a\n  [..] seal-ro  /b\n"


def test_normalise_labels_the_runtime_binds_without_hiding_any_line(
    tmp_path, monkeypatch
):
    import sys

    from aisan.launch import own_source_root

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    paths = [sys.prefix, sys.base_prefix]
    lines = "".join(f"  [ {i:>2}] ro       {p}\n" for i, p in enumerate(paths))
    out = normalise(f"  [ 10] rw-root  /a\n{lines}  [ 99] seal-ro  /b\n")
    assert out.count("\n") == 4, f"a mount line went missing:\n{out}"
    assert "<AISAN PREFIX>" in out
    assert "<AISAN BASE PREFIX>" in out
    assert own_source_root() is None or "<AISAN SRC>" in normalise(
        f"  [  0] ro       {own_source_root()}\n"
    )


def test_normalise_removes_no_line_from_a_report():

    # The report is the audit, so normalisation may rename a path but never drop
    # a mount. Runtime binds are included because eliding them once swallowed
    # the fixed system surface on a host where the two overlapped.
    from aisan.launch import launcher_binds

    runtime = "".join(
        f"  --ro-bind\n  {b.path}\n  {b.path}\n" for b in launcher_binds()
    )
    text = f"  --ro-bind\n  /usr\n  /usr\n{runtime}  --ro-bind\n  /etc\n  /etc\n"
    out = normalise(text)
    assert out.count("\n") == text.count("\n"), f"a line went missing:\n{out}"
    assert "\n  /usr\n  /usr\n" in out
    assert "\n  /etc\n  /etc\n" in out


def test_normalise_leaves_a_system_root_alone_when_it_is_a_prefix(monkeypatch):
    import sys

    # A venv created from the distro interpreter reports the system root as
    # its base prefix. Labelling it would rewrite every path under /usr,
    # including the fixed system surface the report exists to show.
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    text = "  --ro-bind\n  /usr\n  /usr\n  [  3] ro       /usr/bin/tool\n"
    assert (
        normalise(text)
        == "  --ro-bind\n  /usr\n  /usr\n  [..] ro       /usr/bin/tool\n"
    )


def test_the_package_does_not_re_export_the_renderer():
    import aisan as pkg
    import aisan.explain as mod

    assert pkg.explain is mod
    assert "explain" not in pkg.__all__


def test_a_credential_exposing_spec_renders_as_a_refusal(tmp_path):

    from aisan import Box
    from aisan.egress.anthropic import AnthropicBackend
    from aisan.explain import explain
    from aisan.sandbox import RW, Bind
    from aisan.spec import BoxSpec, Limits

    wt = tmp_path / "wt"
    wt.mkdir()
    creds = tmp_path / "creds" / "k.json"
    creds.parent.mkdir()
    spec = BoxSpec(
        root=wt,
        binds=(Bind(creds.parent, RW),),
        tmpfs=(),
        env=(),
        egress=(AnthropicBackend(credentials=creds),),
        unshare_net=True,
        limits=Limits(use_cgroup=False),
    )
    text = explain(Box(spec, box_id="t"), inputs=())
    assert "BOX ASSEMBLY REFUSED" in text
    assert "would expose the anthropic backend" in text


def test_a_bind_at_the_fixed_system_surface_renders_as_a_refusal(tmp_path):

    from aisan import Box
    from aisan.explain import explain
    from aisan.sandbox import RW, Bind
    from aisan.spec import BoxSpec, Limits

    wt = tmp_path / "wt"
    wt.mkdir()
    spec = BoxSpec(
        root=wt,
        binds=(Bind(Path("/dev"), RW),),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=False,
        limits=Limits(use_cgroup=False),
    )
    text = explain(Box(spec, box_id="t"), inputs=())
    assert "BOX ASSEMBLY REFUSED" in text
    assert "mount conflict at /dev" in text


def test_explaining_a_worktree_does_not_mutate_the_host_git(tmp_path):

    from aisan import Box
    from aisan.explain import explain
    from aisan.presets.depot_tools_job import depot_tools_job

    main = tmp_path / "main"
    wt = tmp_path / "wt"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (wt).mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")

    guards = [
        main / ".git" / "config",
        main / ".git" / "config.worktree",
        main / ".git" / "objects" / "info" / "alternates",
        main / ".git" / "hooks",
        main / ".git" / "objects" / "pack",
    ]
    before = {g: g.exists() for g in guards}
    assert not any(before.values()), "fixture should start without the guard files"

    spec = depot_tools_job(wt, depot_tools=None, unshare_net=True)
    assert not any(g.exists() for g in guards), "spec build mutated the host .git"

    box = Box(spec, box_id="explain-purity")
    with box.staged():
        report = explain(box, argv=False)
        assert "BOX ASSEMBLY REFUSED" not in report
        assert all(g.exists() for g in guards)

    assert not any(g.exists() for g in guards), "explain left files in the host .git"


def _linked_worktree(tmp_path):
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    return wt


def test_explain_cli_applies_egress_and_user_binds(tmp_path, capsys):

    from aisan.explain import main as explain_main

    wt = _linked_worktree(tmp_path)
    sisoenv = wt / "build" / "config" / "siso" / ".sisoenv"
    sisoenv.parent.mkdir(parents=True)
    sisoenv.write_text("SISO_PROJECT=rbe-chromium-untrusted\n")
    tool = tmp_path / "tooldir"
    tool.mkdir()
    binds = tmp_path / "tools.toml"
    binds.write_text(f'ro = ["{tool}"]\n')

    rc = explain_main(
        [
            "depot_tools_job",
            str(wt),
            "--egress",
            "v8-rbe",
            "--binds",
            str(binds),
            "--no-argv",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0

    assert "rbe" in out and "127.0.0.1:8712" in out

    assert "ro-sub" in out

    assert str(tool) in out

    assert "v8-rbe" in out
    assert "tools.toml" in out


def test_explain_cli_routes_a_bad_egress_or_binds_to_a_clean_refusal(tmp_path, capsys):

    from aisan.explain import main as explain_main

    wt = _linked_worktree(tmp_path)
    rc = explain_main(
        ["depot_tools_job", str(wt), "--binds", str(tmp_path / "absent.toml")]
    )
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_a_hole_is_reported_only_where_every_seal_at_that_path_allows_it(tmp_path):
    """Two seals may share a destination, and each refuses on its own list.

    The report is the audit, so it must show the holes that survive both.
    """
    root = tmp_path / "wt"
    root.mkdir()
    sealed = tmp_path / "worktrees"
    mine, theirs = sealed / "mine", sealed / "theirs"
    mine.mkdir(parents=True)
    theirs.mkdir()
    spec = BoxSpec(
        root=root,
        binds=(
            # The wider list comes last, so taking the last one would report
            # a hole the narrower seal refuses.
            Seal(sealed, allow=(mine,)),
            Seal(sealed, allow=(mine, theirs)),
            Bind(mine, RW),
        ),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=False,
        limits=Limits(use_cgroup=False),
    )
    report = explain(Box(spec, box_id="holes"), argv=False)
    assert f"    hole {mine}\n" in report
    assert "theirs" not in report
