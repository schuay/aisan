# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The explain renderer: parse_wrapper over REAL Sandbox.wrapper() output.

These tests drive the actual wrapper() argv (not a hand-rolled token list), so
a reorder in the sandbox that changes mount precedence is caught here, from the
side that runs it. Assembly asserts the invariant; this is the caller
checking that the argv it hands bwrap still has the ordering it depends on.

The classification these exercise is what makes the report an audit artifact
rather than a dump: a pin is derived from the ordering, not from a field, so the
label cannot disagree with what the box gets.
"""

from pathlib import Path

from aisan.explain import normalise, parse_wrapper
from aisan.sandbox import RO, RW, Bind, BindOver, Sandbox, Seal


def _idx(prof, path: Path) -> int:
    """argv index of the mount at `path`, which is what decides precedence."""
    for m in prof.mounts:
        if Path(m.path).resolve() == path.resolve():
            return m.idx
    raise AssertionError(f"{path} is not mounted at all")


def test_wrapper_phases_ro_ancestor_before_the_tmpfs(tmp_path):
    # The motivating /usr-vs-$HOME case: a ro bind covering the home tmpfs must land
    # BEFORE it, or it shadows the blanked home read-only (and exposes the real
    # one). Core owns that phasing; this is the consumer-side guard that the
    # argv the caller actually runs still has it.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO),),  # ro ancestor of the home tmpfs
        tmpfs=((str(home), 64 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    prof = parse_wrapper(sb.wrapper(), sb)
    assert _idx(prof, tmp_path) < _idx(prof, home)


def test_descendant_ro_under_home_lands_after_the_tmpfs(tmp_path):
    # depot_tools at ~/depot_tools is a DESCENDANT of home: it must land AFTER
    # the tmpfs to stay visible on top of the blanked home. The inverse of the
    # case above, and the reason phasing is a split rather than a global sort.
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
    # Emitted as `--size N --tmpfs MNT`; an index slip here silently reports
    # "no size" for every tmpfs, which reads as "no cap configured".
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


def test_classifies_a_pin_by_the_ordering_not_by_a_field(tmp_path):
    # A dump that renders every ro bind the same way buries the ones that matter:
    # a reviewer scanning it should see the pinned .git config distinctly, not as
    # one line in a wall of "ro". There is no longer a field to read that off, so
    # the inspector derives it from what a pin IS in this model -- an ro bind
    # after an rw one that covers it -- which means the label comes from the same
    # ordering the box gets and cannot disagree with it.
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
            Bind(ctrl, RW),
            Bind(gitdir, RW),
            Bind(gitcfg, RO),  # ro AFTER the rw parent: a pin
        ),
        tmpfs=(("/tmp", 64 << 20),),
        use_cgroup=False,
    )
    kinds = {Path(m.path).name: m.kind for m in parse_wrapper(sb.wrapper(), sb).mounts}
    assert kinds["config"] == "ro-pin"  # inside an rw mount: the guard
    assert kinds["dep"] == "ro"  # plain ro: not over anything writable
    assert kinds["wt"] == "rw-root"
    assert kinds["ctrl"] == "rw"
    assert kinds["usr"] == "system"  # from the fixed surface, not the policy


def test_a_ro_bind_a_later_rw_defeats_is_labelled_shadowed(tmp_path):
    # The negative that makes the pin label mean something, and the honesty fix
    # for it. Same two binds as the pin test, other order: the rw parent is
    # written AFTER, so it wins and the file is WRITABLE in the box. Labelling
    # that "ro" tells a reviewer the file is protected when the box can rewrite
    # it -- the H7-class rendering. "ro-shadow" says what is actually true.
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    gitcfg = gitdir / "config"
    gitcfg.write_text("")
    sb = Sandbox(
        root=root,
        binds=(Bind(gitcfg, RO), Bind(gitdir, RW)),
        use_cgroup=False,
    )
    kinds = {Path(m.path).name: m.kind for m in parse_wrapper(sb.wrapper(), sb).mounts}
    assert kinds["config"] == "ro-shadow"


def test_a_bind_over_is_labelled_a_substitution_not_a_plain_ro(tmp_path):
    # A BindOver mounts src AT dst -- the box reads a DIFFERENT file than the
    # host's own dst (ReapiBackend's /etc/hosts redirect, a per-box .sisoenv).
    # Rendered as a plain "ro" of dst it reads as "the host's dst is mounted",
    # the opposite of what happens. src != dst in the argv is the tell.
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
    # /proc, /dev and the merged-usr symlinks are part of what the box can
    # touch, but the inspector recognised only --ro-bind/--bind and skipped
    # them, so the report never said the box had a /proc or a /dev. They are
    # named now, each by its own kind.
    root = tmp_path / "wt"
    root.mkdir()
    sb = Sandbox(root=root, binds=(), use_cgroup=False)
    mounts = parse_wrapper(sb.wrapper(), sb).mounts
    by_kind = {m.kind for m in mounts}
    assert "proc" in by_kind
    assert "dev" in by_kind
    assert "symlink" in by_kind
    # The link paths, not the merged-usr targets: /bin is what exists in the box.
    symlinks = {m.path for m in mounts if m.kind == "symlink"}
    assert "/bin" in symlinks and "/lib" in symlinks
    procs = {m.path for m in mounts if m.kind == "proc"}
    assert "/proc" in procs


def test_a_seal_is_reported_as_its_own_kind(tmp_path):
    # A seal reaches the argv as two ops -- an empty tmpfs and, at the very end,
    # a --remount-ro. A dump that showed only the first would report a writable
    # scratch directory where the box has an immutable empty one, and one that
    # showed neither would leave the .git dance unexplained.
    root = tmp_path / "wt"
    root.mkdir()
    sealed = tmp_path / "worktrees"
    sealed.mkdir()
    sb = Sandbox(root=root, binds=(Seal(sealed),), use_cgroup=False)
    mounts = parse_wrapper(sb.wrapper(), sb).mounts
    at_sealed = [m.kind for m in mounts if Path(m.path) == sealed]
    assert at_sealed == ["tmpfs", "seal-ro"]
    # And the remount is last, which is what makes holes through the seal
    # possible at all -- see the core suite for why.
    assert mounts[-1].kind == "seal-ro"


def test_normalise_replaces_the_longest_host_prefix_first(tmp_path, monkeypatch):
    # A home under the temp dir is a real CI layout, and it is exactly where a
    # naive replace order goes wrong: <TMP> eats the prefix, the home rule never
    # matches, and the snapshot carries a path fragment that differs per host --
    # which is a snapshot that fails for a reason nobody caused.
    home = tmp_path / "t" / "home" / "u"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "t"))
    out = normalise(f"a {home}/x b {tmp_path / 't'}/y\n")
    assert out == "a <HOME>/x b <TMP>/y\n"


def test_normalise_leaves_the_policy_alone(tmp_path):
    # The complement, stated as a test because it is the whole contract: host
    # facts get placeholders, everything else -- ordering, kinds, sizes, flags
    # -- is the policy under audit and must survive verbatim.
    text = "  ro-pin  /x/config\n  --unshare-net\n  ( 4294967296)\n"
    assert normalise(text) == text


def test_normalise_blanks_the_mount_index_but_not_the_order(tmp_path):
    # The index is a token offset into an argv whose first thirty-odd tokens are
    # a fixed system preamble, so it says nothing the line ORDER does not -- and
    # it shifts wholesale when anything earlier gains a mount, turning one real
    # change into a diff against every line below it. Blank the number, keep the
    # sequence, which is the part that IS the policy.
    text = "  [ 44] rw-root  /a\n  [131] seal-ro  /b\n"
    assert normalise(text) == "  [..] rw-root  /a\n  [..] seal-ro  /b\n"


def test_normalise_collapses_aisans_own_runtime_binds():
    # Aisan binds its interpreter, venv and editable source roots into EVERY box
    # (Box._sandbox), so how many lines that is depends on how aisan is
    # installed here: one site-packages tree for a released install, one per
    # workspace member for a checkout. It is identical in every box by
    # construction, so it can never be what a preset diff changed -- but left in
    # a snapshot it fails on somebody else's install layout while staying green
    # through a real policy change.
    from aisan.launch import launcher_binds

    paths = [str(b.path) for b in launcher_binds()]
    assert len(paths) > 1, "the collapse is only interesting for a run of lines"
    lines = "".join(f"  [ {i:>2}] ro       {p}\n" for i, p in enumerate(paths))
    out = normalise(f"  [ 10] rw-root  /a\n{lines}  [ 99] seal-ro  /b\n")
    assert out == "  [..] rw-root  /a\n  <AISAN RUNTIME>\n  [..] seal-ro  /b\n"


def test_normalise_drops_the_bind_flag_with_the_path_it_names():
    # In the argv section a bind is three lines (--ro-bind, src, dst). Dropping
    # only the paths would leave a bare --ro-bind pointing at the NEXT mount's
    # source, which is a snapshot that reads as a real argv and is not one.
    from aisan.launch import launcher_binds

    p = str(launcher_binds()[0].path)
    out = normalise(f"  --ro-bind\n  {p}\n  {p}\n  --unshare-net\n")
    assert out == "  <AISAN RUNTIME>\n  --unshare-net\n"


def test_the_package_does_not_re_export_the_renderer():
    """`aisan.explain` is the MODULE, and only the module.

    A package that re-exports a function under its own submodule's name has two
    meanings for one attribute, and which one a caller gets depends on whether
    anything imported the submodule first -- so `explain(box, ...)` works until
    an unrelated import reorders, then raises TypeError. Reached through the
    package's own deferred-import hook it was worse: the from-import form probes
    hasattr() before importing, the probe re-entered the hook, and a real
    the previous inspector died in a RecursionError with no mention of either
    name. Found by hand-running the command, not by the suite.
    """
    import aisan as pkg
    import aisan.explain as mod

    assert pkg.explain is mod
    assert "explain" not in pkg.__all__


def test_a_credential_exposing_spec_renders_as_a_refusal(tmp_path):
    # The Box refuses to assemble a spec whose binds would expose a backend
    # credential, and explain renders that refusal as the finding rather than
    # crashing on it -- which is what the mode is FOR: an operator who ran
    # `--explain` on a merged profile with a bad user bind sees the refusal,
    # not a traceback, before any box owns their terminal.
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


def test_explaining_a_worktree_does_not_mutate_the_host_git(tmp_path):
    # M6: git_binds used to mkdir/touch the shared .git at spec-build time, so
    # `aisan explain <worktree>` -- a dry run -- wrote config.worktree, hooks/,
    # objects/info/alternates and objects/pack into the main checkout. The
    # creation moved to the Box, which undoes it on the inspector's staged path,
    # so a review leaves the host exactly as it found it.
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

    # Building the spec must not mutate: git_binds is pure now.
    spec = depot_tools_job(wt, depot_tools=None, unshare_net=True)
    assert not any(g.exists() for g in guards), "spec build mutated the host .git"

    box = Box(spec, box_id="explain-purity")
    with box.staged():
        # Inside staging the guards exist, so the profile resolves and renders.
        report = explain(box, argv=False)
        assert "BOX ASSEMBLY REFUSED" not in report
        assert all(g.exists() for g in guards)
    # ...and are gone again afterward: a dry run left no trace.
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
    # A review of a preset ALONE omits the largest thing a caller adds: the
    # egress route and the user bind files. `aisan explain` takes both now, the
    # same knobs the launchers do, and renders their effect.
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
    # The egress backend is rendered on its loopback port...
    assert "rbe" in out and "127.0.0.1:8712" in out
    # ...its .sisoenv redirect reads as a substitution, not a plain ro...
    assert "ro-sub" in out
    # ...the user bind landed...
    assert str(tool) in out
    # ...and the inputs name both, so the review states its own configuration.
    assert "v8-rbe" in out
    assert "tools.toml" in out


def test_explain_cli_routes_a_bad_egress_or_binds_to_a_clean_refusal(tmp_path, capsys):
    # The shared applier raises LaunchRefused, which the CLI turns into a nonzero
    # exit and a named reason rather than a traceback -- the same convention the
    # launchers speak.
    from aisan.explain import main as explain_main

    wt = _linked_worktree(tmp_path)
    rc = explain_main(
        ["depot_tools_job", str(wt), "--binds", str(tmp_path / "absent.toml")]
    )
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err
