# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


import importlib
import subprocess
from pathlib import Path

import pytest
from conftest import run_boxed

from aisan import Box
from aisan.gitbinds import external_symlink_targets, git_binds, git_host_files
from aisan.presets.claude_code import claude_code
from aisan.presets.codex import codex
from aisan.presets.depot_tools_job import depot_tools_grant, depot_tools_job
from aisan.presets.opencode import opencode
from aisan.sandbox import RO, RW, Bind, Overlay, Seal
from aisan.spec import Grant

depot_tools_job_module = importlib.import_module("aisan.presets.depot_tools_job")


def _fake_checkout(tmp_path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    for d in ("build", "third_party/icu", "third_party/ninja", ".git/worktrees/wt"):
        (main / d).mkdir(parents=True)
    (wt / "third_party").mkdir(parents=True)
    (wt / "src").mkdir()
    (wt / "out").mkdir()
    (wt / "build").symlink_to(main / "build")
    (wt / "third_party" / "icu").symlink_to(main / "third_party" / "icu")
    (wt / "third_party" / "ninja").symlink_to(main / "third_party" / "ninja")
    (wt / "src" / "local").symlink_to(wt / "out")
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")

    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    return main, wt


def _mounts(spec, box_id: str = "test-box"):
    box = Box(spec, box_id=box_id)
    with box.staged():
        return box.mounts()


def _mode_at(spec, path: Path) -> str:
    op = None
    for m in _mounts(spec):
        if Path(m.dst) == path:
            op = m.op
    if op is None:
        raise AssertionError(f"{path} is not mounted at all")
    return op


def _ro_sources(spec) -> list[Path]:
    from aisan.launch import launcher_binds

    aisan = {Path(str(b.path)) for b in launcher_binds()}
    return [
        Path(m.dst) for m in _mounts(spec) if m.op == "ro" and Path(m.dst) not in aisan
    ]


def test_external_symlink_targets_finds_outside_symlinks(tmp_path):
    main, wt = _fake_checkout(tmp_path)
    assert external_symlink_targets(wt) == [
        main / "build",
        main / "third_party" / "icu",
        main / "third_party" / "ninja",
    ]


def test_git_binds_is_the_policy_read_top_to_bottom(tmp_path):

    main, wt = _fake_checkout(tmp_path)
    git = main / ".git"
    assert git_binds(wt) == [
        Bind(git, RW),
        Bind(wt / ".git", RO),
        Bind(git / "config", RO),
        Bind(git / "config.worktree", RO),
        Bind(git / "objects" / "info" / "alternates", RO),
        Bind(git / "hooks", RO),
        Seal(git / "worktrees"),
        Bind(git / "worktrees" / "wt", RW),
        Bind(git / "worktrees" / "wt" / "commondir", RO),
        Bind(git / "worktrees" / "wt" / "config.worktree", RO),
    ]

    absent = git / "config.worktree"
    assert not absent.exists()
    git_binds(wt)
    assert not absent.exists()
    assert absent in {e.path for e in git_host_files(wt, pin_packs=True)}

    box = Box(depot_tools_job(wt, depot_tools=None), box_id="t")
    with box.staged():
        assert absent.exists()
        box.mounts()
    assert not absent.exists()


def test_a_sibling_needs_no_pin_because_the_seal_removes_it(tmp_path):

    main, wt = _fake_checkout(tmp_path)
    other = main / ".git" / "worktrees" / "other"
    other.mkdir()
    (other / "commondir").write_text("../..\n")
    paths = [b.path for b in git_binds(wt)]
    assert not any("other" in p.name for p in paths)
    assert Seal(main / ".git" / "worktrees") in git_binds(wt)


def test_a_sibling_pruned_mid_scan_cannot_break_assembly(tmp_path):

    main, wt = _fake_checkout(tmp_path)
    (main / ".git" / "worktrees" / "vanishing").mkdir()
    paths = [b.path for b in git_binds(wt)]
    assert not any("vanishing" in str(p) for p in paths)
    assert main / ".git" / "worktrees" / "wt" / "commondir" in paths


def test_git_binds_refuses_a_gitdir_pointer_outside_the_worktrees_layout(tmp_path):

    _main, wt = _fake_checkout(tmp_path)
    evil = wt / "evil_gitdir"
    evil.mkdir()
    (wt / ".git").write_text(f"gitdir: {evil}\n")
    with pytest.raises(ValueError, match="worktrees"):
        git_binds(wt)


def test_git_binds_refuses_a_pointer_into_an_unrelated_repo(tmp_path):

    victim = tmp_path / "victim"
    (victim / ".git" / "worktrees" / "z").mkdir(parents=True)

    (victim / ".git" / "worktrees" / "z" / "gitdir").write_text(
        f"{tmp_path}/victim_wt/.git\n"
    )
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {victim}/.git/worktrees/z\n")
    with pytest.raises(ValueError, match="does not own"):
        git_binds(wt)

    assert sorted(p.name for p in (victim / ".git").iterdir()) == ["worktrees"]


def test_git_binds_is_empty_without_a_git_dir(tmp_path):
    assert git_binds(tmp_path) == []
    assert git_host_files(tmp_path) == ()


def test_git_binds_pins_a_plain_checkouts_steering_files(tmp_path):

    git = tmp_path / ".git"
    (git / "hooks").mkdir(parents=True)
    (git / "config").write_text("[core]\n")
    assert git_binds(tmp_path, pin_packs=True) == [
        Bind(git, RW),
        Bind(git / "config", RO),
        Bind(git / "config.worktree", RO),
        Bind(git / "objects" / "info" / "alternates", RO),
        Bind(git / "hooks", RO),
        Seal(git / "worktrees"),
        Bind(git / "objects" / "pack", RO),
    ]

    assert not (git / "worktrees").exists()
    assert {e.path for e in git_host_files(tmp_path, pin_packs=True)} == {
        git / "hooks",
        git / "config",
        git / "config.worktree",
        git / "objects" / "info" / "alternates",
        git / "worktrees",
        git / "objects" / "pack",
    }


def test_git_binds_refuses_a_symlinked_git_dir(tmp_path):

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".git").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        git_binds(tmp_path / "repo")


@pytest.mark.skipif(
    __import__("shutil").which("bwrap") is None
    or __import__("shutil").which("git") is None,
    reason="bubblewrap and git needed",
)
async def test_a_plain_checkouts_git_cannot_be_steered_or_replaced_end_to_end(
    tmp_path,
):
    import dataclasses
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (tmp_path / "state").mkdir()
    spec = claude_code(root, state=tmp_path / "state")
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="plain-test")
    with box.staged():
        out = await run_boxed(
            f"cd {root} && "
            "git config core.fsmonitor /tmp/x 2>&1 || echo config_refused; "
            "echo x > .git/hooks/pre-commit 2>&1 || echo hook_refused; "
            "mv .git .git.old 2>&1 || echo rename_refused; "
            f"git worktree add -q {tmp_path}/wt 2>&1 || echo worktree_refused; "
            "git -c user.name=a -c user.email=a@b commit -q --allow-empty -m m"
            " && echo committed",
            sandbox=_WrapperOnly(box),
        )
    assert "committed" in out
    for refused in (
        "config_refused",
        "hook_refused",
        "rename_refused",
        "worktree_refused",
    ):
        assert refused in out, out

    assert (root / ".git").is_dir()
    assert "fsmonitor" not in (root / ".git" / "config").read_text()
    assert not (root / ".git" / "hooks" / "pre-commit").exists()
    assert not (tmp_path / "wt").exists()


def test_depot_tools_job_profile(tmp_path):
    main, wt = _fake_checkout(tmp_path)

    control = tmp_path / "control"
    control.mkdir(parents=True)

    (tmp_path / "depot_tools").mkdir()
    spec = depot_tools_job(
        wt,
        depot_tools=tmp_path / "depot_tools",
        memory_max="8G",
    ).with_binds([Bind(control, RW, optional=True)])
    assert spec.root == wt

    assert _mode_at(spec, control) == "rw"
    assert _mode_at(spec, main / ".git") == "rw"
    assert _mode_at(spec, main / ".git" / "config") == "ro"
    assert _mode_at(spec, main / ".git" / "hooks") == "ro"
    assert _mode_at(spec, main / ".git" / "config.worktree") == "ro"
    assert _mode_at(spec, main / "build") == "ro"
    assert _mode_at(spec, tmp_path / "depot_tools") == "ro"
    env = dict(spec.env)
    assert env["HOME"] == str(Path.home())
    assert str(tmp_path / "depot_tools") in env["PATH"]
    assert env["GIT_CONFIG_KEY_0"] == "gc.auto"

    box = Box(spec, box_id="t")
    with box.staged():
        assert "AI_AGENT" in box.wrapper()
    assert env["AI_AGENT"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert spec.limits.memory_max == "8G"


def test_depot_tools_job_refuses_a_root_that_is_not_there(tmp_path):
    gone = tmp_path / "reclaimed"
    with pytest.raises(ValueError, match=str(gone)):
        depot_tools_job(gone)


def test_depot_tools_job_binds_no_credential_at_all(tmp_path):
    _, wt = _fake_checkout(tmp_path)
    spec = depot_tools_job(wt, depot_tools=tmp_path / "depot_tools")
    env = dict(spec.env)

    assert not [k for k in env if "CREDENTIAL" in k.upper() or k.startswith("RBE_")]

    ro = _ro_sources(spec)
    assert all(tmp_path in p.parents for p in ro), ro


def test_depot_tools_job_skips_presubmit_network(tmp_path):

    _, wt = _fake_checkout(tmp_path)
    assert dict(depot_tools_job(wt).env)["PRESUBMIT_SKIP_NETWORK"] == "1"


def test_depot_tools_job_extra_path_lands_between_depot_tools_and_usr(tmp_path):
    _main, wt = _fake_checkout(tmp_path)
    spec = depot_tools_job(
        wt,
        extra_path=("/venv/bin", "/home/u/.local/bin"),
        depot_tools=tmp_path / "depot_tools",
    )
    assert (
        dict(spec.env)["PATH"]
        == f"{tmp_path / 'depot_tools'}:/venv/bin:/home/u/.local/bin:/usr/bin"
    )


@pytest.mark.skipif(
    __import__("shutil").which("bwrap") is None, reason="bubblewrap not installed"
)
async def test_a_job_cannot_plant_a_worktree_config_end_to_end(tmp_path):
    import dataclasses

    main, wt = _fake_checkout(tmp_path)

    spec = depot_tools_job(wt, depot_tools=None)
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    (main / ".git" / "refs").mkdir(exist_ok=True)
    wts = main / ".git" / "worktrees"
    box = Box(spec, box_id="seal-test")

    with box.staged():
        out = await run_boxed(
            f"mkdir {wts}/planted 2>&1; "
            f"echo x > {wts}/wt/config.worktree 2>&1; "
            f"touch {wts}/wt/index.lock && echo wrote_lock; "
            f"touch {main}/.git/refs/probe && echo wrote_refs",
            sandbox=_WrapperOnly(box),
        )
    assert "wrote_lock" in out
    assert "wrote_refs" in out
    assert out.count("Read-only file system") >= 2


class _WrapperOnly:
    def __init__(self, box) -> None:
        self._box = box

    def wrapper(self) -> list[str]:
        return self._box.wrapper()


def test_the_vpython_cache_is_an_overlay_not_a_bind(tmp_path, monkeypatch):

    preset = importlib.import_module("aisan.presets.depot_tools_job")

    cache = tmp_path / "vpython-root.999"
    cache.mkdir()
    monkeypatch.setattr(preset, "_vpython_cache", lambda: cache)
    _, wt = _fake_checkout(tmp_path)

    spec = depot_tools_job(wt)

    assert _mode_at(spec, cache) == "overlay"
    assert Overlay(cache) in spec.binds


def test_a_host_without_vpython_still_builds_a_profile(tmp_path, monkeypatch):

    preset = importlib.import_module("aisan.presets.depot_tools_job")

    monkeypatch.setattr(preset, "_vpython_cache", lambda: tmp_path / "nope")
    _, wt = _fake_checkout(tmp_path)
    assert not [b for b in depot_tools_job(wt).binds if isinstance(b, Overlay)]


def _linked_worktree(tmp_path: Path) -> tuple[Path, Path, str]:
    git = ("git", "-c", "user.email=t@t", "-c", "user.name=t")
    main = tmp_path / "main"
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "base"], cwd=main, check=True
    )
    wt, sib = tmp_path / "wt", tmp_path / "sib"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(wt), "-b", "b"], cwd=main, check=True
    )
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(sib)], cwd=main, check=True
    )
    (sib / "f.txt").write_text("only reachable from the sibling's HEAD\n")
    subprocess.run(["git", "add", "f.txt"], cwd=sib, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "detached"], cwd=sib, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=sib,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return wt, sib, sha


def test_an_interactive_preset_applies_the_worktree_git_policy(tmp_path):
    wt, _, _ = _linked_worktree(tmp_path)
    (tmp_path / "state").mkdir()
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=True)
    modes = {b.path: _mode_at(spec, b.path) for b in spec.binds if hasattr(b, "path")}
    common = (tmp_path / "main" / ".git").resolve()
    assert modes[common] == "rw"
    assert modes[common / "hooks"] == "ro"
    assert modes[common / "config"] == "ro"
    assert any(isinstance(b, Seal) for b in spec.binds)


@pytest.mark.parametrize("unshare_net", [True, False])
def test_the_pack_pin_follows_the_network_mode(tmp_path, unshare_net):
    wt, _, _ = _linked_worktree(tmp_path)
    (tmp_path / "state").mkdir()
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=unshare_net)
    packs = (tmp_path / "main" / ".git" / "objects" / "pack").resolve()

    pinned = Bind(packs, RO) in spec.binds
    assert pinned is unshare_net


def test_a_plain_checkout_gets_the_git_policy_too(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    state = tmp_path / "state"
    state.mkdir()
    spec = claude_code(repo, state=state, unshare_net=True)
    assert spec.binds == (*git_binds(repo, pin_packs=True), Bind(state, RW))
    assert Bind(repo / ".git" / "hooks", RO) in spec.binds


@pytest.mark.parametrize(
    "preset",
    [
        lambda wt, st: claude_code(wt, state=st, unshare_net=True),
        lambda wt, st: codex(wt, state=st, unshare_net=True),
        lambda wt, st: opencode(wt, state=st, unshare_net=True),
        lambda wt, st: depot_tools_job(wt, unshare_net=True),
    ],
)
def test_every_preset_disables_git_gc_in_the_box(tmp_path, preset):
    wt, _, _ = _linked_worktree(tmp_path)
    env = dict(preset(wt, tmp_path / "state").env)
    assert env["GIT_CONFIG_COUNT"] == "3"
    keys = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(3)}
    assert keys == {
        "gc.auto": "0",
        "gc.pruneExpire": "never",
        "gc.worktreePruneExpire": "never",
    }


@pytest.mark.parametrize(
    "preset",
    [
        lambda wt, st, ro: claude_code(wt, state=st, unshare_net=True, extra_ro=ro),
        lambda wt, st, ro: codex(wt, state=st, unshare_net=True, extra_ro=ro),
        lambda wt, st, ro: opencode(wt, state=st, unshare_net=True, extra_ro=ro),
        lambda wt, st, ro: depot_tools_job(wt, unshare_net=True, extra_ro=ro),
    ],
)
def test_only_extra_ro_is_plain_in_a_preset(tmp_path, preset):
    """Every bind a preset emits on its own is a guard; `extra_ro` carries the
    operator's read-only extras and is the only plain set."""
    wt, _, _ = _linked_worktree(tmp_path)
    extra = (tmp_path / "refs-a", tmp_path / "refs-b")
    for p in extra:
        p.mkdir()
    binds = preset(wt, tmp_path / "state", extra).binds
    flagged = [b for b in binds if isinstance(b, (Bind, Overlay))]
    assert {b.path for b in flagged if not b.guard} == set(extra)
    assert all(b.guard for b in flagged if b.path not in extra)


async def test_an_in_box_gc_cannot_destroy_a_sibling_worktrees_objects(tmp_path):
    import dataclasses

    wt, sib, sha = _linked_worktree(tmp_path)
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=True)

    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    (tmp_path / "state").mkdir()
    box = Box(spec, box_id=f"gc-{tmp_path.name}")
    with box.staged():
        out = await run_boxed(
            f"cd {wt} && git gc --prune=now; git status --porcelain", sandbox=box
        )

    assert "exit 0" in out
    alive = subprocess.run(
        ["git", "cat-file", "-e", sha], cwd=tmp_path / "main", check=False
    )
    assert alive.returncode == 0, f"the sibling's commit was collected:\n{out}"
    still_registered = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=sib,
        capture_output=True,
        text=True,
        check=False,
    )
    assert still_registered.returncode == 0, f"the sibling worktree was broken:\n{out}"


def test_a_planted_symlink_guard_source_is_not_followed(tmp_path):

    import dataclasses

    main, wt = _fake_checkout(tmp_path)
    target = tmp_path / "outside-the-git"
    assert not target.exists()
    planted = main / ".git" / "config.worktree"
    planted.symlink_to(target)

    spec = depot_tools_job(wt, depot_tools=None, unshare_net=True)
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="w12")
    with box.staged():
        pass
    assert not target.exists(), "a planted symlink was followed into a host write"


def test_a_submodule_gitdir_config_and_hooks_are_pinned(tmp_path):

    from aisan.gitbinds import git_binds

    main, wt = _fake_checkout(tmp_path)
    sub = main / ".git" / "modules" / "dep"
    sub.mkdir(parents=True)
    (sub / "config").write_text("")
    (sub / "hooks").mkdir()
    pinned = {b.path for b in git_binds(wt) if getattr(b, "mode", None) == RO}
    assert sub / "config" in pinned
    assert sub / "hooks" in pinned


def test_external_symlink_targets_confined_to_the_main_checkout(tmp_path, caplog):

    import logging

    from aisan.gitbinds import external_symlink_targets

    main, wt = _fake_checkout(tmp_path)

    (main / "buildtools").mkdir()
    (wt / "buildtools").symlink_to(main / "buildtools")

    secret = tmp_path / "victim" / ".ssh"
    secret.mkdir(parents=True)
    (wt / "deps").symlink_to(secret)
    (wt / "slash").symlink_to("/")

    with caplog.at_level(logging.WARNING, logger="aisan.gitbinds"):
        targets = external_symlink_targets(wt)

    assert main / "buildtools" in targets
    assert secret not in targets
    assert Path("/") not in targets
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "deps" in warned and "outside the main checkout" in warned


def test_the_depot_tools_grant_is_the_tree_its_cache_its_path_and_its_env(
    tmp_path, monkeypatch
):
    depot_tools = tmp_path / "depot_tools"
    depot_tools.mkdir()
    cache = tmp_path / "vpython-root.1000"
    cache.mkdir()
    monkeypatch.setattr(depot_tools_job_module, "_vpython_cache", lambda: cache)

    grant = depot_tools_grant(depot_tools)

    assert Overlay(cache) in grant.binds
    assert Bind(depot_tools, RO, optional=True) in grant.binds
    assert grant.path == (depot_tools,)
    assert dict(grant.env)["DEPOT_TOOLS_UPDATE"] == "0"
    assert dict(grant.env)["PRESUBMIT_SKIP_NETWORK"] == "1"


def test_a_host_without_depot_tools_has_an_empty_grant(monkeypatch):
    monkeypatch.setattr(depot_tools_job_module.shutil, "which", lambda _: None)

    grant = depot_tools_grant()

    assert not grant
    assert grant.binds == () and grant.path == () and grant.env == ()


def test_depot_tools_job_still_carries_the_depot_tools_environment(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    depot_tools = tmp_path / "depot_tools"
    depot_tools.mkdir()

    env = dict(depot_tools_job(worktree, depot_tools=depot_tools).env)

    assert env["DEPOT_TOOLS_UPDATE"] == "0"
    assert env["PRESUBMIT_SKIP_NETWORK"] == "1"
    assert env["PATH"].split(":")[0] == str(depot_tools)


def test_a_grant_cannot_name_path_through_its_environment():
    with pytest.raises(ValueError, match="names PATH through"):
        Grant(env=(("PATH", "/opt/tools"),))
