# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The V8 job preset and the git bind policy under it.

`v8_job` is a pure function from arguments to a `BoxSpec`, so most of this reads
the spec directly. Where the claim is about what the box ENDS UP with (a pin that
wins over the rw mount under it, a tmpfs that is not shadowed) the question is
about mount ORDER, and the only honest answer comes from the resolved list --
which is `Box`'s, since resolution is what a Box does. Hence `_mounts` below.

What this file does NOT cover is which of a consumer's turns get a box, or where
one is rooted -- those are the consumer's routing decisions and its own suite
asserts them. Here the subject is the profile.
"""

import importlib
import subprocess
from pathlib import Path

import pytest
from conftest import run_boxed

from aisan import Box
from aisan.gitbinds import external_symlink_targets, git_binds
from aisan.presets.claude_code import claude_code
from aisan.presets.codex import codex
from aisan.presets.opencode import opencode
from aisan.presets.v8_job import v8_job
from aisan.sandbox import RO, RW, Bind, Overlay, Seal


def _fake_checkout(tmp_path) -> tuple[Path, Path]:
    """A main checkout + linked vt-style worktree: symlinked deps at the two
    real depths, a gitdir file, and some non-dep content."""
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
    (wt / "src" / "local").symlink_to(wt / "out")  # inside: not a dep
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    # Every real private worktree dir carries one; it is a pinned pointer, so
    # the fixture must have it for the profile to assemble.
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    return main, wt


def _mounts(spec, box_id: str = "test-box"):
    """The spec's mounts in the order bwrap applies them.

    Through a Box because that is the only thing that resolves a spec, and
    resolving it twice by two routes is how the inspector used to drift from the
    real path. box_id is arbitrary: with no egress it names a runtime dir
    nothing touches.
    """
    return Box(spec, box_id=box_id).mounts()


def _mode_at(spec, path: Path) -> str:
    """What the box's LAST mount at `path` makes it -- the one that wins.

    The profile is one ordered list, so "is this path ro" is not a question about
    membership in a field but about which mount is in effect at the end. Reading
    the last one is the same answer the kernel gives.
    """
    op = None
    for m in _mounts(spec):
        if Path(m.dst) == path:
            op = m.op
    if op is None:
        raise AssertionError(f"{path} is not mounted at all")
    return op


def _ro_sources(spec) -> list[Path]:
    """Every path THE PRESET mounts read-only, whatever role it plays.

    Aisan's own runtime (interpreter, venv, editable source roots) is subtracted:
    a Box binds it into every box regardless of spec, so it is not something the
    preset chose and not something a preset change could remove. Subtracted by
    asking `launcher_binds` rather than by pattern, so a preset that bound one of
    those paths deliberately would still be invisible here -- an accepted blind
    spot, and a small one, since they are already in the box either way.
    """
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
    # The whole common .git is rw (so git can create packed-refs.lock and never
    # prints the misleading "Read-only file system" on commit); everything that
    # STEERS host-side git is pinned ro AFTER it, since a planted hook or a
    # core.fsmonitor runs as the host user on the next host-side git.
    #
    # Asserted as the exact list, in order, because the order IS the policy: a
    # membership check would pass on a permutation that leaves .git wide open.
    main, wt = _fake_checkout(tmp_path)
    git = main / ".git"
    assert git_binds(wt) == [
        Bind(git, RW),  # in-worktree git works
        Bind(wt / ".git", RO),  # the pointer: repointing it bypasses every pin
        Bind(git / "config", RO),
        Bind(git / "config.worktree", RO),
        Bind(git / "objects" / "info" / "alternates", RO),  # object redirect
        Bind(git / "hooks", RO),
        Seal(git / "worktrees"),  # siblings absent, and nothing creatable
        Bind(git / "worktrees" / "wt", RW),  # ...except our own, for index.lock
        Bind(git / "worktrees" / "wt" / "commondir", RO),  # re-pinned inside it
        Bind(git / "worktrees" / "wt" / "config.worktree", RO),
    ]
    # Every non-optional source must exist or wrapper() refuses the profile, and
    # some of these are routinely absent on a real checkout (config.worktree
    # until someone runs `git config --worktree`, hooks/ under core.hooksPath):
    # git_binds creates them empty rather than leaving the profile unbuildable.
    for b in git_binds(wt):
        assert b.path.exists()


def test_a_sibling_needs_no_pin_because_the_seal_removes_it(tmp_path):
    # The behaviour change the seal buys, stated as a test rather than left
    # implicit. Before, every sibling's config.worktree and commondir were pinned
    # individually: create() enables extensions.worktreeConfig repo-wide, so a
    # SIBLING's config.worktree is live for host-side git in that worktree, and a
    # job able to write another job's private dir gets host exec on the next
    # squash there. The seal makes the whole directory empty in the box, so the
    # sibling is not read-only -- it is not there, and neither is anywhere to
    # plant a new one. No per-sibling scan, and nothing that a directory created
    # after assembly can slip past.
    main, wt = _fake_checkout(tmp_path)
    other = main / ".git" / "worktrees" / "other"
    other.mkdir()
    (other / "commondir").write_text("../..\n")
    paths = [b.path for b in git_binds(wt)]
    assert not any("other" in p.name for p in paths)
    assert Seal(main / ".git" / "worktrees") in git_binds(wt)


def test_a_sibling_pruned_mid_scan_cannot_break_assembly(tmp_path):
    # Assembly holds no repo lock, so a concurrent `vt remove` can prune another
    # job's private dir while the profile is built. That used to matter: a
    # half-deleted sibling's pin was a missing mandatory source, which refuses
    # the profile and abandons THIS job's turn over an unrelated job's
    # lifecycle. With the seal, siblings are not named at all, so there is
    # nothing to race.
    main, wt = _fake_checkout(tmp_path)
    (main / ".git" / "worktrees" / "vanishing").mkdir()  # no commondir: mid-removal
    paths = [b.path for b in git_binds(wt)]
    assert not any("vanishing" in str(p) for p in paths)
    assert main / ".git" / "worktrees" / "wt" / "commondir" in paths  # ours stays


def test_git_binds_refuses_a_gitdir_pointer_outside_the_worktrees_layout(tmp_path):
    # A pointer already repointed at an agent-controlled directory must not be
    # taken at face value: main_git is derived from it and bound RW, so trusting
    # it would bind the attacker's directory as if it were the shared .git.
    _main, wt = _fake_checkout(tmp_path)
    evil = wt / "evil_gitdir"
    evil.mkdir()
    (wt / ".git").write_text(f"gitdir: {evil}\n")
    with pytest.raises(ValueError, match="worktrees"):
        git_binds(wt)


def test_git_binds_plain_repo_is_noop(tmp_path):
    (tmp_path / ".git").mkdir()
    assert git_binds(tmp_path) == []


def test_v8_job_profile(tmp_path):
    main, wt = _fake_checkout(tmp_path)
    # A realistic layout: the control dir is the consumer's own bind, appended
    # to the preset's result rather than passed into it -- the preset knows
    # nothing about a job control dir, which is the point of with_binds.
    control = tmp_path / "control"
    control.mkdir(parents=True)
    # depot_tools has to be real for the assertions below to mean anything: an
    # optional bind whose source does not exist is dropped, so a fixture that
    # skips it would leave every claim about it vacuously unmounted.
    (tmp_path / "depot_tools").mkdir()
    spec = v8_job(
        wt,
        depot_tools=tmp_path / "depot_tools",
        memory_max="8G",
    ).with_binds([Bind(control, RW, optional=True)])
    assert spec.root == wt
    # The effective mount at each path, not the field it was passed in: the
    # control dir writable as a whole, the common .git writable for
    # packed-refs.lock, and the steering files read-only ON TOP of it -- which is
    # a claim about what wins, so it is asked of the resolved list.
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
    # siso reads AI_AGENT to go quiet. The spec is the box's WHOLE environment
    # (--clearenv, then --setenv per entry, plus only the backends' client vars),
    # so a defang default that is not named here is one the box does not get.
    assert "AI_AGENT" in Box(spec, box_id="t").wrapper()
    assert env["AI_AGENT"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert spec.limits.memory_max == "8G"


def test_v8_job_binds_no_credential_at_all(tmp_path):
    """The box holds nothing a credential could be read out of, in any mode.

    This guards the trap the legacy path used to be: it bound the chrome_infra
    refresh token (gerrit + cloud scoped, non-expiring, renewable in-box), and
    later a per-job dir holding a brokered bearer. Both are gone -- egress runs
    through host-side backends that mint per request -- so the assertion is now
    unconditional, and a regression here is a security regression rather than a
    behaviour change.
    """
    _, wt = _fake_checkout(tmp_path)
    spec = v8_job(wt, depot_tools=tmp_path / "depot_tools")
    env = dict(spec.env)
    # No credential-bearing env of any kind: no credshelper for siso, reclient,
    # or bb, and nothing pointing luci-auth at an in-box login.
    assert not [k for k in env if "CREDENTIAL" in k.upper() or k.startswith("RBE_")]
    # And no bind reaches outside the checkout for one. Every ro mount the
    # profile asks for is under the fake root here -- checked by containment
    # rather than by name, since a name test would pass on a credential dir
    # called something else.
    ro = _ro_sources(spec)
    assert all(tmp_path in p.parents for p in ro), ro


def test_v8_job_skips_presubmit_network(tmp_path):
    # `git cl presubmit` reaches Gerrit over authenticated /a/ endpoints for the
    # CL description and OWNERS. The box has no SSO cookie by design, so those
    # fail on git-remote-sso ("SSO login required") -- which reads as a
    # broken tree and burns revise rounds. depot_tools' own switch drops exactly
    # the network half and keeps the local checks.
    _, wt = _fake_checkout(tmp_path)
    assert dict(v8_job(wt).env)["PRESUBMIT_SKIP_NETWORK"] == "1"


def test_v8_job_extra_path_lands_between_depot_tools_and_usr(tmp_path):
    _main, wt = _fake_checkout(tmp_path)
    spec = v8_job(
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
    """The host-exec vector this seal closes, exercised through the real profile.

    A config.worktree the box writes is executed BY THE HOST: with
    extensions.worktreeConfig on (create() enables it repo-wide), host-side git
    running in that worktree reads core.fsmonitor from it and runs it as the
    host user. Pinning the files that exist at assembly time does not cover a
    directory the box creates afterwards, so the directory itself is sealed.
    """
    import dataclasses

    main, wt = _fake_checkout(tmp_path)
    # cgroup off: a transient systemd scope is not available in every test env,
    # and what is under test is the mount policy.
    spec = v8_job(wt, depot_tools=None)
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    (main / ".git" / "refs").mkdir(exist_ok=True)  # the fixture has no refs/ dir
    wts = main / ".git" / "worktrees"
    out = await run_boxed(
        f"mkdir {wts}/planted 2>&1; "
        f"echo x > {wts}/wt/config.worktree 2>&1; "
        f"touch {wts}/wt/index.lock && echo wrote_lock; "
        f"touch {main}/.git/refs/probe && echo wrote_refs",
        # The runner takes something with wrapper(); a Box is that, and using the
        # real one keeps this exercising the complete argv.
        sandbox=_WrapperOnly(Box(spec, box_id="seal-test")),
    )
    assert "wrote_lock" in out  # git can still lock its own index
    assert "wrote_refs" in out  # refs stay writable (documented give-up)
    assert out.count("Read-only file system") >= 2  # no plant, no own-cfg rewrite


class _WrapperOnly:
    """Adapts a Box to the runner's `sandbox=` parameter, which wants a wrapper()
    and nothing else. A Box is not a Sandbox -- it OWNS one -- so the adapter is
    the honest way to say which half is being used."""

    def __init__(self, box) -> None:
        self._box = box

    def wrapper(self) -> list[str]:
        return self._box.wrapper()


def test_the_vpython_cache_is_an_overlay_not_a_bind(tmp_path, monkeypatch):
    """The tool cache every `git cl` runs through, exposed without exposing it.

    Three of the four possible bindings are wrong and two of them were shipped:
    ro fails vpython's read lock (14 of ~45 friction entries), absent makes it
    hang with no network, and rw would let one box rewrite the interpreter every
    other box -- and the host -- then executes. Only the overlay is both warm and
    contained, so the FIELD is asserted, not merely the path.
    """
    # The MODULE, not the function of the same name: `presets.v8_job` is the
    # function (the package re-exports it), so reaching the module it lives in
    # takes an explicit import.
    preset = importlib.import_module("aisan.presets.v8_job")

    cache = tmp_path / "vpython-root.999"
    cache.mkdir()
    monkeypatch.setattr(preset, "_vpython_cache", lambda: cache)
    _, wt = _fake_checkout(tmp_path)

    spec = v8_job(wt)
    # The OP is asserted, not merely the path: rw and ro are both present in the
    # same list now, so "the cache is bound" says nothing -- three of the four
    # ways to bind it are bugs, and two of them shipped.
    assert _mode_at(spec, cache) == "overlay"
    assert Overlay(cache) in spec.binds


def test_a_host_without_vpython_still_builds_a_profile(tmp_path, monkeypatch):
    # A missing cache is skipped rather than raising: the box then behaves as it
    # did before the overlay existed, which is the honest degradation for a host
    # that has never run vpython.
    preset = importlib.import_module("aisan.presets.v8_job")

    monkeypatch.setattr(preset, "_vpython_cache", lambda: tmp_path / "nope")
    _, wt = _fake_checkout(tmp_path)
    assert not [b for b in v8_job(wt).binds if isinstance(b, Overlay)]


# --- the shared .git of a linked worktree ------------------------------------


def _linked_worktree(tmp_path: Path) -> tuple[Path, Path, str]:
    """A main checkout, our worktree, and a SIBLING holding a commit only its
    own detached HEAD references. Returns (worktree, sibling, sha).

    Real git rather than a hand-built layout: what is under test is what git
    itself concludes from the layout, and the pointers/reachability rules are
    exactly the part a fake would get to invent.
    """
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
    """The interactive presets get the same .git policy as the job profile.

    Before this they got none, so a linked worktree simply did not work in a
    session -- and the operator's fix was a hand-written rw bind of the common
    .git, which is the policy MINUS every pin. Making it the default is what
    stops the escape hatch from being the dangerous path.
    """
    wt, _, _ = _linked_worktree(tmp_path)
    (tmp_path / "state").mkdir()
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=True)
    modes = {b.path: _mode_at(spec, b.path) for b in spec.binds if hasattr(b, "path")}
    common = (tmp_path / "main" / ".git").resolve()
    assert modes[common] == "rw"  # git needs to write refs and objects
    assert modes[common / "hooks"] == "ro"  # but not what steers host-side git
    assert modes[common / "config"] == "ro"
    assert any(isinstance(b, Seal) for b in spec.binds)  # siblings are absent


@pytest.mark.parametrize("unshare_net", [True, False])
def test_the_pack_pin_follows_the_network_mode(tmp_path, unshare_net):
    """Pinned when the box has its own namespace, absent when it shares the
    host's. The pin is what stops an in-box gc pruning a sibling's objects, and
    its cost is that index-pack cannot run -- which only matters in a box that
    could have fetched in the first place."""
    wt, _, _ = _linked_worktree(tmp_path)
    (tmp_path / "state").mkdir()
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=unshare_net)
    packs = (tmp_path / "main" / ".git" / "objects" / "pack").resolve()
    # Asked of the bind list rather than the resolved mounts: when the pin is
    # absent the path is not a mount point at all, it is writable through the
    # common .git bind above it.
    pinned = Bind(packs, RO) in spec.binds
    assert pinned is unshare_net


def test_a_plain_checkout_gets_no_git_policy_at_all(tmp_path):
    """The hazard is a .git SHARED with worktrees the box cannot see. A plain
    checkout's .git is inside the rw root and inside the session's perimeter,
    so the policy is empty rather than merely harmless."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    state = tmp_path / "state"
    state.mkdir()
    spec = claude_code(repo, state=state, unshare_net=True)
    assert spec.binds == (Bind(state, RW),)  # the state dir, and nothing else


@pytest.mark.parametrize(
    "preset",
    [
        lambda wt, st: claude_code(wt, state=st, unshare_net=True),
        lambda wt, st: codex(wt, state=st, unshare_net=True),
        lambda wt, st: opencode(wt, state=st, unshare_net=True),
        lambda wt, st: v8_job(wt, unshare_net=True),
    ],
)
def test_every_preset_disables_git_gc_in_the_box(tmp_path, preset):
    """Through the environment, so it needs no host state and cannot outlive
    the box. Defence in depth rather than the guard -- the pack pin is the
    guard -- but it stops the automatic path before it starts."""
    wt, _, _ = _linked_worktree(tmp_path)
    env = dict(preset(wt, tmp_path / "state").env)
    assert env["GIT_CONFIG_COUNT"] == "3"
    keys = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(3)}
    assert keys == {
        "gc.auto": "0",
        "gc.pruneExpire": "never",
        "gc.worktreePruneExpire": "never",
    }


async def test_an_in_box_gc_cannot_destroy_a_sibling_worktrees_objects(tmp_path):
    """The regression this policy exists for, from inside a real box.

    The chain it breaks: the box binds the SHARED .git but not the sibling
    worktrees' directories, so git concludes those worktrees were deleted (it
    reports them `prunable`), and the seal that stops it acting on that also
    removes their HEAD and index as reachability roots -- after which a gc
    prunes objects only they reference, out of a store the host is still using.
    Measured before the pin: the sibling's commit was destroyed and it was left
    at "fatal: bad object HEAD".

    Asserted on the HOST after the box exits, because that is where the damage
    would be, and with `--prune=now` because the danger is the command an agent
    types rather than the one git runs on its own.
    """
    import dataclasses

    wt, sib, sha = _linked_worktree(tmp_path)
    spec = claude_code(wt, state=tmp_path / "state", unshare_net=True)
    # cgroup off: a transient systemd scope is not available in every test env,
    # and what is under test is the .git guard, not the resource limits.
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    (tmp_path / "state").mkdir()
    box = Box(spec, box_id=f"gc-{tmp_path.name}")
    with box.staged():
        out = await run_boxed(
            f"cd {wt} && git gc --prune=now; git status --porcelain", sandbox=box
        )
    # The session's own git still works: the point is a guard, not a broken box.
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
