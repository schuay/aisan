# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from aisan.session import box_id, staged_directory, state_dir


def test_session_commands_are_packaged():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    expected = {
        "aisan": "aisan.cli.main:main",
        "aisan-box-launch": "aisan.launch:main",
    }
    scripts = project["project"]["scripts"]
    assert scripts == expected
    for command, target in expected.items():
        assert scripts[command] == target
        module_name, attribute = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_equal_basename_repositories_have_distinct_state(tmp_path):
    a = tmp_path / "a" / "repo"
    b = tmp_path / "b" / "repo"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    home = tmp_path / "home"
    assert state_dir("opencode", a, home=home) != state_dir("opencode", b, home=home)


def test_state_identity_is_stable_but_live_box_identity_is_unique(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    assert state_dir("claude", repo, home=home) == state_dir("claude", repo, home=home)
    assert box_id("claude", repo) != box_id("claude", repo)


def test_staged_directory_removes_only_what_it_created(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    path = existing / "client" / "repo"
    with staged_directory(path):
        assert path.is_dir()
    assert existing.is_dir()
    assert not (existing / "client").exists()


def _cli(tmp_path, argv):
    """A launcher run against a HOME of its own.

    The scrub matters: the launchers discover host MCP config through these
    variables, and one inherited from THIS environment (an aisan box exports
    OPENCODE_CONFIG) leaks the host's servers into a test that owns only HOME.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "aisan.cli.main", *argv],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.parametrize(
    "command",
    [
        "claude",
        "codex",
        "opencode",
    ],
)
def test_explain_leaves_no_persistent_session_state(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = _cli(tmp_path, [command, "--explain", str(repo)])

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home" / ".cache").exists()


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_net_explain_describes_shared_transport_without_live_secrets(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = _cli(tmp_path, [command, "--net", "--explain", str(repo)])

    assert result.returncode == 0, result.stderr
    assert "shared host namespace" in result.stdout
    assert "authenticated host-loopback TCP" in result.stdout
    assert "<per-box proxy token>" in result.stdout
    assert "WARNING:" not in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_bind_specs_compose_in_the_order_given(tmp_path, command):
    """`--binds` is repeatable so a shared tool spec and a per-project one
    compose without either knowing about the other -- later wins, which is the
    same rule that governs two entries in one file.

    Through `--explain`, because the merged profile is what the operator
    reviews: the assertions are on the rendered mounts and on the box's PATH,
    the two halves a `path` entry has to reach.
    """
    repo = tmp_path / "repo"
    tools = tmp_path / "tools"
    refs = tmp_path / "refs"
    for d in (repo, tools, refs):
        d.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    first = tmp_path / "tools.toml"
    first.write_text(f'ro = ["{tools}"]\npath = ["{tools}"]\n')
    second = tmp_path / "project.toml"
    second.write_text(f'ro = ["{refs}"]\n')

    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisan.cli.main",
            command,
            "--explain",
            "--binds",
            str(first),
            "--binds",
            str(second),
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert str(tools) in result.stdout
    assert str(refs) in result.stdout
    # Both files are named as inputs, so the artifact says what it was merged
    # from rather than only what came out.
    assert str(first) in result.stdout and str(second) in result.stdout
    path_line = next(
        line for line in result.stdout.splitlines() if line.strip().startswith("PATH=")
    )
    assert path_line.strip().removeprefix("PATH=").split(os.pathsep)[0] == str(tools)


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_refused_bind_spec_stops_the_launch(tmp_path, command):
    """The refusal reaches the operator as a nonzero exit and the file's name,
    not as a box quietly missing what the file asked for."""
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    bad = tmp_path / "bad.toml"
    bad.write_text(f'ro = ["{tmp_path}/a"]\npath = ["{tmp_path}/b"]\n')
    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisan.cli.main",
            command,
            "--explain",
            "--binds",
            str(bad),
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 2
    assert str(bad) in result.stderr and "not covered" in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_egress_profile_reaches_the_box(tmp_path, command):
    repo = tmp_path / "repo"
    (repo / "build" / "config" / "siso").mkdir(parents=True)
    (repo / "build" / "config" / "siso" / ".sisoenv").write_text("SISO_PROJECT=x\n")

    result = _cli(tmp_path, [command, "--egress", "v8-rbe", "--explain", str(repo)])

    assert result.returncode == 0, result.stderr
    # The backend's own line in the egress section, not any path that
    # happens to spell it.
    assert "rbe.sock" in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_egress_profile_refuses_the_shared_network(tmp_path, command):
    # The backend has no authenticated host-loopback path, so on a shared
    # namespace its relay port would be a credential capability for everything
    # on the machine. BoxSpec already refuses that mix; this refuses it in the
    # operator's terms, before anything is assembled.
    repo = tmp_path / "repo"
    (repo / "build" / "config" / "siso").mkdir(parents=True)
    (repo / "build" / "config" / "siso" / ".sisoenv").write_text("SISO_PROJECT=x\n")

    result = _cli(
        tmp_path, [command, "--egress", "v8-rbe", "--net", "--explain", str(repo)]
    )

    assert result.returncode == 2
    assert "--net" in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_unknown_egress_profile_is_refused_by_name(tmp_path, command):
    result = _cli(tmp_path, [command, "--egress", "nope", "--explain", str(tmp_path)])

    assert result.returncode == 2
    assert "v8-rbe" in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_profile_that_finds_nothing_says_so(tmp_path, command):
    # A tree the profile has nothing for is not a refusal, but silence would
    # leave a box that looks identical to a working one: the operator asked for
    # a route and got none.
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _cli(tmp_path, [command, "--egress", "v8-rbe", "--explain", str(repo)])

    assert result.returncode == 0, result.stderr
    assert "v8-rbe found nothing" in result.stderr
    assert "rbe.sock" not in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_naming_one_profile_twice_is_not_two_of_it(tmp_path, command):
    # Two of one backend collide on name and port, and BoxSpec refuses the mix.
    # The operator typed a repeat, not a request for a second route.
    repo = tmp_path / "repo"
    (repo / "build" / "config" / "siso").mkdir(parents=True)
    (repo / "build" / "config" / "siso" / ".sisoenv").write_text("SISO_PROJECT=x\n")

    result = _cli(
        tmp_path,
        [command, "--egress", "v8-rbe", "--egress", "v8-rbe", "--explain", str(repo)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rbe.sock") == 1


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_the_shared_net_profile_is_the_one_that_survives_net(tmp_path, command):
    # The other profile refuses --net; this one exists for it. What it grants is
    # a mount of the credential store, so the launch has to say so.
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "home" / ".config" / "chrome_infra" / "auth").mkdir(parents=True)

    result = _cli(
        tmp_path,
        [
            command,
            "--egress",
            "v8-rbe-with-net-unsafe",
            "--net",
            "--explain",
            str(repo),
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "chrome_infra" in result.stdout
    assert "luci credential" in result.stderr


@pytest.mark.parametrize("command", ["claude"])
def test_the_two_rbe_profiles_are_mutually_exclusive(tmp_path, command):
    # One keeps the credential on the host and serves a proxy; the other hands
    # the box the credential. Together they would serve a proxy to a box that
    # already holds what the proxy exists to withhold, and the credential guard
    # is what says so.
    repo = tmp_path / "repo"
    (repo / "build" / "config" / "siso").mkdir(parents=True)
    (repo / "build" / "config" / "siso" / ".sisoenv").write_text("SISO_PROJECT=x\n")
    (tmp_path / "home" / ".config" / "chrome_infra" / "auth").mkdir(parents=True)

    result = _cli(
        tmp_path,
        [
            command,
            "--egress",
            "v8-rbe",
            "--egress",
            "v8-rbe-with-net-unsafe",
            "--explain",
            str(repo),
        ],
    )

    assert "REFUSED" in result.stdout + result.stderr
    assert "expose the rbe backend's credential" in result.stdout + result.stderr
    # A refused profile is a failed review: --explain must exit nonzero so a
    # scripted `--explain && run` does not read this as a pass.
    assert result.returncode == 2, result.stdout + result.stderr
