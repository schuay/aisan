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
    home = tmp_path / "home"
    home.mkdir()
    # The launchers discover host MCP config through these when set; without
    # the scrub, a value inherited from THIS environment (an aisan box exports
    # OPENCODE_CONFIG) leaks the host's servers into a test that owns only HOME.
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
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert not (home / ".cache").exists()


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_net_explain_describes_shared_transport_without_live_secrets(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisan.cli.main",
            command,
            "--net",
            "--explain",
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
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
