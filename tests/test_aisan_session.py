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
