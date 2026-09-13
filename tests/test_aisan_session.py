# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from aisan.sandbox import RO, Bind, BindOver
from aisan.session import (
    LauncherFlags,
    box_id,
    git_config_binds,
    run_interactive,
    staged_directory,
    state_dir,
)


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


def _cli(tmp_path, argv, host_path=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    if host_path is not None:
        env["PATH"] = str(host_path)
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

    assert str(first) in result.stdout and str(second) in result.stdout
    path_line = next(
        line for line in result.stdout.splitlines() if line.strip().startswith("PATH=")
    )
    assert path_line.strip().removeprefix("PATH=").split(os.pathsep)[0] == str(tools)


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_refused_bind_spec_stops_the_launch(tmp_path, command):
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

    assert "rbe.sock" in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_egress_profile_refuses_the_shared_network(tmp_path, command):

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

    repo = tmp_path / "repo"
    repo.mkdir()

    result = _cli(tmp_path, [command, "--egress", "v8-rbe", "--explain", str(repo)])

    assert result.returncode == 0, result.stderr
    assert "v8-rbe found nothing" in result.stderr
    assert "rbe.sock" not in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_naming_one_profile_twice_is_not_two_of_it(tmp_path, command):

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

    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_missing_binds_file_is_refused_not_a_traceback(tmp_path, command):

    repo = tmp_path / "repo"
    repo.mkdir()
    result = _cli(
        tmp_path, [command, "--explain", "--binds", "/no/such.toml", str(repo)]
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "binds" in result.stderr and "/no/such.toml" in result.stderr
    assert "Traceback" not in result.stderr


def _write_host_mcp(home: Path, command: str, name: str = "ghost") -> None:
    home.mkdir(exist_ok=True)
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {name: {"type": "stdio", "command": command}}})
    )
    codex = home / ".codex"
    codex.mkdir(exist_ok=True)
    (codex / "config.toml").write_text(f'[mcp_servers.{name}]\ncommand = "{command}"\n')
    opencode = home / ".config" / "opencode"
    opencode.mkdir(parents=True, exist_ok=True)
    (opencode / "opencode.json").write_text(
        json.dumps({"mcp": {name: {"type": "local", "command": [command]}}})
    )


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_uninstalled_mcp_server_is_refused_not_a_traceback(tmp_path, command):

    _write_host_mcp(tmp_path / "home", "aisan-definitely-absent-mcp")
    repo = tmp_path / "repo"
    repo.mkdir()
    binds = tmp_path / "mcp.toml"
    binds.write_text('mcp = ["ghost"]\n')
    result = _cli(tmp_path, [command, "--explain", "--binds", str(binds), str(repo)])

    assert result.returncode == 2, result.stdout + result.stderr
    assert "aisan-definitely-absent-mcp" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_unselected_server_cannot_refuse_the_launch(tmp_path, command):
    """Only a server this box asked for has to resolve on the host."""
    _write_host_mcp(tmp_path / "home", "aisan-definitely-absent-mcp")
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _cli(tmp_path, [command, "--explain", str(repo)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "withheld" in result.stderr
    assert "ghost (aisan-definitely-absent-mcp)" in result.stderr


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_spec_entry_naming_no_declaration_is_a_note_not_a_refusal(tmp_path, command):

    _write_host_mcp(tmp_path / "home", "aisan-definitely-absent-mcp")
    repo = tmp_path / "repo"
    repo.mkdir()
    binds = tmp_path / "mcp.toml"
    binds.write_text('mcp = ["typoed-name"]\n')

    result = _cli(tmp_path, [command, "--explain", "--binds", str(binds), str(repo)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "typoed-name" in result.stderr
    assert "name no local stdio server this host declares" in result.stderr


def test_a_missing_client_binary_is_reported_on_stderr(tmp_path):

    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    empty = tmp_path / "empty-path"
    empty.mkdir()
    drop = {"OPENCODE_CONFIG", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    env["PATH"] = str(empty)
    result = subprocess.run(
        [sys.executable, "-m", "aisan.cli.main", "claude", str(repo)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no `claude` on PATH" in result.stderr
    assert "no `claude` on PATH" not in result.stdout


async def test_the_run_path_routes_an_assembly_refusal_to_return_2(
    tmp_path, monkeypatch, capsys
):

    class _RefusingBox:
        def __init__(self, spec, *, box_id):
            self.spec = spec

        async def __aenter__(self):
            raise FileNotFoundError("bind-over source missing: /gone")

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr("aisan.session.Box", _RefusingBox)
    repo = tmp_path / "repo"
    repo.mkdir()

    code = await run_interactive(
        client="claude",
        harness="claude-code",
        executable="claude",
        repo=repo,
        state=tmp_path / "state",
        spec=object(),
        command=lambda _box: ["claude"],
        binary=lambda: repo / "claude",
        flags=LauncherFlags(),
        explain_only=False,
    )
    assert code == 2
    assert "refused" in capsys.readouterr().err


async def test_the_run_path_normalizes_a_signal_death_to_shell_status(
    tmp_path, monkeypatch
):

    class _Box:
        def __init__(self, spec, *, box_id):
            self.spec = spec
            self.env = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def command(self, cmd):
            return cmd

    class _Spec:
        unshare_net = True

    monkeypatch.setattr("aisan.session.Box", _Box)
    monkeypatch.setattr(
        "aisan.session.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=-15),
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    code = await run_interactive(
        client="claude",
        harness="claude-code",
        executable="claude",
        repo=repo,
        state=tmp_path / "state",
        spec=_Spec(),
        command=lambda _box: ["claude"],
        binary=lambda: repo / "claude",
        flags=LauncherFlags(),
        explain_only=False,
    )
    assert code == 143


@pytest.mark.parametrize(
    ("client", "relative"),
    [
        ("claude", ".credentials.json"),
        ("codex", "auth.json"),
        ("opencode", "opencode/auth.json"),
    ],
)
async def test_a_credential_written_inside_a_box_refuses_the_next_launch(
    tmp_path, monkeypatch, capsys, client, relative
):
    started = False

    class _Box:
        def __init__(self, spec, *, box_id):
            self.spec = spec
            self.env = {}

        async def __aenter__(self):
            nonlocal started
            started = True
            return self

        async def __aexit__(self, *exc):
            return None

        def command(self, cmd):
            return cmd

    class _Spec:
        unshare_net = True

    monkeypatch.setattr("aisan.session.Box", _Box)
    monkeypatch.setattr(
        "aisan.session.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=0),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    state = tmp_path / "state"
    planted = state / relative
    planted.parent.mkdir(parents=True)
    planted.write_text('{"a-real-token": "written in the box"}')

    code = await run_interactive(
        client=client,
        harness=client,
        executable=client,
        repo=repo,
        state=state,
        spec=_Spec(),
        command=lambda _box: [client],
        binary=lambda: repo / client,
        flags=LauncherFlags(),
        explain_only=False,
    )

    assert code == 2
    assert not started
    err = capsys.readouterr().err
    assert str(planted) in err
    assert "delete the file" in err


def test_git_config_binds_only_the_file_never_the_credential_store(
    tmp_path, monkeypatch
):

    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".config" / "git" / "config").write_text("[user]\n  name = Jane\n")
    (home / ".config" / "git" / "credentials").write_text("https://jane:TOKEN@x\n")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    binds = git_config_binds()

    assert binds == [Bind(home / ".config" / "git" / "config", RO)]
    named = {getattr(b, "path", None) or getattr(b, "src", None) for b in binds}
    assert home / ".config" / "git" / "credentials" not in named
    assert home / ".config" / "git" not in named


def test_git_config_binds_is_xdg_aware_on_the_source(tmp_path, monkeypatch):

    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    (xdg / "git" / "config").write_text("[user]\n  name = Jane\n")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    binds = git_config_binds()
    assert binds == [
        BindOver(xdg / "git" / "config", home / ".config" / "git" / "config")
    ]


def test_git_config_binds_empty_without_a_config(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path / "empty"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert git_config_binds() == []


def _fake_depot_tools(tmp_path):
    depot_tools = tmp_path / "depot_tools"
    depot_tools.mkdir()
    autoninja = depot_tools / "autoninja"
    autoninja.write_text("#!/bin/sh\nexit 0\n")
    autoninja.chmod(0o755)
    return depot_tools


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_grant_brings_its_mounts_its_path_and_its_environment(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()
    depot_tools = _fake_depot_tools(tmp_path)

    result = _cli(
        tmp_path,
        [command, "--grant", "depot_tools", "--explain", str(repo)],
        host_path=depot_tools,
    )

    assert result.returncode == 0, result.stderr
    assert f"ro        {depot_tools}" in result.stdout
    assert "DEPOT_TOOLS_UPDATE=0" in result.stdout
    path_line = next(
        line for line in result.stdout.splitlines() if line.strip().startswith("PATH=")
    )
    assert path_line.strip().removeprefix("PATH=").split(os.pathsep)[0] == str(
        depot_tools
    )


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_grant_this_host_lacks_says_so(tmp_path, command):

    repo = tmp_path / "repo"
    repo.mkdir()
    empty = tmp_path / "empty-path"
    empty.mkdir()

    result = _cli(
        tmp_path,
        [command, "--grant", "depot_tools", "--explain", str(repo)],
        host_path=empty,
    )

    assert result.returncode == 0, result.stderr
    assert "grant depot_tools found nothing" in result.stderr
    assert "DEPOT_TOOLS_UPDATE" not in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_a_user_bind_file_still_shadows_a_grant(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()
    depot_tools = _fake_depot_tools(tmp_path)
    spec_file = tmp_path / "mine.toml"
    spec_file.write_text(f'rw = ["{depot_tools}"]\n')

    result = _cli(
        tmp_path,
        [
            command,
            "--grant",
            "depot_tools",
            "--binds",
            str(spec_file),
            "--explain",
            str(repo),
        ],
        host_path=depot_tools,
    )

    assert result.returncode == 0, result.stderr

    assert f"ro-shadow {depot_tools}" in result.stdout
    assert f"rw        {depot_tools}" in result.stdout


@pytest.mark.parametrize("command", ["claude", "codex", "opencode"])
def test_an_unknown_grant_is_refused_by_name(tmp_path, command):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _cli(tmp_path, [command, "--grant", "nope", "--explain", str(repo)])

    assert result.returncode == 2
    assert "nope" in result.stderr and "depot_tools" in result.stderr
