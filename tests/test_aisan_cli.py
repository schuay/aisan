# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aisan.cli import main as cli
from aisan.egress.anthropic import DEFAULT_UPSTREAM


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_dispatcher_passes_every_remaining_token_untouched(monkeypatch, command):
    received = []

    def handler(argv):
        received.append(argv)
        return 17

    monkeypatch.setitem(cli.COMMANDS, command, handler)
    remainder = ["repo", "--binds", "x.toml", "--", "--model", "m", "--help"]
    assert cli.main([command, *remainder]) == 17
    assert received == [remainder]


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_each_subcommand_owns_its_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: aisan {command}")


def test_top_level_help_lists_public_commands(capsys):
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "usage: aisan <command>" in output
    assert all(command in output for command in cli.COMMANDS)


@pytest.mark.parametrize(
    "argv, message",
    [([], "usage: aisan <command>"), (["other"], "unknown command: other")],
)
def test_missing_and_unknown_commands_fail(argv, message, capsys):
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_main_accepts_explicit_argv(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    received = []

    async def run(argv):
        received.append(argv)
        return 23

    monkeypatch.setattr(module, "_main", run)
    argv = ["repo", "--", "--model", "m"]
    assert module.main(argv) == 23
    assert received == [argv]


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_parsers_preserve_explain_and_payload(module_name):
    module = importlib.import_module(module_name)
    args, payload = module.parse_args(
        ["repo", "--explain", "--", "--model", "m", "--help"]
    )
    assert args.repo == "repo"
    assert args.explain is True
    assert payload == ["--model", "m", "--help"]


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_parsers_default_offline_and_accept_net(module_name):
    module = importlib.import_module(module_name)
    default, _ = module.parse_args([])
    enabled, _ = module.parse_args(["--net"])
    assert default.net is False
    assert enabled.net is True


def test_claude_defaults_to_the_anthropic_api(monkeypatch):
    monkeypatch.delenv("AISAN_UPSTREAM", raising=False)
    module = importlib.import_module("aisan.cli.claude")
    args, payload = module.parse_args([])
    assert args.upstream == DEFAULT_UPSTREAM
    assert payload == []


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [("aisan.cli.claude", "CLAUDE.md"), ("aisan.cli.codex", "AGENTS.md")],
)
def test_user_memory_lands_where_the_cli_reads_it(module_name, filename, tmp_path):
    """The state dir is the box's config dir (CLAUDE_CONFIG_DIR, CODEX_HOME),
    which is where each client reads user memory -- a mirror anywhere else, the
    host path included, is a file the box never opens."""
    module = importlib.import_module(module_name)
    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "source.md"
    source.write_text("host memory\n")

    module.seed_user_memory(state, source)
    assert (state / filename).read_text() == "host memory\n"

    # The host file is the source of truth: an in-box edit lasts one session.
    (state / filename).write_text("agent memory\n")
    source.write_text("host memory, edited\n")
    module.seed_user_memory(state, source)
    assert (state / filename).read_text() == "host memory, edited\n"


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [("aisan.cli.claude", "CLAUDE.md"), ("aisan.cli.codex", "AGENTS.md")],
)
def test_a_host_without_user_memory_leaves_the_state_dir_alone(
    module_name, filename, tmp_path
):
    """Nothing here can tell a stale mirror from memory the agent wrote, so a
    missing source seeds nothing rather than deleting the operator's file."""
    module = importlib.import_module(module_name)
    state = tmp_path / "state"
    state.mkdir()

    module.seed_user_memory(state, tmp_path / "absent.md")
    assert not (state / filename).exists()

    (state / filename).write_text("agent memory\n")
    module.seed_user_memory(state, tmp_path / "absent.md")
    assert (state / filename).read_text() == "agent memory\n"


def test_the_seeded_sources_are_the_hosts_own_user_memory():
    """The constants the launchers seed FROM, named once so a rename of either
    host file does not silently turn seeding into a no-op."""
    claude = importlib.import_module("aisan.cli.claude")
    codex = importlib.import_module("aisan.cli.codex")
    assert Path.home() / ".claude" / "CLAUDE.md" == claude.USER_MEMORY
    assert Path.home() / ".codex" / "AGENTS.md" == codex.USER_MEMORY


def test_opencode_mounts_user_memory_where_the_box_reads_it(tmp_path):
    """opencode's config root is the HOME tmpfs, so its global AGENTS.md is a
    bind, and the destination is the box's path rather than the host's -- the
    two differ exactly when the host sets XDG_CONFIG_HOME."""
    opencode = importlib.import_module("aisan.cli.opencode")
    source = tmp_path / "AGENTS.md"
    source.write_text("host memory\n")

    (bind,) = opencode.user_memory_bind(source)
    assert bind.src == source
    assert bind.dst == Path.home() / ".config" / "opencode" / "AGENTS.md"

    assert opencode.user_memory_bind(tmp_path / "absent.md") == []
