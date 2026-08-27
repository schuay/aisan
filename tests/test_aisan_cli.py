# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib

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


def test_user_memory_lands_where_the_cli_reads_it(tmp_path):
    """The state dir is CLAUDE_CONFIG_DIR, which is where user memory is read
    from -- a mirror anywhere else is a file the box never opens."""
    claude = importlib.import_module("aisan.cli.claude")
    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "CLAUDE.md"
    source.write_text("host memory\n")

    claude.seed_user_memory(state, source)
    assert (state / "CLAUDE.md").read_text() == "host memory\n"

    # The host file is the source of truth: an in-box edit lasts one session.
    (state / "CLAUDE.md").write_text("agent memory\n")
    source.write_text("host memory, edited\n")
    claude.seed_user_memory(state, source)
    assert (state / "CLAUDE.md").read_text() == "host memory, edited\n"


def test_a_host_without_user_memory_leaves_the_state_dir_alone(tmp_path):
    """Nothing here can tell a stale mirror from memory the agent wrote, so a
    missing source seeds nothing rather than deleting the operator's file."""
    claude = importlib.import_module("aisan.cli.claude")
    state = tmp_path / "state"
    state.mkdir()

    claude.seed_user_memory(state, tmp_path / "absent.md")
    assert not (state / "CLAUDE.md").exists()

    (state / "CLAUDE.md").write_text("agent memory\n")
    claude.seed_user_memory(state, tmp_path / "absent.md")
    assert (state / "CLAUDE.md").read_text() == "agent memory\n"
