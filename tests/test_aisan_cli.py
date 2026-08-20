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


def test_claude_defaults_to_the_anthropic_api(monkeypatch):
    monkeypatch.delenv("AISAN_UPSTREAM", raising=False)
    module = importlib.import_module("aisan.cli.claude")
    args, payload = module.parse_args([])
    assert args.upstream == DEFAULT_UPSTREAM
    assert payload == []
