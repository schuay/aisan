# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Unified human-facing command-line entry point."""

from __future__ import annotations

import sys
from collections.abc import Callable

from aisan import explain
from aisan.session import LaunchRefused

from . import claude, codex, opencode

COMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "claude": claude.main,
    "codex": codex.main,
    "opencode": opencode.main,
    "explain": explain.main,
}

_HELP = """usage: aisan <command> [args...]

commands:
  claude    start an interactive Claude Code session
  codex     start an interactive Codex session
  opencode  start an interactive opencode session
  explain   explain a preset's resolved sandbox profile

Run `aisan <command> --help` for command-specific help.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch the first token and pass every remaining token untouched."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(_HELP, end="", file=sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP, end="")
        return 0

    command = args[0]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"aisan: unknown command: {command}", file=sys.stderr)
        print("Run `aisan --help` for available commands.", file=sys.stderr)
        return 2
    try:
        return handler(args[1:])
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code


if __name__ == "__main__":
    raise SystemExit(main())
