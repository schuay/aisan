# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Unified human-facing command-line entry point."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import TextIO

from aisan import explain
from aisan.session import LaunchRefused

from . import claude, codex, opencode

Command = Callable[[list[str] | None], int]

COMMANDS: dict[str, Command] = {
    "claude": claude.main,
    "codex": codex.main,
    "opencode": opencode.main,
    "explain": explain.main,
}

# Read plugin metadata only for help or an unknown command. Built-in dispatch
# must remain independent of other installed distributions.
PLUGIN_GROUP = "aisan.commands"

_HELP = """usage: aisan <command> [args...]

commands:
  claude    start an interactive Claude Code session
  codex     start an interactive Codex session
  opencode  start an interactive opencode session
  explain   explain a preset's resolved sandbox profile
"""

_HELP_TAIL = "\nRun `aisan <command> --help` for command-specific help.\n"


def _installed_entry_points() -> list[EntryPoint]:
    """Return plugin commands declared by installed distributions."""
    return list(entry_points(group=PLUGIN_GROUP))


def _origin(ep: EntryPoint) -> str:
    """Return the plugin distribution name or entry-point target."""
    return ep.dist.name if ep.dist is not None else ep.value


def plugin_commands() -> tuple[dict[str, EntryPoint], list[str]]:
    """Resolve plugin commands and report rejected name claims.

    Plugins can't override built-ins. Sort duplicate claims by provider so the
    winner doesn't depend on ``sys.path`` order.
    """
    try:
        found = _installed_entry_points()
    except Exception as e:
        # Broken package metadata must not disable built-in commands.
        return {}, [f"cannot read plugin commands: {e}"]

    resolved: dict[str, EntryPoint] = {}
    refused: list[str] = []
    for ep in sorted(found, key=lambda ep: (ep.name, _origin(ep))):
        if ep.name in COMMANDS:
            refused.append(
                f"{_origin(ep)} cannot override built-in command {ep.name!r}"
            )
        elif ep.name in resolved:
            refused.append(
                f"{_origin(ep)} lost command {ep.name!r} to {_origin(resolved[ep.name])}"
            )
        else:
            resolved[ep.name] = ep
    return resolved, refused


def _report(refused: list[str]) -> None:
    for line in refused:
        print(f"aisan: {line}", file=sys.stderr)


def _print_help(stream: TextIO) -> None:
    """Print built-in help and installed plugin names and providers."""
    plugins, refused = plugin_commands()
    _report(refused)
    listing = ""
    if plugins:
        entries = "".join(f"  {n:<9} ({_origin(ep)})\n" for n, ep in plugins.items())
        listing = f"\nplugin commands:\n{entries}"
    print(_HELP + listing + _HELP_TAIL, end="", file=stream)


def _load_plugin(command: str) -> Command | None:
    """Load a plugin command, reporting lookup and import failures."""
    plugins, refused = plugin_commands()
    _report(refused)
    ep = plugins.get(command)
    if ep is None:
        print(f"aisan: unknown command: {command}", file=sys.stderr)
        print("Run `aisan --help` for available commands.", file=sys.stderr)
        return None
    try:
        return ep.load()
    except Exception as e:
        # A broken plugin affects only its own command.
        print(
            f"aisan: plugin command {command!r} from {_origin(ep)} "
            f"failed to load: {e!r}",
            file=sys.stderr,
        )
        return None


def main(argv: list[str] | None = None) -> int:
    """Dispatch the first token and pass every remaining token untouched."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        _print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return 0

    command = args[0]
    handler = COMMANDS.get(command)
    if handler is None:
        handler = _load_plugin(command)
    if handler is None:
        return 2
    try:
        return handler(args[1:])
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code


if __name__ == "__main__":
    raise SystemExit(main())
