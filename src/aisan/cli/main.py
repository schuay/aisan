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

# Out-of-tree commands register here (see README, "Plugin commands"). The group
# is read only when the first token names no built-in and when help is printed,
# so the dispatch a built-in takes is the one it took before plugins existed:
# no metadata scan, no plugin import. An eager merge into COMMANDS would give
# that up, and would also make this CLI's behaviour a function of whatever else
# happens to be installed beside it.
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
    """Plugin commands the environment declares. The seam tests replace."""
    return list(entry_points(group=PLUGIN_GROUP))


def _origin(ep: EntryPoint) -> str:
    """The distribution to name in a diagnostic, or the target it points at."""
    return ep.dist.name if ep.dist is not None else ep.value


def plugin_commands() -> tuple[dict[str, EntryPoint], list[str]]:
    """Resolved plugin commands, and one line per claim that was refused.

    Built-ins are not overridable. `aisan claude` has to mean this package's
    launcher whatever else the environment holds, and installing a plugin pulls
    in its whole dependency closure -- every distribution of which can register
    in this group, though only the plugin itself was ever trusted.

    Two plugins claiming one name resolve by sorting, so the winner is not a
    function of sys.path order. Nothing here is silent: a refused claim would
    otherwise present as a command that is simply missing.
    """
    try:
        found = _installed_entry_points()
    except Exception as e:
        # Unreadable metadata somewhere in the environment. Built-in commands
        # do not depend on this and must survive it.
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
    """Help, with installed plugin commands listed by name and provider.

    Name and provider only: a summary would need a convention for plugins to
    declare one, and that is API surface this does not commit to yet.
    """
    plugins, refused = plugin_commands()
    _report(refused)
    listing = ""
    if plugins:
        entries = "".join(f"  {n:<9} ({_origin(ep)})\n" for n, ep in plugins.items())
        listing = f"\nplugin commands:\n{entries}"
    print(_HELP + listing + _HELP_TAIL, end="", file=stream)


def _load_plugin(command: str) -> Command | None:
    """The named plugin command, or None after saying why there is none."""
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
        # One broken plugin is not an unusable aisan: the import happens here,
        # on the path that asked for it, and takes nothing else down with it.
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
