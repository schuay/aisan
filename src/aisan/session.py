# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Shared lifecycle helpers for interactive session launchers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from . import userbinds
from .box import Box
from .egress.base import PreflightError
from .explain import explain
from .presets import EGRESS_PROFILES
from .session_mcp import SessionMCP
from .spec import BoxSpec

_TERM_PASSTHROUGH = ("TERM", "COLORTERM", "LANG", "LC_ALL")


def repo_key(repo: Path) -> str:
    """A readable, stable key that distinguishes equal-basename repositories."""
    resolved = repo.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:32]
    name = resolved.name or "root"
    return f"{name}-{digest}"


def state_dir(client: str, repo: Path, *, home: Path | None = None) -> Path:
    """The persistent state directory for one client and repository."""
    base = home if home is not None else Path.home()
    return base / ".cache" / f"aisan-{client}" / repo_key(repo)


def mirror_user_memory(source: Path, dst: Path) -> None:
    """Copy the host's user-level instructions to where a boxed CLI reads them.

    For the clients whose config directory is redirected at the state dir
    (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`), user memory is read from INSIDE that
    dir -- so the host file is not read wherever it sits, and a ro bind of it
    is the fix that looks right and does nothing: measured, a box holding a
    readable ~/.claude/CLAUDE.md answers that it has no instructions. The
    redirect is not negotiable either, since the CLI writes its config dir
    every turn and the host's holds the credential.

    A copy rather than a mount because the state dir is already the launcher's
    to seed, so this needs no bind and no entry in a user bind spec. Re-copied
    per launch, which makes the host file the source of truth: memory the agent
    writes in-box (the state dir is rw) survives only to the next launch.

    A host with no such file is left alone rather than cleared. Nothing here
    can tell a stale mirror from instructions the agent wrote itself, and of
    the two ways to be wrong, deleting the operator's file is worse than
    leaving a per-repo cache directory holding one.
    """
    if source.exists():
        shutil.copyfile(source, dst)


def box_id(client: str, repo: Path) -> str:
    """A unique live-box identity carrying a stable repository hint."""
    return f"{client}-i-{repo_key(repo)}-{secrets.token_hex(8)}"


def interactive_parser(prog: str, executable: str) -> argparse.ArgumentParser:
    """The common, strictly parsed surface of an interactive launcher."""
    parser = argparse.ArgumentParser(
        prog=prog,
        allow_abbrev=False,
        epilog=f"everything after a literal `--` is forwarded to {executable}.",
    )
    parser.add_argument(
        "repo", nargs="?", default=".", help="git repo to root the box at"
    )
    parser.add_argument(
        "--binds",
        type=Path,
        metavar="FILE",
        action="append",
        help="user bind spec, TOML (keys: ro, rw, overlay, path); appended"
        " after the preset's binds, so these shadow. Repeatable, applied in"
        " the order given (see aisan.userbinds)",
    )
    parser.add_argument(
        "--egress",
        metavar="NAME",
        action="append",
        choices=sorted(EGRESS_PROFILES),
        help="add a named egress profile's backends to the box (one of:"
        f" {', '.join(sorted(EGRESS_PROFILES))}). Repeatable; needs the"
        " isolated network, so not with --net",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the resolved profile and exit; nothing runs",
    )
    parser.add_argument(
        "--net",
        action="store_true",
        help="share host networking (internet, LAN/VPN, and host loopback)",
    )
    return parser


def parse_interactive_args(
    parser: argparse.ArgumentParser, argv: list[str]
) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher args before ``--`` and preserve payload args after it."""
    if "--" not in argv:
        return parser.parse_args(argv), []
    split = argv.index("--")
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def mcp_notice(mcp: SessionMCP) -> str:
    """What the host MCP import puts inside the box, named server by server.

    The import is by operator choice, but it is the one part of a session the
    operator does not spell out at the call site: the declarations come from a
    host config file nobody re-reads at launch, and each is copied VERBATIM --
    so a server whose declaration carries an API token puts that token in a box
    the agent can read, past the credential-absence the rest of the profile
    keeps by subtraction. Everything else that widens the box announces itself
    (`--net` warns, `--binds` is a path the operator typed); this said nothing.

    Named rather than counted, and the environment-carrying ones named again,
    because the action it invites is to go look at a specific declaration.
    """
    lines = [
        (
            f"NOTE: {len(mcp.names)} host MCP server(s) start inside the box:"
            f" {', '.join(mcp.names)}."
        )
    ]
    if mcp.env_names:
        lines.append(
            "      Declared environment values travel with them"
            f" ({', '.join(mcp.env_names)}); any credential there is readable"
            " in the box."
        )
    return "\n".join(lines)


def terminal_env() -> tuple[tuple[str, str], ...]:
    """Terminal variables worth carrying through the box's cleared env."""
    return tuple(
        (key, os.environ[key]) for key in _TERM_PASSTHROUGH if key in os.environ
    )


async def run_interactive(
    *,
    client: str,
    harness: str,
    executable: str,
    repo: Path,
    state: Path,
    spec: BoxSpec,
    command: Callable[[Box], list[str]],
    binary: Callable[[], Path | None],
    binds: list[Path] | None,
    egress_profiles: list[str] | None,
    explain_only: bool,
    prepare: Callable[[], None] | None = None,
    mcp: SessionMCP | None = None,
) -> int:
    """Apply egress profiles and user binds, explain or run, keep the status."""
    # Before the user files, because `userbinds.load` refuses a bind that would
    # expose a backend's credential and can only refuse the backends it is given.
    if egress_profiles and not spec.unshare_net:
        print(
            "--egress needs the box's own network: on the host's loopback the"
            " backend would be an unauthenticated credential capability for"
            " anything on this machine. Drop --net.",
            file=sys.stderr,
        )
        return 2
    for name in egress_profiles or []:
        spec = spec.with_egress(list(EGRESS_PROFILES[name](repo)))

    # In the order given, each file applied whole: later-wins is the model's
    # only precedence rule, so two files compose the same way two entries in
    # one file do -- which is what makes a shared tool spec and a per-project
    # one usable together without either knowing about the other.
    for spec_file in binds or []:
        try:
            user = userbinds.load(spec_file, egress=spec.egress)
        except ValueError as e:
            print(f"binds: {e}", file=sys.stderr)
            return 2
        spec = spec.with_binds(user.binds).with_path_prefix(user.path)

    box = Box(spec, box_id=box_id(client, repo))
    if explain_only:
        with staged_directory(state), box.staged():
            print(
                explain(
                    box,
                    inputs=(
                        ("harness", harness),
                        ("repo", str(repo)),
                        ("binds", ", ".join(map(str, binds)) if binds else "(none)"),
                    ),
                ),
                end="",
            )
        return 0

    if binary() is None:
        print(f"no `{executable}` on PATH: this would be an exec failure inside bwrap")
        return 2

    state.mkdir(parents=True, exist_ok=True)
    if prepare is not None:
        prepare()
    try:
        async with box:
            if not box.spec.unshare_net:
                print(
                    "WARNING: box shares host networking: internet, LAN/VPN, and"
                    " host-local services are reachable; configured model"
                    " credential files are not mounted.",
                    file=sys.stderr,
                )
            if mcp is not None and mcp.enabled:
                print(mcp_notice(mcp), file=sys.stderr)
            done = await asyncio.to_thread(
                subprocess.run,
                box.command(command(box)),
                env={**os.environ, **box.env},
                check=False,
            )
            return done.returncode
    except PreflightError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3


@contextmanager
def staged_directory(path: Path) -> Iterator[None]:
    """Make `path` bindable for inspection, leaving no new empty directories.

    Existing directories are never removed. Cleanup is best-effort and only
    uses rmdir, so a concurrent writer that puts anything in a newly-created
    directory turns it into persistent state rather than losing its files.
    """
    created: list[Path] = []
    cursor = path
    while not cursor.exists():
        created.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        for directory in created:
            with contextlib.suppress(OSError):
                directory.rmdir()
