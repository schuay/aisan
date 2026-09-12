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
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from . import userbinds
from .box import Box
from .egress.base import PreflightError
from .explain import assembly_refusal, explain
from .launch import exit_status
from .presets import EGRESS_PROFILES, GRANTS
from .sandbox import RO, Bind, BindOver, BindSpec
from .session_mcp import SessionMCP, mcp_ro_binds
from .spec import BoxSpec
from .statedir import planted_credentials, prepare_state_dir, write_sealed

_TERM_PASSTHROUGH = ("TERM", "COLORTERM", "LANG", "LC_ALL")


class LaunchRefused(Exception):
    """A launcher refusal and the exit code to report."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def mcp_launcher_binds(mcp: SessionMCP | None) -> tuple[Path, ...]:
    """Resolve MCP launcher binds and convert resolution errors to refusals.

    Resolution fails before the box starts when a command is missing, its tool
    root can't be proven, or a required bind would expose a credential store.
    """
    if mcp is None:
        return ()
    try:
        return mcp_ro_binds(mcp)
    except (FileNotFoundError, ValueError) as e:
        raise LaunchRefused(f"refused: {e}") from e


def git_config_binds() -> list[BindSpec]:
    """Bind the global Git config file without exposing adjacent credentials.

    The file supplies commit identity. Binding its directory could expose the
    plaintext credential store. Resolve the host source through
    ``XDG_CONFIG_HOME``, then bind it to the default path used by the box's
    cleared environment. Return no binds when the host has no global config.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    host = (Path(xdg) if xdg else Path.home() / ".config") / "git" / "config"
    if not host.is_file():
        return []
    box_dst = Path.home() / ".config" / "git" / "config"
    if host.resolve() == box_dst.resolve():
        return [Bind(host, RO)]
    return [BindOver(host, box_dst)]


def repo_key(repo: Path) -> str:
    """Return a stable key that distinguishes repositories with the same name."""
    resolved = repo.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:32]
    name = resolved.name or "root"
    return f"{name}-{digest}"


def state_dir(client: str, repo: Path, *, home: Path | None = None) -> Path:
    """Return the persistent state directory for a client and repository."""
    base = home if home is not None else Path.home()
    return base / ".cache" / f"aisan-{client}" / repo_key(repo)


def mirror_user_memory(source: Path, dst: Path) -> None:
    """Copy host instructions into the boxed client's redirected config.

    The state directory is writable in the box, so ``write_sealed`` prevents a
    planted destination symlink from redirecting the host-side write. Refresh
    the copy at launch, but preserve the destination when the source is absent.
    """
    if source.exists():
        write_sealed(dst, source.read_bytes())


def box_id(client: str, repo: Path) -> str:
    """Return a unique box ID containing a stable repository hint."""
    return f"{client}-i-{repo_key(repo)}-{secrets.token_hex(8)}"


def interactive_parser(prog: str, executable: str) -> argparse.ArgumentParser:
    """Build the common strict argument parser for interactive launchers."""
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
        help="user bind spec, TOML (keys: ro, rw, overlay, path, include); appended"
        " after the preset's binds, so these shadow. Repeatable, applied in"
        " the order given (see aisan.userbinds)",
    )
    parser.add_argument(
        "--egress",
        metavar="NAME",
        action="append",
        choices=sorted(EGRESS_PROFILES),
        help="add a named egress profile's backends and mounts to the box"
        f" (one of: {', '.join(sorted(EGRESS_PROFILES))}). Repeatable; a"
        " profile serving a proxy needs the isolated network, so not with"
        " --net",
    )
    parser.add_argument(
        "--grant",
        metavar="NAME",
        action="append",
        choices=sorted(GRANTS),
        help="add a named grant: the mounts, PATH entries and environment a"
        " tree needs inside a box with no network"
        f" (one of: {', '.join(sorted(GRANTS))}). Repeatable",
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
    """Describe the host MCP declarations copied into the box.

    Name servers that carry environment values because those values may include
    credentials readable by the box.
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
    """Return terminal settings to restore after clearing the environment."""
    return tuple(
        (key, os.environ[key]) for key in _TERM_PASSTHROUGH if key in os.environ
    )


def apply_launcher_flags(
    spec: BoxSpec,
    repo: Path,
    *,
    egress_profiles: list[str] | None,
    grants: list[str] | None,
    binds: list[Path] | None,
) -> BoxSpec:
    """Apply egress profiles, grants, and user bind files to ``spec``.

    Apply them in that order so user bind files can shadow preset and grant
    paths. Report unavailable optional profiles and raise ``LaunchRefused`` for
    invalid configuration.
    """
    # Ignore duplicate names from repeatable flags.
    for name in dict.fromkeys(egress_profiles or []):
        profile = EGRESS_PROFILES[name](repo)
        if not profile:
            # Optional profiles may be unavailable for this repository.
            print(
                f"NOTE: egress profile {name} found nothing to serve in {repo};"
                " the box gets no route from it.",
                file=sys.stderr,
            )
            continue
        # Each backend defines whether its transport is safe on host loopback.
        stranded = [b.name for b in profile.backends if not b.supports_shared_net]
        if stranded and not spec.unshare_net:
            raise LaunchRefused(
                f"--egress {name} needs the box's own network: {', '.join(stranded)}"
                " would otherwise answer on the host's loopback, unauthenticated."
                " Drop --net."
            )
        try:
            spec = spec.with_egress(list(profile.backends))
        except ValueError as e:
            raise LaunchRefused(f"--egress {name}: {e}") from e
        spec = spec.with_binds(list(profile.binds))
        if profile.notice:
            print(profile.notice, file=sys.stderr)

    # Apply each named grant once.
    for name in dict.fromkeys(grants or []):
        grant = GRANTS[name]()
        if not grant:
            # An unavailable optional tool grant doesn't prevent launching.
            print(
                f"NOTE: grant {name} found nothing on this host;"
                " the box gets none of it.",
                file=sys.stderr,
            )
            continue
        spec = (
            spec.with_binds(list(grant.binds))
            .with_path_prefix(grant.path)
            .with_env(grant.env)
        )

    # Later files take precedence, matching bind order within one file.
    for spec_file in binds or []:
        try:
            user = userbinds.load(spec_file, egress=spec.egress)
        except ValueError as e:
            raise LaunchRefused(f"binds: {e}") from e
        except OSError as e:
            # Report unreadable files through the same launcher refusal path.
            raise LaunchRefused(f"binds: cannot read {spec_file}: {e}") from e
        spec = spec.with_binds(user.binds).with_path_prefix(user.path)
    return spec


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
    grants: list[str] | None,
    explain_only: bool,
    prepare: Callable[[], None] | None = None,
    mcp: SessionMCP | None = None,
) -> int:
    """Apply launcher options, then explain or run the box."""
    try:
        spec = apply_launcher_flags(
            spec, repo, egress_profiles=egress_profiles, grants=grants, binds=binds
        )
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code

    box = Box(spec, box_id=box_id(client, repo))
    if explain_only:
        # The report shows mounts; this notice attributes them to MCP imports.
        if mcp is not None and mcp.enabled:
            print(mcp_notice(mcp), file=sys.stderr)
        with staged_directory(state), box.staged():
            print(
                explain(
                    box,
                    inputs=(
                        ("harness", harness),
                        ("repo", str(repo)),
                        ("binds", ", ".join(map(str, binds)) if binds else "(none)"),
                    ),
                    color=sys.stdout.isatty(),
                ),
                end="",
            )
            # Probe while generated bind-over sources still exist.
            refused = assembly_refusal(box) is not None
        return 2 if refused else 0

    if binary() is None:
        print(
            f"no `{executable}` on PATH: this would be an exec failure inside bwrap",
            file=sys.stderr,
        )
        return 2

    # A credential planted in persistent state would enter every later box read-write.
    planted = planted_credentials(state, client)
    if planted:
        print(
            "refused: the state directory holds a credential written inside a"
            f" box: {', '.join(map(str, planted))}\n"
            "aisan never writes these; a session with host networking logged in"
            " from inside the box, and every later box for this repository"
            " would get that credential rw.\n"
            "delete the file, then log in on the HOST if the login is one you"
            " want.",
            file=sys.stderr,
        )
        return 2

    prepare_state_dir(state)
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
            # Convert a negative signal return code to shell exit status.
            return exit_status(done.returncode)
    except PreflightError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as e:
        # Bind sources can disappear and mount resolution can still fail after preflight.
        print(f"refused: {e}", file=sys.stderr)
        return 2


@contextmanager
def staged_directory(path: Path) -> Iterator[None]:
    """Create ``path`` for inspection and remove newly created empty directories.

    Cleanup uses ``rmdir``, so it preserves existing directories and any files
    created concurrently.
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
