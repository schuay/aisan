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
from .statedir import prepare_state_dir, write_sealed

_TERM_PASSTHROUGH = ("TERM", "COLORTERM", "LANG", "LC_ALL")


class LaunchRefused(Exception):
    """A launcher-construction refusal carrying the exit code to report it with.

    Raised from the spec-building work a launcher does before `run_interactive`,
    where a bad host config would otherwise traceback out of `_main` with exit 1
    instead of the return-2/return-3 vocabulary every other refusal speaks. The
    dispatcher catches it so the three launchers need not each repeat the catch.
    """

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def mcp_launcher_binds(mcp: SessionMCP | None) -> tuple[Path, ...]:
    """The MCP launcher ro-binds for a preset's `extra_ro`, refusal routed.

    `mcp_ro_binds` raises `FileNotFoundError` when a declared server's command is
    not installed -- by design, so it fails before the box owns the terminal
    rather than as a per-server failure buried inside it. But a host MCP config
    nobody re-reads at launch naming one uninstalled server is a refusal, not a
    bug in aisan: it should reach the operator as a nonzero exit and the server's
    name, the same as any other unmet precondition, not as a stack trace that
    also takes down `--explain`. `ValueError` is the resolver refusing a
    launcher it cannot bind narrowly -- an unproven tool root, or a bind that
    would cover a credential store -- and routes the same way.
    """
    if mcp is None:
        return ()
    try:
        return mcp_ro_binds(mcp)
    except (FileNotFoundError, ValueError) as e:
        raise LaunchRefused(f"refused: {e}") from e


def git_config_binds() -> list[BindSpec]:
    """Bind ONLY the user's global git config FILE, not all of ~/.config/git.

    The launchers bind this so an in-box commit is attributed (user.name/email).
    The whole directory also holds git-credential-store's `credentials` (plaintext
    `https://user:token@host`) and a config may carry `[http] extraHeader` or a
    `[credential] helper` -- secrets the box, which has a model egress route, can
    read and send out. Just the file closes the separate credentials store; a
    secret an operator puts INSIDE config is the residual they chose.

    XDG-aware on the SOURCE side only: the box reads `~/.config/git/config` (its
    cleared env sets no XDG_CONFIG_HOME), while the host keeps the file wherever
    its XDG_CONFIG_HOME points -- so the source is looked up XDG-aware and mounted
    over the path the box reads. Empty when the host has no config: an in-box
    commit then fails "tell me who you are", the honest state rather than a
    silently mounted credential dir.
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

    The write goes through `write_sealed`, not `copyfile`: `dst` sits in the
    box-writable state dir, and a plain copy would follow a symlink the agent
    planted there and overwrite its target as the operator.
    """
    if source.exists():
        write_sealed(dst, source.read_bytes())


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


def apply_launcher_flags(
    spec: BoxSpec,
    repo: Path,
    *,
    egress_profiles: list[str] | None,
    grants: list[str] | None,
    binds: list[Path] | None,
) -> BoxSpec:
    """Apply `--egress`, `--grant` and `--binds` to `spec`, the shared way an
    interactive launcher and `aisan explain` both need them.

    A review that omitted these would describe a box no operator runs -- these
    are the largest thing a caller adds on top of a preset. Prints notices to
    stderr (a profile that found nothing, a profile's own notice) and raises
    `LaunchRefused` carrying the exit code on a hard error, so the CLI that
    called it prints the reason and exits with that code rather than
    tracebacking. The order is the model's: egress first (its backends gate the
    credential guard the user binds run through), then grants, then each bind file
    whole -- so a user file shadows a grant the same way it shadows the preset.
    """
    # Named once each: the flag is repeatable so several profiles compose, and
    # naming one twice is a typo rather than a request for two of it -- which
    # BoxSpec would refuse anyway, on a port collision nobody typed.
    for name in dict.fromkeys(egress_profiles or []):
        profile = EGRESS_PROFILES[name](repo)
        if not profile:
            # A profile whose tree has nothing for it is not a refusal, but it
            # must not be silent either: the operator asked for a route, and a
            # box that quietly has none looks identical to one that works.
            print(
                f"NOTE: egress profile {name} found nothing to serve in {repo};"
                " the box gets no route from it.",
                file=sys.stderr,
            )
            continue
        # Per profile rather than for --egress as a whole, because whether a
        # route survives a shared namespace is a property of its transport: a
        # backend without the authenticated host-loopback path would be a
        # credential capability for everything on the machine, while a profile
        # that is only mounts has no port to expose.
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

    # Named once each, like the egress flag, and applied whole: a grant is
    # aisan's own knowledge about what a tree needs, so there is nothing here
    # for the operator to get subtly wrong except which grants to ask for.
    for name in dict.fromkeys(grants or []):
        grant = GRANTS[name]()
        if not grant:
            # The tree is not on this host. Not a refusal -- the box is still a
            # box -- but silence would look identical to a grant that worked.
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

    # In the order given, each file applied whole: later-wins is the model's
    # only precedence rule, so two files compose the same way two entries in
    # one file do -- which is what makes a shared tool spec and a per-project
    # one usable together without either knowing about the other.
    for spec_file in binds or []:
        try:
            user = userbinds.load(spec_file, egress=spec.egress)
        except ValueError as e:
            raise LaunchRefused(f"binds: {e}") from e
        except OSError as e:
            # A missing or unreadable spec file is the operator's path being
            # wrong (`userbinds.load` lets it propagate for exactly this caller
            # to name); route it like a malformed one rather than tracebacking.
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
    """Apply the composition flags, explain or run, keep the status."""
    try:
        spec = apply_launcher_flags(
            spec, repo, egress_profiles=egress_profiles, grants=grants, binds=binds
        )
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code

    box = Box(spec, box_id=box_id(client, repo))
    if explain_only:
        # The review path discloses what an MCP import dragged in the same way
        # the run path does: the mounts show up in the report, but only this
        # notice says an import is what caused them.
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
            # Inside `staged()`, where bind-over sources exist on disk: probing
            # assembly after it closes would fail every box on a missing source.
            # A refused profile is not a passed review -- `--explain && run` must
            # not read exit 0 off a box that would refuse to assemble.
            refused = assembly_refusal(box) is not None
        return 2 if refused else 0

    if binary() is None:
        print(
            f"no `{executable}` on PATH: this would be an exec failure inside bwrap",
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
            # A box killed by a signal comes back as `-N`; report it the way a
            # shell would, matching the unattended launcher's own status.
            return exit_status(done.returncode)
    except PreflightError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as e:
        # Assembly can still refuse after preflight: staging a bind-over whose
        # source is gone, or `wrapper()` finding the egress binds did not survive
        # resolution. Only the child's status comes back as a return code; these
        # come back as exceptions, so route them rather than let one traceback
        # out with exit 1 next to the refusal convention its siblings speak.
        print(f"refused: {e}", file=sys.stderr)
        return 2


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
