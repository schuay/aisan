# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive Codex session in an externally sandboxed repository.

Usage: aisan codex [repo] [flags] -- [codex args...]

Env: AISAN_CODEX_MODEL (default: Codex's current default),
AISAN_CODEX_UPSTREAM (default https://chatgpt.com/backend-api/codex). The host
must be logged in with ``codex login``. aisan reads that ChatGPT subscription
login but leaves token refresh and credential-file writes to host Codex.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from aisan.egress.openai_responses import (
    DEFAULT_UPSTREAM,
    CodexBackend,
)
from aisan.presets.codex import codex, codex_argv, codex_binary
from aisan.session import (
    LaunchRefused,
    git_config_binds,
    interactive_parser,
    mcp_launcher_binds,
    mirror_user_memory,
    parse_interactive_args,
    resolve_launcher_flags,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import SessionMCP, codex_host_mcp, mcp_search_path
from aisan.statedir import read_sealed_text

USER_MEMORY = Path.home() / ".codex" / "AGENTS.md"
# Codex scans both as user-level skill roots. The first is where its own
# skill-installer writes, so it wins when a host has skills in both.
USER_SKILLS = (
    Path.home() / ".codex" / "skills",
    Path.home() / ".agents" / "skills",
)


def user_skills(sources: tuple[Path, ...] = USER_SKILLS) -> Path | None:
    """Return the host skills directory to mount, or ``None`` when absent.

    ``CODEX_HOME`` is the box's state directory, so host skills below the
    host's ``~/.codex`` are never read. Only one source can be mounted, because
    the box exposes a single directory at the root Codex reads.
    """
    return next((s for s in sources if s.is_dir()), None)


def seed_user_memory(state: Path, source: Path) -> None:
    """Copy the host ``AGENTS.md`` into the box's ``CODEX_HOME``."""
    mirror_user_memory(source, state / "AGENTS.md")


def preserved_settings(path: Path) -> dict[str, object]:
    """Read the settings Codex saved to its profile file.

    Codex 0.155 writes the TUI's choices (status line, model, reasoning
    effort, folder trust) to the active profile file, so every top-level
    table except ``mcp_servers`` carries over a rewrite. ``read_sealed_text``
    rejects symlinks planted in the box-writable state directory; a missing
    or unparsable file yields nothing.
    """
    text = read_sealed_text(path)
    if text is None:
        return {}
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    return {key: value for key, value in parsed.items() if key != "mcp_servers"}


def write_host_mcp(mcp: SessionMCP, path: Path) -> None:
    """Write imported MCP servers while keeping Codex's own profile settings.

    The host import owns ``mcp_servers`` outright, so a stale or planted
    server is replaced rather than merged.
    """
    replace(mcp, document={**preserved_settings(path), **mcp.document}).write(path)


def parse_args(argv: list[str]):
    parser = interactive_parser("aisan codex", "codex")
    parser.add_argument(
        "--model",
        default=os.environ.get("AISAN_CODEX_MODEL", ""),
        help="model id passed as a Codex CLI override"
        " (default: $AISAN_CODEX_MODEL or Codex's current default)",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get("AISAN_CODEX_UPSTREAM", DEFAULT_UPSTREAM),
        help="the full Responses API base URL"
        " (default: $AISAN_CODEX_UPSTREAM or ChatGPT Codex)",
    )
    return parse_interactive_args(parser, argv)


async def _main(argv: list[str]) -> int:
    args, payload = parse_args(argv)
    repo = Path(args.repo).resolve()
    state = state_dir("codex", repo)
    mcp_config = state / "aisan-host-mcp.config.toml"
    backend = CodexBackend(model=args.model, upstream=args.upstream)
    try:
        flags = resolve_launcher_flags(
            repo,
            base_egress=(backend,),
            unshare_net=not args.net,
            egress_profiles=args.egress,
            grants=args.grant,
            binds=args.binds,
        )
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code
    # The bind specs decide which host declarations this box may start.
    mcp = codex_host_mcp(allow=flags.mcp)
    spec = codex(
        repo,
        state=state,
        egress=(backend,),
        skills=user_skills(),
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(("PATH", mcp_search_path(local_bin=mcp.enabled)), *terminal_env()),
        unshare_net=not args.net,
    )
    # Bind the XDG-aware Git config file without its credential directory.
    spec = spec.with_binds(git_config_binds())
    payload_args = tuple(payload)
    if mcp.enabled:
        payload_args = ("--profile", "aisan-host-mcp", *payload_args)

    def prepare() -> None:
        seed_user_memory(state, USER_MEMORY)
        if mcp.enabled:
            write_host_mcp(mcp, mcp_config)

    return await run_interactive(
        client="codex",
        harness="codex",
        executable="codex",
        repo=repo,
        state=state,
        spec=spec,
        command=lambda box: codex_argv(
            payload_args,
            overrides=backend.config_overrides(port=box.activation(backend).port),
        ),
        binary=codex_binary,
        flags=flags,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
