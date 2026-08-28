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
from pathlib import Path

from aisan.egress.openai_responses import (
    DEFAULT_UPSTREAM,
    CodexBackend,
)
from aisan.presets.codex import codex, codex_argv, codex_binary
from aisan.session import (
    interactive_parser,
    mirror_user_memory,
    parse_interactive_args,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import codex_host_mcp, mcp_ro_binds, mcp_search_path

GIT_CONFIG = Path.home() / ".config" / "git"
USER_MEMORY = Path.home() / ".codex" / "AGENTS.md"


def seed_user_memory(state: Path, source: Path) -> None:
    """Mirror the host's global AGENTS.md into the state dir, which is the
    config dir the box reads it from -- `CODEX_HOME`, never ~/.codex.
    Measured against codex-cli 0.147: the file is read from $CODEX_HOME.
    See `mirror_user_memory` for why this is a copy and not a bind."""
    mirror_user_memory(source, state / "AGENTS.md")


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
    mcp = codex_host_mcp()
    mcp_config = state / "aisan-host-mcp.config.toml"
    backend = CodexBackend(model=args.model, upstream=args.upstream)
    spec = codex(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=(GIT_CONFIG, *mcp_ro_binds(mcp)),
        extra_env=(("PATH", mcp_search_path()), *terminal_env()),
        unshare_net=not args.net,
    )
    payload_args = tuple(payload)
    if mcp.enabled:
        payload_args = ("--profile", "aisan-host-mcp", *payload_args)

    def prepare() -> None:
        seed_user_memory(state, USER_MEMORY)
        if mcp.enabled:
            mcp.write(mcp_config)

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
        binds=args.binds,
        egress_profiles=args.egress,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
