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
    git_config_binds,
    interactive_parser,
    mcp_launcher_binds,
    mirror_user_memory,
    parse_interactive_args,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import SessionMCP, codex_host_mcp, mcp_search_path
from aisan.statedir import read_sealed_text

USER_MEMORY = Path.home() / ".codex" / "AGENTS.md"


def seed_user_memory(state: Path, source: Path) -> None:
    """Mirror the host's global AGENTS.md into the state dir, which is the
    config dir the box reads it from -- `CODEX_HOME`, never ~/.codex.
    Measured against codex-cli 0.147: the file is read from $CODEX_HOME.
    See `mirror_user_memory` for why this is a copy and not a bind."""
    mirror_user_memory(source, state / "AGENTS.md")


def preserved_trust(path: Path) -> dict[str, object]:
    """Carry Codex's own answer to the folder-trust prompt across a rewrite.

    The launcher rewrites the profile file from the host MCP import on every
    launch, and Codex persists that answer INTO the file the active profile
    names -- measured against codex-cli 0.153.4: with `--profile NAME` the entry
    lands in `$CODEX_HOME/NAME.config.toml`, and only without one does it land
    in `config.toml`. So the truncating write asked the question again every
    session. Nothing here answers it; this only stops the import from erasing
    the answer the operator gave.

    Trust stays inside the state dir. It is not read from the host's ~/.codex
    and never written back there, so an entry an agent could have authored
    steers nothing outside its own box.

    TODO: only the trust field survives -- any OTHER setting Codex persists into
    the profile file is still dropped. 0.153.4 writes nothing else there (model,
    reasoning effort and the tui keys go to config.toml, which aisan does not
    rewrite), so today that costs nothing. A later version that persists more
    here would show up as a second setting that will not stick.

    Read without following a symlink, like every other read of this dir: a
    planted `aisan-host-mcp.config.toml -> ~/.codex/config.toml` must not be
    read through, which would copy the host's file into the box. A planted link
    reads as absent and costs one more trust prompt.
    """
    text = read_sealed_text(path)
    if text is None:
        return {}
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    projects = parsed.get("projects")
    if not isinstance(projects, dict):
        return {}
    # Every path the file names, not just this box's root: Codex keys the entry
    # on the directory it was started in, and a box rooted at a subdirectory
    # would otherwise silently stop persisting. One field of each is narrow
    # enough -- an entry for another path steers only a later box on that path.
    kept = {
        name: {"trust_level": entry["trust_level"]}
        for name, entry in projects.items()
        if isinstance(entry, dict) and isinstance(entry.get("trust_level"), str)
    }
    return {"projects": kept} if kept else {}


def write_host_mcp(mcp: SessionMCP, path: Path) -> None:
    """Rewrite the profile file from the host import, keeping what Codex wrote.

    The import owns `mcp_servers` outright -- a server the agent planted in this
    file is replaced, not merged. `preserved_trust` names the one thing that
    survives, and why.
    """
    replace(mcp, document={**mcp.document, **preserved_trust(path)}).write(path)


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
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(("PATH", mcp_search_path(local_bin=mcp.enabled)), *terminal_env()),
        unshare_net=not args.net,
    )
    # Only the git config FILE, XDG-aware; see session.git_config_binds.
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
        binds=args.binds,
        egress_profiles=args.egress,
        grants=args.grant,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
