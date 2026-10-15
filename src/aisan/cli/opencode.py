# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Launch an interactive OpenCode session inside a repository box.

The worktree is writable, while Git steering files remain read-only and sibling
worktrees remain hidden. The payload runs in a thread so the host proxy event
loop can serve model requests. Terminal settings survive the cleared
environment.

Custom providers use OpenCode v2.0.29 standalone mode so the service inherits
the box configuration. ``--auto`` approves tools within the box boundary.

Usage:
    aisan opencode [repo] [flags] -- [opencode args...]

Env: AISAN_OPENCODE_PROVIDER (default zai-coding-plan), AISAN_OPENCODE_MODEL
(default glm-5.3), AISAN_OPENCODE_UPSTREAM (default: the provider's api base
from the host's opencode catalog). Flags override env.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from aisan.egress.openai_compat import DEFAULT_CATALOG, OpenAICompatBackend
from aisan.opencode_config import config_file, read_config
from aisan.opencode_provider import resolve_provider
from aisan.presets.opencode import opencode, opencode_binary
from aisan.sandbox import BindOver, BindSpec
from aisan.session import (
    LaunchRefused,
    git_config_binds,
    interactive_parser,
    mcp_launcher_binds,
    parse_interactive_args,
    resolve_launcher_flags,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import mcp_search_path, opencode_host_mcp

# Honor the host's XDG config location for global OpenCode instructions.
USER_MEMORY = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "opencode"
    / "AGENTS.md"
)


def user_memory_bind(source: Path) -> list[BindSpec]:
    """Bind host instructions at the default path used inside the box.

    The host source honors ``XDG_CONFIG_HOME``. The box leaves that variable
    unset to preserve Git config lookup, so use ``BindOver`` when the source
    exists.
    """
    if not source.exists():
        return []
    return [BindOver(source, Path.home() / ".config" / "opencode" / "AGENTS.md")]


def parse_args(argv: list[str]):
    parser = interactive_parser("aisan opencode", "opencode")
    parser.add_argument(
        "--provider",
        default=os.environ.get("AISAN_OPENCODE_PROVIDER", "zai-coding-plan"),
        help="provider id in host config or the opencode catalog"
        " (default: $AISAN_OPENCODE_PROVIDER or zai-coding-plan)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AISAN_OPENCODE_MODEL", "glm-5.3"),
        help="bare model key, selected through inline config"
        " (default: $AISAN_OPENCODE_MODEL or"
        " glm-5.3)",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get("AISAN_OPENCODE_UPSTREAM"),
        help="full API base URL; overrides host config and catalog"
        " (default: $AISAN_OPENCODE_UPSTREAM, host baseURL, then catalog)",
    )
    parser.add_argument(
        "--no-upstream-auth",
        action="store_true",
        help="send no upstream Authorization header; never read auth.json",
    )
    return parse_interactive_args(parser, argv)


async def _main(argv: list[str]) -> int:
    args, payload = parse_args(argv)
    repo = Path(args.repo).resolve()

    # Keep session history outside the worktree.
    state = state_dir("opencode", repo)
    mcp_config = state / "aisan-host-mcp.json"

    source = config_file()
    try:
        document = read_config(source, strict=True)
        selected = resolve_provider(
            args.provider,
            args.model,
            source=source,
            document=document,
            catalog=DEFAULT_CATALOG,
            upstream=args.upstream,
        )
    except (OSError, ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 2
    auth_mode = "none" if args.no_upstream_auth else "api-key"
    backend = OpenAICompatBackend(
        provider=args.provider,
        model=args.model,
        resolved=selected,
        upstream_auth=auth_mode,
    )
    print(
        selected.description(auth_mode), file=sys.stdout if args.explain else sys.stderr
    )
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
    mcp = opencode_host_mcp(source, allow=flags.mcp, document=document)
    spec = opencode(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(
            # Include mounted home launchers when MCP is enabled.
            ("PATH", mcp_search_path(local_bin=mcp.enabled)),
            *((("OPENCODE_CONFIG", str(mcp_config)),) if mcp.enabled else ()),
            *terminal_env(),
        ),
        unshare_net=not args.net,
    )
    # Bind the XDG-aware Git config file without its credential directory.
    spec = spec.with_binds(git_config_binds())
    # A user bind file cannot replace this: a bind-over is a guard, and a user
    # entry at the same path is refused as a conflict.
    spec = spec.with_binds(user_memory_bind(USER_MEMORY))
    return await run_interactive(
        client="opencode",
        harness="opencode",
        executable="opencode",
        repo=repo,
        state=state,
        spec=spec,
        # The outer sandbox provides the tool permission boundary.
        command=lambda _box: [
            "opencode",
            "--auto",
            *(["--standalone"] if selected.custom else []),
            *payload,
        ],
        binary=opencode_binary,
        flags=flags,
        explain_only=args.explain,
        prepare=(lambda: mcp.write(mcp_config)) if mcp.enabled else None,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
