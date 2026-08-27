# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive opencode session in a box, rooted at a git repo.

The sibling of the Claude session launcher, and most of that launcher's
reasoning carries across unchanged: the boundary is the perimeter, not
the repo (an interactive session on a tree the operator chose, so the whole
worktree is the session's to write and `gitbinds` pinning is deliberately
absent); the host half of egress is an aiohttp server on this process's event
loop, so the payload waits in a thread or every model call hangs; and TERM
travels with the caller because a TUI in a cleared environment renders as a
dumb terminal.

What does NOT carry across: the first-run seeding. Claude Code's onboarding
begins with a connectivity check that ignores the proxy and exits on failure,
so its flag has to arrive already set. opencode has no such gate -- measured,
a headless turn against a stub upstream ran first-try on a clean XDG surface,
and anything the TUI asks on first run (a theme, a trust prompt) is a dialog
that persists in the state dir once answered, not a dead end.

The model rides the backend's inline config rather than argv because the TUI
takes no `--model` flag (measured, 1.18.18: only `opencode run` has one).
The launcher also passes opencode's `--auto` flag so tool permissions are
approved automatically during a session.

The CLI: launcher flags are parsed STRICTLY (no abbreviation -- `--bind` must
not resolve to `--binds` in a tool whose flags gate mounts), and everything
after a literal `--` is forwarded to opencode verbatim. The split is manual
because argparse's own `--` handling would let the `repo` positional eat a
payload flag. `--binds` names a user bind spec (see `aisan.userbinds`), and is
repeatable so a shared tool spec and a per-project one compose; it is
appended after the preset's binds, so user mounts shadow the template.
`--explain` prints the resolved profile and exits -- the review path for a
merge of several bind sources, rendered from the same Box the real path uses.

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

from aisan.egress.openai_compat import OpenAICompatBackend
from aisan.presets.opencode import opencode, opencode_binary
from aisan.session import (
    interactive_parser,
    parse_interactive_args,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import mcp_ro_binds, mcp_search_path, opencode_host_mcp

# HOME is a tmpfs in the box, so anything under it the payload needs must be
# named. Git's identity lives here; without it a commit inside the box dies
# with "Please tell me who you are". ro: the agent reads config, it does not
# edit it.
GIT_CONFIG = Path.home() / ".config" / "git"


def parse_args(argv: list[str]):
    parser = interactive_parser("aisan opencode", "opencode")
    parser.add_argument(
        "--provider",
        default=os.environ.get("AISAN_OPENCODE_PROVIDER", "zai-coding-plan"),
        help="provider id in the opencode catalog"
        " (default: $AISAN_OPENCODE_PROVIDER or zai-coding-plan)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AISAN_OPENCODE_MODEL", "glm-5.3"),
        help="bare model id; rides the backend's inline config, since the"
        " TUI takes no --model flag (default: $AISAN_OPENCODE_MODEL or"
        " glm-5.3)",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get("AISAN_OPENCODE_UPSTREAM"),
        help="the provider's full api base URL; default resolves it from the"
        " host's opencode catalog ($AISAN_OPENCODE_UPSTREAM overrides)",
    )
    return parse_interactive_args(parser, argv)


async def _main(argv: list[str]) -> int:
    args, payload = parse_args(argv)
    repo = Path(args.repo).resolve()

    # Outside the repo, deliberately: this is the agent's own session history,
    # and the preset only puts it inside the root when a dry run has no job to
    # own it.
    state = state_dir("opencode", repo)
    mcp = opencode_host_mcp()
    mcp_config = state / "aisan-host-mcp.json"

    backend = OpenAICompatBackend(
        provider=args.provider, model=args.model, upstream=args.upstream
    )
    spec = opencode(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=(GIT_CONFIG, *mcp_ro_binds(mcp)),
        extra_env=(
            # /usr is bound ro unconditionally, but the preset's PATH is only
            # /usr/bin -- anything elsewhere has to be named.
            ("PATH", mcp_search_path()),
            *((("OPENCODE_CONFIG", str(mcp_config)),) if mcp.enabled else ()),
            *terminal_env(),
        ),
        unshare_net=not args.net,
    )
    return await run_interactive(
        client="opencode",
        harness="opencode",
        executable="opencode",
        repo=repo,
        state=state,
        spec=spec,
        # The sandbox is the permission boundary for this launcher; opencode's
        # own approval prompts would otherwise interrupt tool use in the TUI.
        command=lambda _box: ["opencode", "--auto", *payload],
        binary=opencode_binary,
        binds=args.binds,
        explain_only=args.explain,
        prepare=(lambda: mcp.write(mcp_config)) if mcp.enabled else None,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
