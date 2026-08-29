# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive Claude Code session in a box, rooted at a git repo.

This is the interactive call site over the same primitive as the headless
preset. Everything here that differs from a
`-p` turn is a consequence of there being a terminal and of the CLI's first-run
path, and each one is noted where it is decided.

**The boundary here is the perimeter, not the repo.** An interactive session is
one the operator chose, on a tree the operator chose, sitting in front of it --
so the box's job is to keep the agent out of the rest of the machine, and
everything inside the worktree (`.git` included) is the session's to write. That
is a different model from the unattended job profile: `v8_job` pins `.git`'s
hooks and configs ro through `gitbinds.git_binds`, because there a poisoned hook
runs on the host with nobody watching. Here it is not a boundary violation, it
is the agent using the repo it was pointed at.

The CLI: launcher flags are parsed STRICTLY (no abbreviation -- `--bind` must
not resolve to `--binds` in a tool whose flags gate mounts), and everything
after a literal `--` is forwarded to claude verbatim, after the
`--permission-mode` this wrapper always sets. The split is manual because
argparse's own `--` handling would let the `repo` positional eat a payload
flag. `--binds` names a user bind spec (see `aisan.userbinds`); it is appended
after the preset's binds, so user mounts shadow the template, and it is
repeatable so a shared tool spec and a per-project one compose. `--explain`
prints the resolved profile and exits -- the review path for a merge of several
bind sources, rendered from the same Box the real path uses. `--egress NAME` adds a named egress profile's backends (see
`aisan.presets.EGRESS_PROFILES`), which needs the box's own network and so
refuses alongside `--net`.

Usage: aisan claude [repo] [flags] -- [claude args...]

Env: AISAN_UPSTREAM overrides the Anthropic API base URL.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from aisan.egress.anthropic import DEFAULT_UPSTREAM, PLACEHOLDER_KEY, AnthropicBackend
from aisan.presets.claude_code import claude_code, claude_code_binary
from aisan.session import (
    interactive_parser,
    mcp_launcher_binds,
    mirror_user_memory,
    parse_interactive_args,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import claude_host_mcp, mcp_search_path
from aisan.statedir import read_sealed_text, write_sealed

GIT_CONFIG = Path.home() / ".config" / "git"
USER_MEMORY = Path.home() / ".claude" / "CLAUDE.md"


def seed_state(state: Path, repo: Path) -> None:
    """Answer the CLI's first-run gates, because one of them cannot be answered.

    Onboarding's first step is a connectivity check that fetches
    api.anthropic.com and platform.claude.com DIRECTLY -- not through
    ANTHROPIC_BASE_URL, so the relay does not carry it -- and calls
    process.exit(1) when it fails. Under `unshare_net` it always fails, so
    onboarding cannot be completed from inside the box and the flag has to
    arrive already set. Reproduced: without this the session dies at "Unable to
    connect to Anthropic services / ECONNREFUSED".

    Do NOT produce this file by running `claude` on the host with
    CLAUDE_CONFIG_DIR pointed here. That path can complete an OAuth login and
    write a real credential into a directory bound rw into the box, which is the
    one thing the profile exists to prevent.

    The other three are ordinary dialogs that persist here once answered, so
    seeding them only saves answering them the first time: the folder-trust
    prompt, "Detected a custom API key" (the placeholder trips it, defaulting to
    No), and the bypassPermissions disclaimer.

    The file lives in the box-writable state dir, so its contents are attacker-
    reachable between sessions and this must never propagate a crash: a
    non-object, non-JSON, or wrong-typed nested value rebuilds the offending
    piece rather than raising and wedging every later launch of this repo.
    """
    path = state / ".claude.json"
    # Read without following a symlink: the state dir is box-writable, so a
    # planted `.claude.json -> /etc/passwd` would otherwise be read here (and the
    # merged result written back through the link). A non-regular file reads as
    # absent, rebuilding from scratch.
    existing = read_sealed_text(path)
    try:
        parsed = json.loads(existing) if existing else {}
    except json.JSONDecodeError:
        parsed = {}
    config = parsed if isinstance(parsed, dict) else {}
    config["hasCompletedOnboarding"] = True  # the mandatory one, see above
    config["bypassPermissionsModeAccepted"] = True
    responses = _object_at(config, "customApiKeyResponses")
    approved = responses.get("approved")
    if not isinstance(approved, list):
        approved = responses["approved"] = []
    if PLACEHOLDER_KEY[-20:] not in approved:  # the CLI stores the last 20 chars
        approved.append(PLACEHOLDER_KEY[-20:])
    _object_at(_object_at(config, "projects"), str(repo))["hasTrustDialogAccepted"] = (
        True
    )
    write_sealed(path, json.dumps(config, indent=2))


def _object_at(config: dict, key: str) -> dict:
    """`config[key]` as a dict, replacing a wrong-typed or missing value.

    The state file is box-writable, so any nested value may have been swapped
    for a list, a string, or deleted since it was last seeded; the seed only
    needs its own keys well-formed, so a coercion is the tolerant repair."""
    value = config.get(key)
    if not isinstance(value, dict):
        value = config[key] = {}
    return value


def seed_user_memory(state: Path, source: Path) -> None:
    """Mirror the host's user memory into the state dir, which is the config
    dir the box reads it from -- `CLAUDE_CONFIG_DIR`, never ~/.claude. See
    `mirror_user_memory` for why this is a copy and not a bind."""
    mirror_user_memory(source, state / "CLAUDE.md")


def parse_args(argv: list[str]):
    parser = interactive_parser("aisan claude", "claude")
    parser.add_argument(
        "--upstream",
        default=os.environ.get("AISAN_UPSTREAM", DEFAULT_UPSTREAM),
        help=f"Anthropic API base URL (default: $AISAN_UPSTREAM or {DEFAULT_UPSTREAM})",
    )
    return parse_interactive_args(parser, argv)


async def _main(argv: list[str]) -> int:
    args, payload = parse_args(argv)
    repo = Path(args.repo).resolve()

    # Outside the repo, deliberately: this is the agent's own session history,
    # and claude_code_default only puts it inside the root because a dry run has
    # no job to own it.
    state = state_dir("claude", repo)
    mcp = claude_host_mcp()
    mcp_config = state / "aisan-host-mcp.json"

    backend = AnthropicBackend(upstream=args.upstream)
    spec = claude_code(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=(GIT_CONFIG, *mcp_launcher_binds(mcp)),
        extra_env=(
            # /usr is bound ro unconditionally, but the preset's PATH is only
            # /usr/bin -- anything elsewhere has to be named.
            ("PATH", mcp_search_path()),
            *terminal_env(),
            # Undo the preset's headless legibility knob: it is 0 so an
            # unattended box fails fast instead of looking like it is working.
            # At a terminal a transient blip is worth retrying, and either way
            # there is someone watching.
            ("CLAUDE_CODE_MAX_RETRIES", "3"),
        ),
        unshare_net=not args.net,
    )
    command = ["claude"]
    if mcp.enabled:
        command += ["--mcp-config", str(mcp_config)]
    command += ["--permission-mode", "bypassPermissions", *payload]

    def prepare() -> None:
        seed_state(state, repo)
        seed_user_memory(state, USER_MEMORY)
        if mcp.enabled:
            mcp.write(mcp_config)

    return await run_interactive(
        client="claude",
        harness="claude-code",
        executable="claude",
        repo=repo,
        state=state,
        spec=spec,
        command=lambda _box: command,
        binary=claude_code_binary,
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
