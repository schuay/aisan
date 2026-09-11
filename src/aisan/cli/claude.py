# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive Claude Code session in a box, rooted at a git repo.

This is the interactive call site over the same primitive as the headless
preset. Everything here that differs from a
`-p` turn is a consequence of there being a terminal and of the CLI's first-run
path, and each one is noted where it is decided.

**The boundary here is the perimeter, not the repo.** An interactive session is
one the operator chose, on a tree the operator chose, sitting in front of it --
so the box's job is to keep the agent out of the rest of the machine, and the
worktree it was pointed at is the session's to write. The one exception inside
that tree is `.git`'s steering surface, which `gitbinds.git_binds` pins ro for a
plain checkout and a linked worktree alike: the files that steer host-side git
(hooks, configs, object pointers) are untracked, never show in a diff, and run
as the operator the next time host git touches the repo. A LINKED worktree's
`.git` is additionally a store shared with the host and the sibling worktrees,
so the siblings are sealed away too; otherwise an in-box `git gc` or `git
worktree prune` collects a sibling worktree out of the store they share.

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

`--api-key` bills the turn to $ANTHROPIC_API_KEY instead of the host's Claude
Code login. The default is the login, because that is the credential aisan
reads: the box is dressed to match whichever it is, so the in-box client reports
the billing it is actually on rather than the shape of its relay token.

Env: AISAN_CLAUDE_UPSTREAM overrides the Anthropic API base URL (the legacy
AISAN_UPSTREAM is still read, so the naming matches AISAN_CODEX_UPSTREAM and
AISAN_OPENCODE_UPSTREAM).
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
    git_config_binds,
    interactive_parser,
    mcp_launcher_binds,
    mirror_user_memory,
    parse_interactive_args,
    run_interactive,
    state_dir,
    terminal_env,
)
from aisan.session_mcp import claude_config_file, claude_host_mcp, mcp_search_path
from aisan.statedir import read_sealed_object, write_sealed

USER_MEMORY = Path.home() / ".claude" / "CLAUDE.md"


def host_config() -> Path:
    """The host's own Claude Code config, read for the caches it holds.

    A function, not a constant, for the reason `default_credentials` is one: the
    answer depends on CLAUDE_CONFIG_DIR and has to track the environment rather
    than whatever it held at import. `session_mcp` owns the rule -- it seeds the
    MCP servers out of this same file.

    Never bound into the box. The CLI counts this file among the sensitive ones,
    and it holds the whole of the operator's project history besides.
    """
    return claude_config_file()


def seed_state(state: Path, host_config_path: Path | None = None) -> None:
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

    The one other gate here is "Detected a custom API key" (the placeholder
    trips it, defaulting to No), which is about the dress aisan itself chose and
    so presumes nothing about the tree. The bypassPermissions disclaimer is the
    same kind of answer but no longer lives in this file -- see `seed_settings`.

    The folder-trust prompt is deliberately NOT seeded. It asks whether the
    contents of this repository are trusted, which is the operator's call and
    not one a launcher can make for them. The CLI persists the answer into this
    same file, and the merge below leaves it alone, so a box asks once per
    repository and remembers -- inside the state dir only: the host's own
    ~/.claude.json is never read for it and never written.

    The same file carries the caches the box cannot fill for itself, which are
    not gates at all -- see `mirror_host_caches`; they are merged here because
    this is the one writer of the seeded config.

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
    config = read_sealed_object(path)
    config["hasCompletedOnboarding"] = True  # the mandatory one, see above
    responses = _object_at(config, "customApiKeyResponses")
    approved = responses.get("approved")
    if not isinstance(approved, list):
        approved = responses["approved"] = []
    # Only the API-key dress trips this dialog -- measured: with the box holding
    # ANTHROPIC_API_KEY and no seeded approval, the CLI blocks on "Detected a
    # custom API key in your environment", which in an unattended box is a hang.
    # The subscription dress has no key to approve and never sees it. Seeded
    # unconditionally anyway: it costs one list entry, and a state file that is
    # correct for whichever dress the next run picks cannot go stale.
    if PLACEHOLDER_KEY[-20:] not in approved:  # the CLI stores the last 20 chars
        approved.append(PLACEHOLDER_KEY[-20:])
    mirror_host_caches(config, host_config_path or host_config())
    write_sealed(path, json.dumps(config, indent=2))


def seed_settings(state: Path) -> None:
    """Accept the bypassPermissions disclaimer where the CLI reads it today.

    The answer used to be `bypassPermissionsModeAccepted` in `.claude.json`, and
    2.1.259 migrates it on startup: it writes `skipDangerousModePermissionPrompt`
    into user settings -- `$CLAUDE_CONFIG_DIR/settings.json`, so the state dir --
    and then strips the old key from `.claude.json`. Seeding the old key still
    worked, at the cost of that migration running on every single launch, since
    the seed put it back each time. This writes what the CLI reads.

    Not an answer about the tree: the disclaimer is about the permission mode
    this launcher itself passes, which the box is the sandbox for.

    The CLI owns this file too and writes its own keys into it, so this reads
    before writing and changes only its own key -- the same shape as Codex's
    profile file, for the same reason. A file the agent left unparseable is
    rebuilt rather than merged, which loses whatever else was in it; of the two
    ways to be wrong, wedging every later launch of this repo is worse.
    """
    path = state / "settings.json"
    # Read without following a symlink, as with `.claude.json` above: a planted
    # `settings.json -> ~/.claude/settings.json` must not be read here, nor the
    # merged result written back through it.
    settings = read_sealed_object(path)
    settings["skipDangerousModePermissionPrompt"] = True
    write_sealed(path, json.dumps(settings, indent=2))


# The keys Claude Code fills from a server it dials at its compiled-in address,
# with the type each has to have to be worth copying. Both fetches go to
# api.anthropic.com directly -- ANTHROPIC_BASE_URL does not redirect either, so
# no proxy allowlist reaches them and under `unshare_net` both simply fail.
#
#   additionalModelOptionsCache -- the /model picker's rows for models outside
#     the CLI's compiled-in catalog, from GET /api/claude_cli/bootstrap.
#   cachedGrowthBookFeatures and its three companions -- the feature flags, from
#     a remote evaluation POSTed to /api/eval-authed. The CLI writes the four as
#     a unit, so they are copied as one.
#
# The flags matter here because they are EVALUATED PER ACCOUNT: a box that gets
# no answer keeps whatever snapshot it last had, and a date-gated flag whose
# date has since passed then reads as live. Observed: with the host evaluating
# `tengu_saffron_lattice` to {"enabled": false}, a box holding an older copy of
# it announced "Fable 5.1 now uses usage credits" on every model switch. The
# consent that dialog collects cannot be persisted in a box either -- the CLI
# files it under the account uuid from `oauthAccount`, which the box has no
# fetch to fill, and falls back to a latch that dies with the session.
MIRRORED_CACHES = (
    ("additionalModelOptionsCache", list),
    ("cachedGrowthBookFeatures", dict),
    ("cachedGrowthBookFeaturesAt", int),
    ("cachedExperimentFeatures", list),
    ("cachedExperimentData", dict),
)


def mirror_host_caches(config: dict, source: Path) -> None:
    """Carry in the caches the box has no route to fill for itself.

    The host's copies are the answer: they were filled by the same account the
    relay bills to. Copied verbatim, entries the server marked disabled included
    -- an entitlement the host does not have must not become one the box appears
    to. What is then done with them stays the CLI's decision; this only puts
    them where a box with no route to the API can read them.

    Absent, unreadable or wrong-shaped leaves whatever is already there, per key.
    As with user memory, of the two ways to be wrong, clearing a cache the box is
    using is worse than leaving a stale one -- and a host that has never run
    Claude Code has none to lend.
    """
    try:
        parsed = json.loads(source.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(parsed, dict):
        return
    for key, kind in MIRRORED_CACHES:
        value = parsed.get(key)
        # bool passes an int check; no mirrored key is a number the CLI would
        # accept a bool for, so the narrowing is worth the one line.
        if isinstance(value, kind) and not isinstance(value, bool):
            config[key] = value


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
        default=os.environ.get(
            "AISAN_CLAUDE_UPSTREAM",
            os.environ.get("AISAN_UPSTREAM", DEFAULT_UPSTREAM),
        ),
        help="Anthropic API base URL (default: $AISAN_CLAUDE_UPSTREAM, then the"
        f" legacy $AISAN_UPSTREAM, then {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--api-key",
        action="store_true",
        help="bill to the API key in $ANTHROPIC_API_KEY instead of the host's"
        " Claude Code subscription login",
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

    # A flag rather than "use the key if one happens to be exported": which
    # account a turn is billed to is not something to infer from the ambient
    # environment. The key is read from the environment and never from argv --
    # a credential in a command line is visible to every process on the host.
    api_key = None
    if args.api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print(
                "aisan claude: --api-key needs ANTHROPIC_API_KEY set on the host",
                file=sys.stderr,
            )
            return 2
    backend = AnthropicBackend(upstream=args.upstream, api_key=api_key)
    spec = claude_code(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(
            # /usr is bound ro unconditionally, but the preset's PATH is only
            # /usr/bin -- anything elsewhere has to be named.
            ("PATH", mcp_search_path(local_bin=mcp.enabled)),
            *terminal_env(),
            # Undo the preset's headless legibility knob: it is 0 so an
            # unattended box fails fast instead of looking like it is working.
            # At a terminal a transient blip is worth retrying, and either way
            # there is someone watching.
            ("CLAUDE_CODE_MAX_RETRIES", "3"),
        ),
        unshare_net=not args.net,
    )
    # Only the git config FILE, XDG-aware; see session.git_config_binds.
    spec = spec.with_binds(git_config_binds())
    command = ["claude"]
    if mcp.enabled:
        command += ["--mcp-config", str(mcp_config)]
    command += ["--permission-mode", "bypassPermissions", *payload]

    def prepare() -> None:
        seed_state(state)
        seed_settings(state)
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
        grants=args.grant,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
