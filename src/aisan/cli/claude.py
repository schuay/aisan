# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Launch an interactive Claude Code session inside a repository box.

The worktree is writable, while Git steering files remain read-only and sibling
worktrees remain hidden. Launcher flags are parsed without abbreviations.
Arguments after ``--`` pass directly to Claude Code after the fixed permission
mode. User bind files apply after preset binds.

Usage: aisan claude [repo] [flags] -- [claude args...]

``--api-key`` uses ``ANTHROPIC_API_KEY`` instead of the host Claude Code login.

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
from aisan.session_mcp import claude_config_file, claude_host_mcp, mcp_search_path
from aisan.statedir import read_sealed_object, write_sealed

USER_MEMORY = Path.home() / ".claude" / "CLAUDE.md"
USER_SKILLS = Path.home() / ".claude" / "skills"


def user_skills(source: Path = USER_SKILLS) -> Path | None:
    """Return the host skills directory to mount, or ``None`` when absent.

    Claude Code reads user skills from ``$CLAUDE_CONFIG_DIR/skills``, which the
    box redirects to its state directory, so a bind at the host path is never
    consulted. Skipping a missing source keeps hosts without skills launchable.
    """
    return source if source.is_dir() else None


def host_config() -> Path:
    """Return the environment-dependent host config path used for cache seeding."""
    return claude_config_file()


def seed_state(state: Path, host_config_path: Path | None = None) -> None:
    """Seed first-run state that an isolated session can't obtain itself.

    Claude's onboarding checks fixed Anthropic URLs and exits when the isolated
    network blocks them, so mark onboarding complete. Approve the per-box API
    key placeholder, but preserve the operator's folder-trust choice. Merge host
    caches and tolerate malformed state written by an earlier box.

    Never generate this state by running host Claude with
    ``CLAUDE_CONFIG_DIR`` here; that could store an OAuth credential in a
    directory mounted read-write into the box.
    """
    path = state / ".claude.json"
    # Reject symlinks planted in the box-writable state directory.
    config = read_sealed_object(path)
    config["hasCompletedOnboarding"] = True  # Onboarding can't use the relay.
    responses = _object_at(config, "customApiKeyResponses")
    approved = responses.get("approved")
    if not isinstance(approved, list):
        approved = responses["approved"] = []
    # Without approval, API-key sessions block at the custom-key prompt.
    if PLACEHOLDER_KEY[-20:] not in approved:  # the CLI stores the last 20 chars
        approved.append(PLACEHOLDER_KEY[-20:])
    mirror_host_caches(config, host_config_path or host_config())
    write_sealed(path, json.dumps(config, indent=2))


def seed_settings(state: Path) -> None:
    """Accept the disclaimer for the launcher's fixed permission mode.

    Claude Code 2.1.259 reads this setting from
    ``$CLAUDE_CONFIG_DIR/settings.json``. Preserve other valid settings and
    rebuild malformed state.
    """
    path = state / "settings.json"
    # Reject symlinks planted in the box-writable state directory.
    settings = read_sealed_object(path)
    settings["skipDangerousModePermissionPrompt"] = True
    write_sealed(path, json.dumps(settings, indent=2))


# Claude fetches these values directly from api.anthropic.com, outside the base
# URL override. Copy them from the host account for an isolated session.
#
#   additionalModelOptionsCache -- the /model picker's rows for models outside
#     the CLI's compiled-in catalog, from GET /api/claude_cli/bootstrap.
#   cachedGrowthBookFeatures and its three companions -- the feature flags, from
#     a remote evaluation POSTed to /api/eval-authed. The CLI writes the four as
#     a unit, so they are copied as one.
#
# Feature flags are account-specific. A stale cache caused a disabled
# ``tengu_saffron_lattice`` dialog to reappear on every model switch.
MIRRORED_CACHES = (
    ("additionalModelOptionsCache", list),
    ("cachedGrowthBookFeatures", dict),
    ("cachedGrowthBookFeaturesAt", int),
    ("cachedExperimentFeatures", list),
    ("cachedExperimentData", dict),
)


def mirror_host_caches(config: dict, source: Path) -> None:
    """Copy valid host caches for the account used by the relay.

    Preserve existing values when the host file or individual keys are absent
    or malformed.
    """
    try:
        parsed = json.loads(source.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(parsed, dict):
        return
    for key, kind in MIRRORED_CACHES:
        value = parsed.get(key)
        # bool is an int subclass but isn't valid for the timestamp cache.
        if isinstance(value, kind) and not isinstance(value, bool):
            config[key] = value


def _object_at(config: dict, key: str) -> dict:
    """Return ``config[key]`` as a dict, replacing an invalid value."""
    value = config.get(key)
    if not isinstance(value, dict):
        value = config[key] = {}
    return value


def seed_user_memory(state: Path, source: Path) -> None:
    """Copy host instructions into the box's ``CLAUDE_CONFIG_DIR``."""
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

    # Keep session history outside the worktree.
    state = state_dir("claude", repo)
    mcp_config = state / "aisan-host-mcp.json"

    # Require an explicit billing choice. Read the key from the environment so
    # it doesn't appear in host process arguments.
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
    mcp = claude_host_mcp(allow=flags.mcp)
    spec = claude_code(
        repo,
        state=state,
        egress=(backend,),
        skills=user_skills(),
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(
            # Include mounted home launchers when MCP is enabled.
            ("PATH", mcp_search_path(local_bin=mcp.enabled)),
            *terminal_env(),
            # Interactive sessions can surface progress while retrying.
            ("CLAUDE_CODE_MAX_RETRIES", "3"),
        ),
        unshare_net=not args.net,
    )
    # Bind the XDG-aware Git config file without its credential directory.
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
        flags=flags,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
