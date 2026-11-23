# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive Antigravity CLI session in an externally sandboxed repository.

Usage: aisan agy [repo] [flags] -- [agy args...]

The host must be signed in to agy with a Google account. aisan reads that
login from the host keyring but leaves token refresh and keyring writes to
host agy.

The host's ``~/.gemini`` is not mounted. The box starts past agy's first-run
screens with this repository trusted, and the host's model choice and global
rules files are copied into the box's state on every launch. Host MCP servers
named in a bind spec are imported into the box's ``config/mcp_config.json``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from aisan.egress.antigravity import AntigravityBackend
from aisan.egress.antigravity_keyring import KeyringCredential
from aisan.presets.antigravity import TOKEN_FILE, agy_argv, agy_binary, antigravity
from aisan.session import (
    LaunchRefused,
    git_config_binds,
    git_config_env,
    interactive_parser,
    mcp_launcher_binds,
    mirror_user_memory,
    parse_interactive_args,
    resolve_launcher_flags,
    run_interactive,
    state_dir,
    terminal_env,
    upstream_transport,
)
from aisan.session_mcp import SessionMCP, antigravity_host_mcp, mcp_search_path
from aisan.statedir import ensure_sealed_dir, read_sealed_object, write_sealed

HOST_SETTINGS = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
# agy reads both as global rules, from the directory ``--gemini_dir`` names.
USER_MEMORY = (
    Path.home() / ".gemini" / "GEMINI.md",
    Path.home() / ".gemini" / "AGENTS.md",
)

# What agy 1.5.11 wrote after its colour-scheme and terms screens. With it
# present, the client shows neither; the terms screen would also try to write
# the account's telemetry setting, which the proxy refuses.
ONBOARDING = {
    "consumerOnboardingComplete": True,
    "enterpriseOnboardingComplete": False,
    "onboardingComplete": True,
}


def seed_user_memory(state: Path, source: Path) -> None:
    """Copy a host rules file to the same name in the box's state directory."""
    mirror_user_memory(source, state / source.name)


def host_settings(source: Path) -> dict[str, object]:
    """Copy the host's agy choices that a box should start from.

    A missing or malformed host file contributes nothing.
    """
    try:
        parsed = json.loads(source.read_text())
    except (OSError, ValueError):
        return {}
    model = parsed.get("model") if isinstance(parsed, dict) else None
    return {"model": model} if isinstance(model, str) else {}


def seed_state(state: Path, repo: Path, seed: dict[str, object]) -> None:
    """Write the first-run marker and settings into the box's state directory.

    agy rewrites its settings file, so it is merged rather than bound
    read-only: the host ``seed`` wins, this repository is trusted, and the
    box's other choices stay. The state directory is writable in the box, so
    nothing here is written through a link planted there.
    """
    app = state / TOKEN_FILE.parent
    ensure_sealed_dir(app)
    ensure_sealed_dir(app / "cache")
    write_sealed(app / "cache" / "onboarding.json", json.dumps(ONBOARDING, indent=2))

    path = app / "settings.json"
    settings = {**read_sealed_object(path), **seed}
    trusted = settings.get("trustedWorkspaces")
    trusted = (
        [p for p in trusted if isinstance(p, str)] if isinstance(trusted, list) else []
    )
    if str(repo) not in trusted:
        trusted.append(str(repo))
    settings["trustedWorkspaces"] = trusted
    write_sealed(path, json.dumps(settings, indent=2))


def write_host_mcp(mcp: SessionMCP, state: Path) -> None:
    """Write the imported MCP servers to the config agy reads in the box.

    agy has no flag for a separate MCP config, so the import goes into the
    box's own file and owns ``mcpServers`` outright. It is written even when
    nothing is selected, so a server imported by an earlier launch does not
    start in one whose specs no longer name it. Other keys stay.
    """
    config = state / "config"
    ensure_sealed_dir(config)
    path = config / "mcp_config.json"
    document = {**read_sealed_object(path), **mcp.document}
    replace(mcp, document=document).write(path)


def parse_args(argv: list[str]):
    return parse_interactive_args(interactive_parser("aisan agy", "agy"), argv)


async def _main(argv: list[str]) -> int:
    args, payload = parse_args(argv)
    repo = Path(args.repo).resolve()
    state = state_dir("antigravity", repo)
    backend = AntigravityBackend(
        credential=KeyringCredential(),
        token_file=state / TOKEN_FILE,
        transport=upstream_transport(),
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
    mcp = antigravity_host_mcp(allow=flags.mcp)
    spec = antigravity(
        repo,
        state=state,
        egress=(backend,),
        extra_ro=mcp_launcher_binds(mcp),
        extra_env=(
            # Include mounted home launchers when MCP is enabled.
            ("PATH", mcp_search_path(local_bin=mcp.enabled)),
            *terminal_env(),
        ),
        unshare_net=not args.net,
    )
    # Bind the XDG-aware Git config file without its credential directory.
    spec = spec.with_binds(git_config_binds()).with_env(git_config_env())

    def prepare() -> None:
        for source in USER_MEMORY:
            seed_user_memory(state, source)
        seed_state(state, repo, host_settings(HOST_SETTINGS))
        write_host_mcp(mcp, state)

    return await run_interactive(
        client="antigravity",
        harness="antigravity",
        executable="agy",
        repo=repo,
        state=state,
        spec=spec,
        command=lambda box: agy_argv(tuple(payload), state=state),
        binary=agy_binary,
        flags=flags,
        explain_only=args.explain,
        prepare=prepare,
        mcp=mcp,
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
