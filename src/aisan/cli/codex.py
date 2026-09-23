# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""An interactive Codex session in an externally sandboxed repository.

Usage: aisan codex [repo] [flags] -- [codex args...]

Env: AISAN_CODEX_MODEL (default: Codex's current default),
AISAN_CODEX_UPSTREAM (default https://chatgpt.com/backend-api/codex). The host
must be logged in with ``codex login``. aisan reads that ChatGPT subscription
login but leaves token refresh and credential-file writes to host Codex.

The host's ``~/.codex/config.toml`` is not mounted. Its status line, model,
reasoning effort and this repository's folder trust are copied into the box's
config on every launch, so a choice made once on the host applies to every
box.
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
from aisan.session_mcp import (
    SessionMCP,
    codex_config_file,
    codex_host_mcp,
    mcp_search_path,
    write_toml_document,
)
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


def read_settings(path: Path) -> dict[str, object]:
    """Parse a Codex config file from the box-writable state directory.

    ``read_sealed_text`` rejects symlinks planted there; a missing or
    unparsable file yields nothing.
    """
    text = read_sealed_text(path)
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def preserved_settings(path: Path) -> dict[str, object]:
    """Read the settings Codex saved to its profile file.

    Codex 0.155 writes the TUI's choices (status line, model, reasoning
    effort, folder trust) to the active profile file, so every top-level
    table except ``mcp_servers`` carries over a rewrite.
    """
    return {k: v for k, v in read_settings(path).items() if k != "mcp_servers"}


def host_settings(source: Path, repo: Path) -> dict[str, object]:
    """Copy the host's Codex choices that a box should start from.

    The box's ``CODEX_HOME`` never sees the host's, so the picks Codex saves
    there (status line, model, reasoning effort, and folder trust for this
    repository) are copied on every launch and win over the box's own. A
    missing or malformed host file contributes nothing.
    """
    try:
        parsed = tomllib.loads(source.read_text())
    except (OSError, ValueError):
        return {}
    seed: dict[str, object] = {}
    for key in ("model", "model_reasoning_effort"):
        if isinstance(parsed.get(key), str):
            seed[key] = parsed[key]
    tui = parsed.get("tui")
    if isinstance(tui, dict):
        line = tui.get("status_line")
        if isinstance(line, list) and all(isinstance(item, str) for item in line):
            seed["tui"] = {"status_line": line}
    projects = parsed.get("projects")
    if isinstance(projects, dict):
        entry = projects.get(str(repo))
        if isinstance(entry, dict) and isinstance(entry.get("trust_level"), str):
            seed["projects"] = {str(repo): {"trust_level": entry["trust_level"]}}
    return seed


def _layered(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    """Merge ``overlay`` into ``base``, recursing into tables both define."""
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        if isinstance(below, dict) and isinstance(value, dict):
            merged[key] = _layered(below, value)
        else:
            merged[key] = value
    return merged


def write_host_mcp(
    mcp: SessionMCP, path: Path, seed: dict[str, object] | None = None
) -> None:
    """Write imported MCP servers while keeping Codex's own profile settings.

    The host ``seed`` wins over the box's saved choices. The host import owns
    ``mcp_servers`` outright, so a stale or planted server is replaced rather
    than merged.
    """
    document = _layered(preserved_settings(path), seed or {})
    replace(mcp, document={**document, **mcp.document}).write(path)


def seed_settings(state: Path, seed: dict[str, object]) -> None:
    """Layer the host ``seed`` onto the box's ``config.toml``.

    Without ``--profile`` that file is the top layer Codex reads, and the one
    it saves choices to. Everything else in it, including servers the box
    added, stays.
    """
    if not seed:
        return
    path = state / "config.toml"
    write_toml_document(path, _layered(read_settings(path), seed))


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
        seed = host_settings(codex_config_file(), repo)
        if mcp.enabled:
            write_host_mcp(mcp, mcp_config, seed)
        else:
            seed_settings(state, seed)

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
