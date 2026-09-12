#!/usr/bin/env python3
# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Build reviewable Claude Code profiles for one or more repositories.

Each function returns a ``BoxSpec`` without starting a box. One repository is
the writable root and cwd; additional repositories use explicit binds.

    aisan-claude-specs.py                 # list the variants
    aisan-claude-specs.py multi_repo      # print that one's mount policy

`AISAN_UPSTREAM` overrides the standard Anthropic API base URL, matching the
`aisan claude` command.

Linked worktrees need ``git_binds`` because their ``.git`` file points outside
the worktree. Without those binds, ``git rev-parse`` failed with "not a git
repository: (null)" in a real box. Plain checkouts produce no extra Git binds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aisan.egress.anthropic import DEFAULT_UPSTREAM, AnthropicBackend
from aisan.gitbinds import external_symlink_targets, git_binds
from aisan.presets.claude_code import claude_code
from aisan.sandbox import RO, RW, Bind, BindSpec
from aisan.session import git_config_binds, repo_key
from aisan.spec import BoxSpec

UPSTREAM = os.environ.get(
    "AISAN_CLAUDE_UPSTREAM", os.environ.get("AISAN_UPSTREAM", DEFAULT_UPSTREAM)
)


def _state_for(name: str) -> Path:
    """Create persistent session state outside the repositories.

    The box can write this directory, and later sessions read its history and
    instructions.
    """
    state = Path.home() / ".cache" / "aisan-claude" / name
    state.mkdir(parents=True, exist_ok=True)
    return state


def _repo_binds(repo: Path, mode) -> list[BindSpec]:
    """Bind a checkout before its read-only Git steering pins."""
    return [Bind(repo, mode), *git_binds(repo)]


def single_repo(repo: Path) -> BoxSpec:
    """Build the single-repository baseline used by the other variants."""
    # Bind Git identity without exposing the adjacent credential store.
    return claude_code(
        repo,
        # Include the resolved-path hash so same-named repositories don't share state.
        state=_state_for(repo_key(repo)),
        egress=(AnthropicBackend(upstream=UPSTREAM),),
        extra_env=(
            ("PATH", "/usr/bin:/usr/local/bin"),
            ("CLAUDE_CODE_MAX_RETRIES", "3"),
        ),
    ).with_binds(git_config_binds())


def multi_repo(primary: Path, *others: Path) -> BoxSpec:
    """Bind several writable repositories with cwd in ``primary``.

    A real-box test covered two plain checkouts and one linked worktree. All
    three supported Git and writes, while an unnamed sibling was absent.
    """
    binds: list[BindSpec] = []
    for other in others:
        binds += _repo_binds(other, RW)
    return single_repo(primary).with_binds(binds)


def reference_repos(primary: Path, *readonly: Path) -> BoxSpec:
    """Bind ``primary`` writable and the remaining repositories read-only.

    Read-only repositories can't run Git operations that take locks, including
    commit, stash, and GC.
    """
    binds: list[BindSpec] = []
    for repo in readonly:
        binds += _repo_binds(repo, RO)
    return single_repo(primary).with_binds(binds)


def v8_worktrees(primary: Path, *others: Path) -> BoxSpec:
    """Bind V8 worktrees and their shared dependencies.

    Each worktree receives access to the common Git store with steering files
    pinned read-only and unselected siblings hidden. External gclient symlink
    targets are deduplicated and mounted read-only because all worktrees share
    them.

    These dependency mounts also cover matching paths inside a writable main
    checkout. A real-box test found the main checkout writable while its
    ``build/`` dependency was read-only. Use a single-repository profile for
    DEPS rolls or ``gclient sync``.
    """
    binds: list[BindSpec] = []
    targets: set[Path] = set()
    for wt in others:
        binds += _repo_binds(wt, RW)
    for wt in (primary, *others):
        targets.update(external_symlink_targets(wt))
    # Stable order keeps profile diffs reproducible.
    binds += [Bind(t, RO, optional=True) for t in sorted(targets)]
    return single_repo(primary).with_binds(binds)


def with_reference_docs(spec: BoxSpec, *docs: Path) -> BoxSpec:
    """Add optional read-only reference directories to ``spec``."""
    return spec.with_binds([Bind(d, RO, optional=True) for d in docs])


# Every variant accepts ``primary, *others`` for uniform CLI dispatch.
VARIANTS = {
    "single_repo": single_repo,
    "multi_repo": multi_repo,
    "reference_repos": reference_repos,
    "v8_worktrees": v8_worktrees,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        print(__doc__.strip().split("\n\n")[0])
        print("\nusage: aisan-claude-specs.py <variant> [primary] [others...]")
        print("\nvariants:")
        for name, fn in VARIANTS.items():
            summary = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:18s} {summary}")
        return 0 if len(sys.argv) < 2 else 2

    fn = VARIANTS[sys.argv[1]]
    repos = [Path(a).resolve() for a in sys.argv[2:]] or [Path.cwd()]
    spec = fn(*repos)
    print(f"root (rw, cwd): {spec.root}")
    print(f"egress:         {[b.name for b in spec.egress]}")
    print(f"unshare_net:    {spec.unshare_net}")
    print(f"binds ({len(spec.binds)}), in mount order -- later wins:")
    for b in spec.binds:
        print(f"  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
