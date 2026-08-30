#!/usr/bin/env python3
# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Spec variants for an interactive Claude Code session, above one repo or many.

`aisan claude` is the single-repo call site: one worktree rw, and the
box's cwd is it. Everything here answers the question that starts where that one
stops -- what a session looks like when the work spans several checkouts, or
when some of them are reference material the agent should not edit.

Each function returns a `BoxSpec`. None of them runs anything: composing the
spec and running a box are separate so a variant can be printed, diffed and
reviewed before it owns a terminal. `main` builds one and prints its binds.

    aisan-claude-specs.py                 # list the variants
    aisan-claude-specs.py multi_repo      # print that one's mount policy

`AISAN_UPSTREAM` overrides the standard Anthropic API base URL, matching the
`aisan claude` command.

## The two facts that shape all of these

**The root is special, the rest are binds.** `BoxSpec.root` is bound rw at its
real absolute path AND is the box's cwd; there is exactly one. A second repo is
therefore not a second root, it is a `Bind` -- which is why every multi-repo
variant here picks one repo as the place the agent starts and adds the others
beside it.

**A linked worktree needs `git_binds`, and fails loudly without it.** A worktree
created by `git worktree add` has a `.git` FILE pointing at
`<main>/.git/worktrees/<name>`, so binding the worktree alone binds a pointer to
something absent. Measured, in a box with the worktree bound rw and nothing
else: `git -C <worktree> rev-parse` answers `fatal: not a git repository:
(null)`. With `git_binds(wt)` appended: the branch name. A plain (non-worktree)
checkout has its `.git` inside the tree already and needs none of this --
`git_binds` returns an empty list for one, so calling it is always safe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aisan.egress.anthropic import DEFAULT_UPSTREAM, AnthropicBackend
from aisan.gitbinds import external_symlink_targets, git_binds
from aisan.presets.claude_code import claude_code
from aisan.sandbox import RO, RW, Bind, BindSpec
from aisan.session import git_config_binds
from aisan.spec import BoxSpec

UPSTREAM = os.environ.get(
    "AISAN_CLAUDE_UPSTREAM", os.environ.get("AISAN_UPSTREAM", DEFAULT_UPSTREAM)
)


def _state_for(name: str) -> Path:
    """A session-history dir per variant, outside every repo.

    Outside deliberately -- inside the root it would put the agent's own
    transcripts in the tree the agent is editing, and with several repos bound
    there is no longer an obvious one to pick. Note this directory is rw, sits
    outside the repos, and SURVIVES the box: one session can seed the next
    through it (a `CLAUDE.md` placed there is read as instructions), so it is
    part of the perimeter even though nothing in the repos changes.
    """
    state = Path.home() / ".cache" / "aisan-claude" / name
    state.mkdir(parents=True, exist_ok=True)
    return state


def _repo_binds(repo: Path, mode) -> list[BindSpec]:
    """One checkout, bound at its real path, with whatever its `.git` needs.

    The bind before `git_binds`, always: git_binds pins files ON TOP of the
    tree (configs, hooks, the gitdir pointer) and later wins in bwrap's mount
    order, so a bind of the worktree afterwards would cover the pins that make
    the checkout safe to expose.
    """
    return [Bind(repo, mode), *git_binds(repo)]


def single_repo(repo: Path) -> BoxSpec:
    """A single-repo baseline in the shape `aisan claude` builds -- not identical
    to it: the CLI also imports host MCP, terminal env and retry knobs. Here so
    the multi-repo variants below read as a diff against it rather than from
    scratch.
    """
    # git_config_binds: only the config FILE, not all of ~/.config/git (which
    # holds git-credential-store's credentials). HOME is a tmpfs in the box, so
    # git's identity has to be named or a commit inside dies "tell me who you are".
    return claude_code(
        repo,
        state=_state_for(repo.name),
        egress=(AnthropicBackend(upstream=UPSTREAM),),
        extra_env=(
            ("PATH", "/usr/bin:/usr/local/bin"),
            ("CLAUDE_CODE_MAX_RETRIES", "3"),
        ),
    ).with_binds(git_config_binds())


def multi_repo(primary: Path, *others: Path) -> BoxSpec:
    """Several checkouts, all writable, cwd in `primary`.

    The straightforward variant, and the one to reach for when the change spans
    repos that are edited together -- a library and its consumer, a tool and the
    tree it runs against.

    Verified in a real box with three repos bound this way -- two plain
    checkouts and a linked worktree: all three answer `git rev-parse
    --abbrev-ref HEAD`, a write into the second lands, and a sibling repo that
    was NOT named does not exist there at all. That last half is the point --
    the bind list is the whole policy, so an unnamed sibling is absent rather
    than merely unwritten.

    Every repo here is rw, so this is the widest of the variants: an agent that
    goes wrong goes wrong in all of them at once. Prefer `reference_repos` when
    only one of them is actually the subject of the work.
    """
    binds: list[BindSpec] = []
    for other in others:
        binds += _repo_binds(other, RW)
    return single_repo(primary).with_binds(binds)


def reference_repos(primary: Path, *readonly: Path) -> BoxSpec:
    """One repo to edit, the rest to read. The variant to default to.

    Most multi-repo work is asymmetric -- one tree is the subject, the others
    are there to be grepped (a spec, a reference implementation, a knowledge
    base). Binding those ro costs the agent nothing it needs and removes them
    from the blast radius entirely.

    A caveat worth knowing before relying on it: ro is a MOUNT property, so it
    holds against the agent writing files -- but a ro checkout is also one git
    cannot write to, which means no commits, no `git stash`, not even a `git
    gc`. That is usually what you want for reference material and is
    occasionally surprising, because plenty of read-shaped git commands take a
    lock. If the agent needs to commit in a repo, it belongs in `multi_repo`.
    """
    binds: list[BindSpec] = []
    for repo in readonly:
        binds += _repo_binds(repo, RO)
    return single_repo(primary).with_binds(binds)


def v8_worktrees(primary: Path, *others: Path) -> BoxSpec:
    """Several V8 worktrees off one main checkout, plus their shared deps.

    Two things make V8 different from an ordinary multi-repo session, and both
    are measured rather than assumed:

    **The worktrees share one `.git`.** `git_binds` per worktree is what makes
    that safe -- it binds the common `.git` rw (git needs to create
    `packed-refs.lock` on any ref update) while pinning ro every file that
    STEERS host-side git: the configs, the hooks, the gitdir pointers, the
    objects `alternates`. Those are host-code-exec vectors, because the
    orchestrator's own `git cl format` or rebase runs on the host over a tree
    the agent writes. It also SEALS `worktrees/`, so the box cannot reach a
    sibling worktree that is not bound here -- which is exactly the property
    that makes binding two of them a decision rather than an accident.

    **The deps are symlinks out of the tree.** A gclient checkout links
    `build/`, `buildtools/`, `third_party/*` into the main checkout, so a
    worktree bound alone has dangling links and nothing builds.
    `external_symlink_targets` resolves them and they are bound ro -- ro because
    they are shared state belonging to the main checkout, and a box that could
    write them would be editing every other worktree's dependencies at once.

    Deduplicated across worktrees: siblings off one checkout resolve to the same
    targets, and bwrap does not want the same source twice.

    **The consequence to know before using this.** Those targets resolve INTO
    the main checkout (`.../v8/v8/build`, `.../v8/v8/third_party/*`), which is
    also the rw root here -- and later wins, so the ro dep binds land on top of
    it. Measured in a real box with this exact spec: the worktree's `build/`
    resolves and lists, `v8/v8/` itself is writable, and `v8/v8/build/` is
    `Read-only file system`. That is the right trade for a session whose subject
    is the WORKTREE (the shared deps are precisely what should not be edited),
    and it is wrong if the session's job is to change the main checkout's own
    deps -- `gclient sync`, a DEPS roll. For that, make the main checkout the
    only repo and skip this variant.
    """
    binds: list[BindSpec] = []
    targets: set[Path] = set()
    for wt in others:
        binds += _repo_binds(wt, RW)
    for wt in (primary, *others):
        targets.update(external_symlink_targets(wt))
    # Sorted so the same set of worktrees always produces the same spec -- a
    # profile that shuffles is a profile nobody can diff.
    binds += [Bind(t, RO, optional=True) for t in sorted(targets)]
    return single_repo(primary).with_binds(binds)


def with_reference_docs(spec: BoxSpec, *docs: Path) -> BoxSpec:
    """Any of the above, plus ro directories that are not repos at all.

    Manuals, a spec dump, a directory of traces. `optional=True` so a path that
    is not on this host drops out instead of failing assembly -- appropriate for
    reference material, and NOT appropriate for anything load-bearing: an
    optional bind that silently vanishes is a box quietly missing something the
    caller asked for.
    """
    return spec.with_binds([Bind(d, RO, optional=True) for d in docs])


# Every variant takes (primary, *others), so the demo dispatches uniformly and
# the repos come from the command line -- a table of example paths would be
# either this host's, which is not portable, or invented, which would not build.
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
