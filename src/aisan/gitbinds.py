# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Binding a git worktree into a box without handing over the checkout behind it.

Nothing here is V8-shaped: it is the linked-worktree layout git itself defines,
so any consumer boxing a worktree wants exactly this policy. What made it look
domain-specific was living in a module called `sandbox.py` next to the V8 job
profile; the profile is now a preset, and this is a bind helper a preset calls.

The hard part is that a worktree's `.git` is a POINTER, so confining the worktree
confines nothing on its own -- see `git_binds` for the whole argument.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .sandbox import RO, RW, Bind, BindSpec, Seal

log = logging.getLogger(__name__)

# How deep to look for dep symlinks. A checkout tool typically links deps at the
# top level (build, buildtools) and one level down (third_party/*); depth 3 adds
# margin for a layout change without walking the whole tree.
_SYMLINK_SCAN_DEPTH = 3
_SCAN_SKIP = {".git", "out"}  # no deps there, and out/ is huge


def external_symlink_targets(
    root: Path, *, skip: frozenset[str] = frozenset(_SCAN_SKIP)
) -> list[Path]:
    """Resolved targets of symlinks under `root` that point outside it.

    A checkout assembled from shared dependency caches is a tree of symlinks
    into directories the box would otherwise not see, and binding the SYMLINKS
    rather than their targets either breaks the build or -- when the target is
    the main checkout -- hands over a writable tree the box was never meant to
    reach. Bind what they point at, read-only, and the tree resolves.

    Optional in the sense a preset may skip it entirely: a checkout with no
    external links returns an empty list and costs one directory walk.
    """
    root = root.resolve()  # compare resolved against resolved
    targets: set[Path] = set()

    def scan(d: Path, depth: int) -> None:
        for p in d.iterdir():
            if p.name in skip:
                continue
            if p.is_symlink():
                t = p.resolve()
                if t.exists() and not t.is_relative_to(root):
                    targets.add(t)
            elif depth > 1 and p.is_dir():
                scan(p, depth - 1)

    scan(root, _SYMLINK_SCAN_DEPTH)
    return sorted(targets)


def git_binds(worktree: Path) -> list[BindSpec]:
    """The ordered binds for the main checkout's .git behind this worktree.

    A linked worktree's .git is a file pointing at <main>/.git/worktrees/<name>.
    The list reads as the policy, in mount order, later winning:

        Bind(common, RW)              in-worktree git works
        Bind(gitfile, RO)             the pointers and configs that steer
        Bind(common/config, RO)       host-side git are pinned on top of it
        Bind(common/config.worktree, RO)
        Bind(common/objects/info/alternates, RO)
        Bind(common/hooks, RO)
        Seal(common/worktrees)        no sibling, and nothing creatable
        Bind(private, RW)             ...except this job's own dir
        Bind(private/commondir, RO)   whose own pointers are pinned again
        Bind(private/config.worktree, RO)

    The common .git is rw because git locks packed-refs on every ref update, and
    a ro .git root makes that lock uncreatable: every commit then prints "Unable
    to create packed-refs.lock: Read-only file system" -- harmless (the commit
    lands via a loose ref) but the agent misreads it as a failure and thrashes.
    Writable refs/objects are only recoverable repo state.

    What is pinned ro on top is every file that STEERS host-side git, because
    those are host-code-exec vectors: a planted hook, or a
    core.fsmonitor/credential.helper in a config, runs as the host user the
    next time the orchestrator's own git touches the checkout (a squash commit,
    a rebase, a pull, `git cl format` -- all host-side, all in a tree the agent
    writes). Pinning the config CONTENT is not enough on its own: git finds that
    content through two agent-reachable pointers -- <worktree>/.git (the gitdir
    pointer, in the rw root) and <.git>/worktrees/*/commondir (per-worktree, in
    the rw-bound .git). Repointing either makes git read config/hooks from an
    agent-chosen directory, so the content pins are never consulted. Both are
    pinned alongside the content. alternates is a pointer on the same principle:
    it redirects where host-side git resolves objects.

    Sealing worktrees/ rather than pinning each sibling is what makes the policy
    a statement instead of a snapshot. Pins scan the siblings that exist now,
    while the profile outlives the scan (a profile is built per stage, not per
    turn), so a job could write a config.worktree no pin ever covered -- into a
    sibling a concurrent job created afterwards, or into a directory of its own
    invention. The seal answers all of those at once: in the box, worktrees/ is
    an empty directory nothing can be created in, and the siblings are not
    read-only but absent. This job's own dir is bound back through it because git
    needs to create index.lock there, and its own commondir/config.worktree are
    then pinned inside that hole -- which is simply what comes after it in the
    list, not a special case.

    Returns an empty list when the layout is not a linked worktree (a plain .git
    dir is inside the rw root already). Raises when the pointer does not resolve
    to a <main>/.git/worktrees/<name> layout: that means the file was already
    poisoned before this profile was built, and failing assembly abandons the
    turn rather than binding an attacker-named directory rw as if it were .git.
    """
    gitfile = worktree / ".git"
    if gitfile.is_dir() or not gitfile.exists():
        return []
    text = gitfile.read_text().strip()
    if not text.startswith("gitdir:"):
        return []
    private = Path(text.removeprefix("gitdir:").strip())
    main_git = private.parent.parent  # <main>/.git/worktrees/<name> -> <main>/.git
    # Fail closed on a pointer that is not the expected layout. The pin below
    # keeps it honest turn-to-turn (the agent cannot rewrite a ro file), so this
    # only fires on a tree that arrived poisoned -- but binding main_git rw off
    # an unvalidated pointer is exactly the escape, so it is checked, not assumed.
    if private.parent.name != "worktrees" or main_git.name != ".git":
        raise ValueError(
            f"{gitfile} does not point into a <main>/.git/worktrees/<name> "
            f"layout (got {private}); refusing to build a profile from it"
        )
    alternates = main_git / "objects" / "info" / "alternates"
    # A pin needs an existing source (a missing one fails the profile rather
    # than silently dropping a guard), and some are routinely absent:
    # config.worktree only exists once someone ran `git config --worktree`, and
    # hooks/ can be missing after a core.hooksPath setup. Create the missing ones
    # empty -- an empty config contributes nothing and an empty hooks dir runs
    # nothing -- so the guard is always bindable. Only the missing ones: touching
    # an existing file churns the shared checkout's mtimes for nothing.
    (main_git / "hooks").mkdir(exist_ok=True)
    # alternates lives in objects/info, which itself can be absent on a fresh
    # clone; create the parent before touching the file.
    alternates.parent.mkdir(parents=True, exist_ok=True)
    for f in (main_git / "config", main_git / "config.worktree", alternates):
        if not f.exists():
            f.touch()
    binds: list[BindSpec] = [
        Bind(main_git, RW),
        Bind(gitfile, RO),
        Bind(main_git / "config", RO),
        Bind(main_git / "config.worktree", RO),
        Bind(alternates, RO),
        Bind(main_git / "hooks", RO),
    ]
    wts = main_git / "worktrees"
    if wts.is_dir():
        binds.append(Seal(wts))
    # The hole through the seal, then the pointers inside the hole pinned back on
    # top of it. commondir is never absent in a real private dir, so a missing
    # one raises rather than being papered over with an empty file that would
    # point git at the filesystem root; config.worktree is created empty above
    # for the siblings' sake and here for our own.
    binds.append(Bind(private, RW))
    own_worktree_config = private / "config.worktree"
    if not own_worktree_config.exists():
        own_worktree_config.touch()
    binds += [Bind(private / "commondir", RO), Bind(own_worktree_config, RO)]
    return binds
