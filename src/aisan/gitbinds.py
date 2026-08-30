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

from .sandbox import RO, RW, Bind, BindSpec, EnsurePath, Seal

log = logging.getLogger(__name__)

# Box-scoped git configuration, as environment rather than a file: it applies
# to every git invocation inside the box, touches no host state, and cannot
# outlive the box. `gc.auto` is the one that matters most -- the common .git is
# bound rw, so a repack triggered from a turn would rewrite the SHARED
# packed-refs/objects out from under the main checkout and any concurrent
# worktree.
#
# The two prune knobs are defence in depth and NOT the guard: measured, a box
# with all three set still lost a commit that only a sibling worktree's HEAD
# referenced, because `gc.pruneExpire` governs LOOSE objects while repack drops
# unreachable packed ones, and `git gc --prune=now` overrides the setting
# anyway. What actually holds is `pin_packs` -- see `git_binds`. These are kept
# because they cost nothing and stop the automatic path before it starts.
GC_ENV = (
    ("GIT_CONFIG_COUNT", "3"),
    ("GIT_CONFIG_KEY_0", "gc.auto"),
    ("GIT_CONFIG_VALUE_0", "0"),
    ("GIT_CONFIG_KEY_1", "gc.pruneExpire"),
    ("GIT_CONFIG_VALUE_1", "never"),
    ("GIT_CONFIG_KEY_2", "gc.worktreePruneExpire"),
    ("GIT_CONFIG_VALUE_2", "never"),
)

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


def git_binds(worktree: Path, *, pin_packs: bool = False) -> list[BindSpec]:
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
        Bind(common/objects/pack, RO) with pin_packs: nothing in the box repacks

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

    `pin_packs` is the one guard the seal above does not provide, and the seal
    is why it is needed. Sealing worktrees/ makes the siblings ABSENT, which
    also removes their HEAD and index as reachability roots -- so an in-box
    `git gc` sees objects that only a sibling references as unreachable and
    prunes them from the object store it shares with the host. Measured on a
    scratch repo: with the seal and nothing else, a sibling's detached-HEAD
    commit was destroyed and that worktree was left at "fatal: bad object
    HEAD"; with objects/pack ro, the same `git gc --prune=now` answered "fatal:
    failed to run repack" and both survived. Config cannot express this (see
    `GC_ENV`), because the destructive commands are the ones an agent types
    rather than the ones git runs on its own.

    The cost is that nothing in the box may write a pack: repacking, and
    anything running index-pack -- `git fetch`, `git clone`. A box with its own
    network namespace has nothing to fetch FROM, which is why the callers here
    pass `pin_packs=unshare_net` rather than always. Ordinary work is
    unaffected: a commit writes loose objects, and the host repacks them later.

    Returns an empty list when the layout is not a linked worktree (a plain .git
    dir is inside the rw root already) -- pin_packs included, since the hazard
    is a .git SHARED with worktrees the box cannot see, and a plain checkout's
    own .git is inside the session's perimeter. Raises when the pointer does not resolve
    to a <main>/.git/worktrees/<name> layout: that means the file was already
    poisoned before this profile was built, and failing assembly abandons the
    turn rather than binding an attacker-named directory rw as if it were .git.
    """
    layout = _git_layout(worktree)
    if layout is None:
        return []
    gitfile, private, main_git = layout
    alternates = main_git / "objects" / "info" / "alternates"
    # Pure: the guard sources this pins may not exist yet on a fresh checkout,
    # and creating them is a host mutation `git_host_files` declares for the Box
    # to do and undo. A bind is just a path here; resolution checks existence
    # once the ensure-paths are on disk.
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
    # one fails the profile rather than being papered over with an empty file
    # that would point git at the filesystem root -- which is why it is NOT in
    # `git_host_files`, unlike this worktree's own config.worktree.
    binds.append(Bind(private, RW))
    binds += [Bind(private / "commondir", RO), Bind(private / "config.worktree", RO)]
    binds += [Bind(p, RO) for p, _is_dir in _submodule_steering(main_git)]
    if pin_packs:
        binds.append(Bind(main_git / "objects" / "pack", RO))
    return binds


def _submodule_steering(main_git: Path) -> list[tuple[Path, bool]]:
    """(path, is_dir) for each present submodule gitdir's config and hooks.

    A submodule keeps its gitdir under `<main>/.git/modules/<name>`, and
    host-side git run in that submodule reads its config (a core.fsmonitor or
    credential.helper there executes as the host user) and runs its hooks -- the
    same host-exec vector the main .git pins close, one level down and in a tree
    the box can also write. Pinned ro when present.

    Bounded on purpose: top-level submodules only (nested ones are rare and left
    uncovered), and a snapshot like the pre-seal worktree pins -- a submodule
    initialised after the profile is built is not covered. Empty when the repo
    has no `modules/` at all, which is the common case.
    """
    modules = main_git / "modules"
    if not modules.is_dir():
        return []
    steering: list[tuple[Path, bool]] = []
    for sub in sorted(modules.iterdir()):
        # A gitdir, not some other entry: `config` (or `HEAD`) marks one.
        if (sub / "config").exists() or (sub / "HEAD").exists():
            steering.append((sub / "config", False))
            steering.append((sub / "hooks", True))
    return steering


def _git_layout(worktree: Path) -> tuple[Path, Path, Path] | None:
    """`(<worktree>/.git file, its private dir, the main .git)` for a linked
    worktree, or None for a plain checkout whose .git is inside the rw root.

    Raises when the `.git` pointer does not resolve to a
    `<main>/.git/worktrees/<name>` layout OWNED by this worktree: git keeps a
    back-pointer (`<private>/gitdir`) naming the worktree's own `.git`, so a
    poisoned `gitdir: /other/repo/.git/worktrees/z` points back at that repo's
    worktree, not ours, and binding an unrelated repo's store rw off it is the
    escape this refuses. Both the shape check and the back-pointer keep it
    honest; the pin keeps it honest turn-to-turn (the agent cannot rewrite a ro
    file), so this only fires on a tree that arrived poisoned.
    """
    gitfile = worktree / ".git"
    if gitfile.is_dir() or not gitfile.exists():
        return None
    text = gitfile.read_text().strip()
    if not text.startswith("gitdir:"):
        return None
    private = Path(text.removeprefix("gitdir:").strip())
    main_git = private.parent.parent  # <main>/.git/worktrees/<name> -> <main>/.git
    if private.parent.name != "worktrees" or main_git.name != ".git":
        raise ValueError(
            f"{gitfile} does not point into a <main>/.git/worktrees/<name> "
            f"layout (got {private}); refusing to build a profile from it"
        )
    backpointer = private / "gitdir"
    if (
        not backpointer.is_file()
        or Path(backpointer.read_text().strip()).resolve() != gitfile.resolve()
    ):
        raise ValueError(
            f"{gitfile} points at {private}, whose gitdir back-pointer does not "
            f"name {gitfile}; refusing to bind a repo this worktree does not own"
        )
    return gitfile, private, main_git


def git_host_files(
    worktree: Path, *, pin_packs: bool = False
) -> tuple[EnsurePath, ...]:
    """The .git guard sources `git_binds` pins that a fresh checkout may lack.

    Split out so `git_binds` stays a pure function: these are the paths the Box
    creates empty before assembly and removes again on the inspector's staged
    path (see `EnsurePath`). Some are routinely absent -- config.worktree only
    exists once someone ran `git config --worktree`, hooks/ can be missing under
    a core.hooksPath setup, objects/info and objects/pack are absent on a repo
    never repacked -- and an empty config contributes nothing while an empty
    hooks dir runs nothing, so the guard is bindable without the file having to
    pre-exist. Only these; commondir is deliberately NOT here (a missing one is
    a refusal, not something to paper over). Empty for a plain checkout.
    """
    layout = _git_layout(worktree)
    if layout is None:
        return ()
    _gitfile, private, main_git = layout
    ensure = [
        EnsurePath(main_git / "hooks", is_dir=True),
        EnsurePath(main_git / "config", is_dir=False),
        EnsurePath(main_git / "config.worktree", is_dir=False),
        # Its parent objects/info is created for it by the Box.
        EnsurePath(main_git / "objects" / "info" / "alternates", is_dir=False),
        EnsurePath(private / "config.worktree", is_dir=False),
    ]
    ensure += [
        EnsurePath(p, is_dir=is_dir) for p, is_dir in _submodule_steering(main_git)
    ]
    if pin_packs:
        ensure.append(EnsurePath(main_git / "objects" / "pack", is_dir=True))
    return tuple(ensure)
