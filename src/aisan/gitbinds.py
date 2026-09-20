# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Bind a Git worktree without exposing or compromising its main checkout.

A linked worktree's `.git` file points outside the worktree, so confining the
worktree directory alone breaks Git. The shared directory must be writable for
normal Git operations, while files that can influence later host-side Git
commands remain pinned read-only. Plain checkouts need the same pins inside
their in-tree `.git` directory. `git_binds` documents the complete policy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .sandbox import RO, RW, Bind, BindSpec, EnsurePath, Seal

log = logging.getLogger(__name__)

# Box-scoped Git configuration. Environment variables apply to every invocation
# without changing host files. Disabling `gc.auto` prevents an automatic repack
# from rewriting shared objects and refs used by the main checkout and sibling
# worktrees.
#
# The prune settings are supplemental. Testing showed that these three settings
# alone still allowed a manual `git gc --prune=now` to destroy a commit reachable
# only from a sibling. `pin_packs` supplies the actual guard; these settings stop
# automatic cleanup before it starts.
GC_ENV = (
    ("GIT_CONFIG_COUNT", "3"),
    ("GIT_CONFIG_KEY_0", "gc.auto"),
    ("GIT_CONFIG_VALUE_0", "0"),
    ("GIT_CONFIG_KEY_1", "gc.pruneExpire"),
    ("GIT_CONFIG_VALUE_1", "never"),
    ("GIT_CONFIG_KEY_2", "gc.worktreePruneExpire"),
    ("GIT_CONFIG_VALUE_2", "never"),
)

# Dependency symlinks usually appear at the root or below `third_party`. Scan one
# extra level without traversing the entire checkout.
_SYMLINK_SCAN_DEPTH = 3
_SCAN_SKIP = {".git", "out"}  # no deps there, and out/ is huge


def external_symlink_targets(
    root: Path, *, skip: frozenset[str] = frozenset(_SCAN_SKIP)
) -> list[Path]:
    """Return safe external targets of symlinks below `root`.

    Checkouts may link dependencies into directories the box cannot otherwise
    see. Binding those targets read-only allows the links to resolve without
    granting writable access to the main checkout.

    Only targets inside the linked worktree's main checkout are trusted. v8-utils
    creates its dependency links in that shape, as verified against a real V8
    checkout. A writable worktree could otherwise add a link to `~/.ssh`, `/`,
    or another unintended host path. Drop such targets with a warning.

    Plain checkouts provide no external allowlist, so all targets outside the
    root are dropped. Their dependencies normally reside in the checkout.

    A checkout without external links returns an empty list.
    """
    root = root.resolve()
    layout = _git_layout(root)
    allowed = layout[2].parent.resolve() if layout is not None else None
    targets: set[Path] = set()

    def scan(d: Path, depth: int) -> None:
        for p in d.iterdir():
            if p.name in skip:
                continue
            if p.is_symlink():
                t = p.resolve()
                if not t.exists() or t.is_relative_to(root):
                    continue
                if allowed is not None and t.is_relative_to(allowed):
                    targets.add(t)
                else:
                    log.warning(
                        "external_symlink_targets: dropping %s -> %s: outside the"
                        " main checkout %s",
                        p,
                        t,
                        allowed,
                    )
            elif depth > 1 and p.is_dir():
                scan(p, depth - 1)

    scan(root, _SYMLINK_SCAN_DEPTH)
    return sorted(targets)


def git_binds(worktree: Path, *, pin_packs: bool = False) -> list[BindSpec]:
    """Return the binds for the Git metadata behind `worktree`.

    For a linked worktree, `.git` points to
    `<main>/.git/worktrees/<name>`. The policy is:

        Bind(common, RW)              in-worktree git works
        Bind(gitfile, RO)      guard  the pointers and configs that steer
        Bind(common/config, RO)       host-side git are pinned below it
        Bind(common/config.worktree, RO)
        Bind(common/objects/info/alternates, RO)
        Bind(common/hooks, RO)
        Seal(common/worktrees)        no sibling, and nothing creatable
        Bind(private, RW)      guard  ...except this job's own dir
        Bind(private/commondir, RO)   whose own pointers are pinned again
        Bind(private/config.worktree, RO)
        Bind(common/objects/pack, RO) with pin_packs: nothing in the box repacks

    Every pin and the seal are guards, so no plain bind can reopen any part of
    them; see `sandbox.Bind`.

    The common `.git` remains writable because Git locks `packed-refs` during
    ref updates. A read-only directory makes every commit report a lock error,
    even when the loose ref succeeds, and prevents normal repository recovery.

    Read-only pins protect every file that can steer later host-side Git. Hooks,
    `core.fsmonitor`, and credential helpers can execute as the host user during
    commands such as rebase, pull, or `git cl format`. Pinning config contents
    alone is insufficient: the worktree `.git` file and its `commondir` can
    redirect Git to attacker-chosen config and hooks. `alternates` receives the
    same protection because it redirects object lookup.

    Sealing `worktrees/` also covers siblings created after profile assembly.
    Pinning only existing siblings would leave later `config.worktree` files
    writable. The seal hides all siblings and prevents new entries. A guarded
    writable bind below it restores this worktree's private directory so Git
    can create `index.lock`, and the pins below that hole hold its steering
    files.

    A plain checkout has an in-tree `.git` directory. It needs the same pins,
    minus the linked-worktree pointers, plus a self-bind:

        Bind(.git, RW)                a mount point, so the directory itself
                                      cannot be renamed or unlinked from inside
        Bind(.git/config, RO)         the same steering files as above
        Bind(.git/config.worktree, RO)
        Bind(.git/objects/info/alternates, RO)
        Bind(.git/hooks, RO)
        Seal(.git/worktrees)          nothing creatable
        Bind(.git/objects/pack, RO)   with pin_packs

    The self-bind turns `.git` into a mount point that cannot be renamed. Testing
    showed that file pins alone allowed `mv .git .git.old`, after which the box
    could create new config and hooks for host Git to read. The worktrees seal
    also prevents the box from creating a worktree with an attacker-controlled
    `commondir`. Consequently, `git worktree add` is unavailable inside the box.

    `pin_packs` protects objects reachable only from hidden siblings. Because
    the seal hides their HEAD and index, in-box `git gc` can otherwise consider
    those objects unreachable and prune them from the shared store. A scratch
    test destroyed a sibling's detached-HEAD commit without this pin; making
    `objects/pack` read-only caused `git gc --prune=now` to fail before damage.
    `GC_ENV` cannot prevent an explicitly requested destructive command.

    Pack pinning also blocks repacking, `git fetch`, and `git clone`. Callers use
    it for network-isolated boxes, which cannot fetch anyway. Commits still
    write loose objects for the host to repack later.

    Return an empty list when the worktree has no `.git`. Refuse malformed
    linked-worktree pointers instead of mounting an attacker-chosen directory
    writable as Git metadata.
    """
    layout = _git_layout(worktree)
    if layout is None:
        git = _plain_git(worktree)
        if git is None:
            return []
        binds: list[BindSpec] = [
            Bind(git, RW),
            *_steering_pins(git),
            Seal(git / "worktrees"),
        ]
        binds += [Bind(p, RO) for p, _is_dir in _submodule_steering(git)]
        if pin_packs:
            binds.append(Bind(git / "objects" / "pack", RO))
        return binds
    gitfile, private, main_git = layout
    # `git_host_files` declares missing guard sources for `Box` to create and
    # remove. Keeping creation there leaves this function free of host changes.
    binds = [Bind(main_git, RW), Bind(gitfile, RO), *_steering_pins(main_git)]
    wts = main_git / "worktrees"
    if wts.is_dir():
        binds.append(Seal(wts))
    # Restore this worktree's private directory through the seal, then pin its
    # steering files. `commondir` must already exist; fabricating an empty one
    # would make Git interpret the filesystem root as the common directory.
    binds.append(Bind(private, RW))
    binds += [Bind(private / "commondir", RO), Bind(private / "config.worktree", RO)]
    binds += [Bind(p, RO) for p, _is_dir in _submodule_steering(main_git)]
    if pin_packs:
        binds.append(Bind(main_git / "objects" / "pack", RO))
    return binds


def _steering_pins(git: Path) -> list[BindSpec]:
    """Return read-only pins for config, object redirects, and hooks."""
    return [
        Bind(git / "config", RO),
        Bind(git / "config.worktree", RO),
        Bind(git / "objects" / "info" / "alternates", RO),
        Bind(git / "hooks", RO),
    ]


def _steering_host_files(git: Path) -> list[EnsurePath]:
    """Return ensure-paths for steering pins absent in a fresh checkout."""
    return [
        EnsurePath(git / "hooks", is_dir=True),
        EnsurePath(git / "config", is_dir=False),
        EnsurePath(git / "config.worktree", is_dir=False),
        # Box also creates the missing `objects/info` parent.
        EnsurePath(git / "objects" / "info" / "alternates", is_dir=False),
    ]


def _plain_git(root: Path) -> Path | None:
    """`<root>/.git` when it is a directory, None when there is none.

    Refuse a symlink because bwrap cannot place the guard mounts over it, and
    following it could expose a writable directory outside the root.
    """
    git = root / ".git"
    if git.is_symlink():
        raise ValueError(
            f"{git} is a symlink; refusing to pin steering files through it"
        )
    return git if git.is_dir() else None


def _submodule_steering(main_git: Path) -> list[tuple[Path, bool]]:
    """Return config and hooks paths for each present submodule gitdir.

    Submodule config can define `core.fsmonitor` or credential helpers, and its
    hooks can execute during a later host-side Git command. Pin both under
    `<main>/.git/modules/<name>` when present.

    This scan covers only top-level submodules present during profile assembly.
    Nested or subsequently initialized submodules remain outside its coverage.
    """
    modules = main_git / "modules"
    if not modules.is_dir():
        return []
    steering: list[tuple[Path, bool]] = []
    for sub in sorted(modules.iterdir()):
        # `config` or `HEAD` distinguishes a gitdir from unrelated entries.
        if (sub / "config").exists() or (sub / "HEAD").exists():
            steering.append((sub / "config", False))
            steering.append((sub / "hooks", True))
    return steering


def _git_layout(worktree: Path) -> tuple[Path, Path, Path] | None:
    """`(<worktree>/.git file, its private dir, the main .git)` for a linked
    worktree, or None for a plain checkout whose .git is inside the rw root.

    Refuse pointers outside a `<main>/.git/worktrees/<name>` layout owned by this
    worktree. Git's `<private>/gitdir` back-pointer must name this `.git` file;
    otherwise a poisoned pointer could make the box mount another repository's
    object store writable. Read-only pins prevent the pointer from changing
    after validation.
    """
    gitfile = worktree / ".git"
    if gitfile.is_dir() or not gitfile.exists():
        return None
    text = gitfile.read_text().strip()
    if not text.startswith("gitdir:"):
        return None
    private = Path(text.removeprefix("gitdir:").strip())
    main_git = private.parent.parent
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
    """Return missing `.git` guard sources for `Box` to create temporarily.

    Fresh repositories may lack `config.worktree`, `hooks`, `objects/info`, or
    `objects/pack`. Empty versions have no effect but give read-only guard binds
    a source. `Box` creates them before assembly and removes its additions after
    running or inspection, keeping `git_binds` pure.

    `commondir` is excluded because a valid linked worktree must provide it; a
    missing file is an error. Plain checkouts also ensure the `worktrees`
    directory required by the seal. Repositories without `.git` return no paths.
    """
    layout = _git_layout(worktree)
    if layout is None:
        git = _plain_git(worktree)
        if git is None:
            return ()
        ensure = [
            *_steering_host_files(git),
            EnsurePath(git / "worktrees", is_dir=True),
        ]
        ensure += [
            EnsurePath(p, is_dir=is_dir) for p, is_dir in _submodule_steering(git)
        ]
        if pin_packs:
            ensure.append(EnsurePath(git / "objects" / "pack", is_dir=True))
        return tuple(ensure)
    _gitfile, private, main_git = layout
    ensure = [
        *_steering_host_files(main_git),
        EnsurePath(private / "config.worktree", is_dir=False),
    ]
    ensure += [
        EnsurePath(p, is_dir=is_dir) for p, is_dir in _submodule_steering(main_git)
    ]
    if pin_packs:
        ensure.append(EnsurePath(main_git / "objects" / "pack", is_dir=True))
    return tuple(ensure)
