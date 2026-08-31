# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""User-written bind specs: a five-key TOML file, appended to a preset.

The launchers' escape hatch until now was "copy the script and edit it",
which forks the template the moment a session wants one more directory. This
is the smaller format for that case: a file naming paths to bind ro, rw and
overlaid, appended AFTER the preset's own list by the caller's `with_binds`
-- later wins is the model's one precedence rule, so appending is exactly
"these shadow the template", and no composition semantics have to be
defined::

    # a user spec, in full
    ro = [                       # optional: absent on this host -> dropped
        "~/ro_repo",             # reference material comes and goes
    ]
    rw = [                       # mandatory: missing -> the launch refuses
        "~/rw_repo",
    ]
    overlay = [                  # mandatory: warm to read, writable, discarded
        "~/.cache/some-tool-store",
    ]
    path = [                     # prepended to the box PATH
        "~/tooling",             # must be covered by a mount named above
    ]
    include = [                  # other spec files, expanded in place
        "./base-userbinds.toml", # BEFORE this file's own keys, so this
    ]                            # file shadows what it includes

`~` expands, and a relative path resolves against the FILE's directory, so a
spec checked into a repo can name its neighbours without knowing where the
checkout lives. `include` is the same resolution applied to spec files, which
is what lets a set of them live in one directory and name each other by
basename.

`include` exists because the alternative ways to compose a growing collection
are both worse. Repeating `--binds` per file puts the order in the operator's
shell history, where the next reader cannot see it; a glob puts it in
lexicographic filename order, where nobody can see it and a rename silently
changes which file wins. An include list is the order, written down in the file
that depends on it. Expansion is depth-first and in the order given, each file
still applied whole -- so later-wins remains the one precedence rule, and an
including file shadows what it includes.

A file already expanded is not expanded again: a diamond is two paths to one
file, not a request to mount it twice. A file that includes itself, however
transitively, is a refusal naming the chain -- unlike the diamond, that is
never what anybody meant.

The three optionality defaults are the ones the repo measured rather than
picked: a ro reference bind that has vanished is a bind to skip
(`with_reference_docs`), while a named WRITABLE that silently vanishes is a
box quietly missing what the caller asked for. `overlay` is mandatory for a
sharper reason than rw's -- `Overlay`'s docstring measures what the two
plausible alternatives do to a shared tool cache, and the failure a skipped
one produces is a HANG rather than a missing directory. There is deliberately
no per-path `optional` key: when the defaults are wrong for a case, the
escape hatch is still Python, and a data file that grew the vocabulary for it
would be the matching DSL the library explicitly refuses to bless.

`path` is the one key here that is not a mount, and it is admitted because it
names DIRECTORIES rather than values. Every entry must be covered by a mount
this same file names, so the key cannot widen the box by a byte: it makes
what a bind already granted resolvable by name, which is the other half of
the same grant. A tool tree bound and not on the PATH is a directory whose
contents nothing resolves, and the shims that would work around that (a
symlink into a directory already on the PATH) break the argv0-relative
bootstrap such trees are usually written with. Coverage is checked against
this file rather than the merged spec so the refusal names two lines the
reader can see at once.

Same reasoning for what is absent on purpose: env VALUES (capability, not
mount -- a key naming one is the grant this format refuses, and a payload
that needs one has its own configuration to put it in), removals,
reordering, globs, Seal and BindOver. `include` names files rather than
patterns for the same reason globs are absent from the mount keys: the set it
resolves to has to be readable off the page.

The credential guard is why `load` takes the egress tuple. Until now every
bind source was audited code plus a snapshot; a user file is the first
input nobody reviewed, and the credential-absence claim is subtraction -- a
spec naming `~` would re-add exactly what the profile exists to keep out.
`load` refuses such a file naming the key it came from. `Box._sandbox` then
asks the harder question of the assembled box -- whether its mounts leave the
credential readable inside -- so neither a user file nor a hand-composed
caller can route around it. This one is deliberately the stricter of the two:
a path a person wrote down may not name the credential even if some later
mount would go on to cover it.

Git-awareness is NOT applied here. For interactive sessions the perimeter
model treats the whole root as the session's, and hook-pinning user repos
would be a different model applied inconsistently; the one measured trap --
a linked worktree bound alone answers `fatal: not a git repository` -- is
recorded in the session scripts, and the fix for that case is `git_binds` in
a hand-composed spec, which `aisan-claude-specs.py` demonstrates.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .egress.base import Backend, credential_exposure
from .sandbox import RO, RW, Bind, BindSpec, Overlay

__all__ = ["UserSpec", "load"]

# Mount keys in the order their binds are emitted: warm caches, then what the
# box may read, then what it may write. `path` is not one of them -- it grants
# no mount and is checked against these.
_MOUNT_KEYS = ("overlay", "ro", "rw")

# Every key a spec file may carry. `include` is neither a mount nor checked
# against them -- it names other spec files.
_KEYS = (*_MOUNT_KEYS, "path", "include")


@dataclass(frozen=True)
class UserSpec:
    """One user file's contents: mounts, and PATH entries covered by them.

    Two fields rather than one because they reach the spec by two different
    combinators (`with_binds`, `with_path_prefix`) -- and returning them
    together keeps that a property of the file, not of the number of times a
    caller remembered to parse it.
    """

    binds: list[BindSpec]
    path: tuple[Path, ...]


def load(path: Path, *, egress: tuple[Backend, ...]) -> UserSpec:
    """The mounts and PATH entries a user file names, validated and guarded.

    Raises ValueError naming the file and the offending key for every
    malformed input -- a typo'd key (`wr = [...]`) is a refusal, not a
    silently ignored line in a file somebody believes is in effect.
    FileNotFoundError propagates: a missing spec file is the caller's path
    being wrong, not this file's contents.

    `egress` has no default on purpose. It is the credential guard's only input,
    and a default of `()` would be a guard silently switched off for any caller
    that forgot it -- the one failure the guard exists to prevent. A caller with
    no backends passes `()` and says so; a caller with backends cannot omit them
    by accident.
    """
    spec = _load(path, egress, [], set())
    _assert_path_coverage(path, spec)
    return spec


def _assert_path_coverage(path: Path, spec: UserSpec) -> None:
    """Every PATH dir must sit under some mount the box actually gets.

    Over the WHOLE merged spec, so a diamond-included mount counts no matter
    which branch expanded it (see `_load`). Normalised before the containment
    test: `/tools/bin/../../../etc` is `is_relative_to("/tools")` by components
    while resolving outside it, so an un-normalised check would call an escaping
    PATH dir covered. A bind's box-side path is its own (`Bind`/`Overlay` mount
    at their source); userbinds emits no substituting bind."""
    mounted = [Path(os.path.normpath(b.path)) for b in spec.binds if hasattr(b, "path")]
    for d in spec.path:
        if not any(Path(os.path.normpath(d)).is_relative_to(m) for m in mounted):
            raise ValueError(
                f"{path}: path entry {d} is not covered by any ro, rw or overlay"
                " entry in the spec or its includes -- a PATH directory the box"
                " does not mount resolves nothing"
            )


def _load(
    path: Path,
    egress: tuple[Backend, ...],
    chain: list[Path],
    expanded: set[Path],
) -> UserSpec:
    """`load` plus the two things an include tree needs to carry.

    `chain` is the include path taken to get here, resolved, for cycle
    detection and for the message that names it. `expanded` is every file the
    whole tree has already applied, so a diamond mounts once.
    """
    here = path.resolve()
    if here in chain:
        raise ValueError(
            f"{path}: include cycle: " + " -> ".join(str(p) for p in (*chain, here))
        )
    expanded.add(here)
    try:
        with path.open("rb") as f:
            doc = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} is not valid TOML: {e}") from e

    unknown = sorted(set(doc) - set(_KEYS))
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {', '.join(map(repr, unknown))};"
            f" expected any of {', '.join(map(repr, _KEYS))}"
        )

    inner_binds: list[BindSpec] = []
    inner_dirs: list[Path] = []
    for spec_file in _entries(path, doc.get("include", []), "include"):
        # Skip only a file some OTHER branch already applied. One that is on
        # the way here is a cycle, and must reach _load's check rather than be
        # quietly dropped as a repeat.
        target = spec_file.resolve()
        if target in expanded and target not in (*chain, here):
            continue
        if not spec_file.is_file():
            # Not the bare FileNotFoundError the caller's own path raises: this
            # one has an includer to name, and the pair is the fix.
            raise ValueError(f"{path}: include {spec_file} is not a file")
        inner = _load(spec_file, egress, [*chain, here], expanded)
        inner_binds += inner.binds
        inner_dirs += inner.path

    # Absent keys default to empty lists -- a file with only `ro` is a file
    # with no writable binds, not a malformed one.
    mounts = {key: _entries(path, doc.get(key, []), key) for key in _MOUNT_KEYS}
    for (a, first), (b, second) in combinations(mounts.items(), 2):
        both = [str(p) for p in first if p in second]
        if both:
            # There is no order between two keys to resolve this with, and
            # picking one silently would be a precedence rule nobody asked for.
            raise ValueError(
                f"{path}: {both[0]} appears in both {a} and {b} -- pick one"
            )

    # Exact equality above is not enough: the keys emit in a fixed order
    # (overlay, then ro, then rw) and later wins, so a broad `rw` naming an
    # ANCESTOR of an earlier `ro` (or `overlay`) silently upgrades that narrower
    # path to writable -- and for an overlay it defeats the copy-up, sending
    # writes to the host tree. Refuse rather than reorder: there is no spelling
    # of "ro under rw" this format should honour by guessing. Compared on the
    # normalised form, which also closes the `..` route around the check above.
    # Scoped to this file's own keys, not the includes: a `rw` here shadowing an
    # included `ro` is the documented later-wins doing its job.
    emitted = [
        (key, p, Path(os.path.normpath(p))) for key in _MOUNT_KEYS for p in mounts[key]
    ]
    for index, (early_key, _early_p, early_norm) in enumerate(emitted):
        for late_key, late_p, late_norm in emitted[index + 1 :]:
            if late_key != early_key and early_norm.is_relative_to(late_norm):
                raise ValueError(
                    f"{path}: {late_key} entry {late_p} covers the earlier"
                    f" {early_key} entry {_early_p} -- the later mount would"
                    " override it (a nested ro or overlay under rw becomes"
                    " writable); remove the nesting"
                )

    # Includes first, so this file's own keys shadow them -- the same
    # later-wins the launcher applies between two --binds files. A path this
    # file mounts rw over an include's ro is that rule doing its job, which is
    # why the cross-key refusal below stays WITHIN one document.
    binds: list[BindSpec] = [*inner_binds]
    binds += [Overlay(p) for p in _dedup(mounts["overlay"])]
    binds += [Bind(p, RO, optional=True) for p in _dedup(mounts["ro"])]
    binds += [Bind(p, RW) for p in _dedup(mounts["rw"])]

    dirs = _dedup([*inner_dirs, *_entries(path, doc.get("path", []), "path")])
    # PATH coverage is checked ONCE over the merged tree, in `load` below, not
    # per file: a diamond include is expanded only on its first path, so its
    # mounts are absent from `inner_binds` on the second -- and a PATH entry
    # covered by that include would be reported uncovered or not depending on
    # include order in a file the reader is not looking at. The final box has
    # every file's mounts, so coverage is a fact about the whole tree.

    exposure = credential_exposure(tuple(binds), egress)
    if exposure is not None:
        src, backend, cred = exposure
        raise ValueError(
            f"{path}: {src} would expose the {backend.name} backend's"
            f" credential at {cred} -- a user bind may not name it, nor"
            " anything containing it or inside it"
        )
    return UserSpec(binds, tuple(dirs))


def _entries(path: Path, raw: object, key: str) -> list[Path]:
    """`raw` as a list of expanded absolute Paths, or a ValueError naming it.

    Expanded here rather than at bind time so the cross-key and dedupe checks
    compare paths, not spellings: `~/a` and its expansion are the same bind,
    and a file that names one twice in two spellings has still named it twice.
    NOT symlink-resolved -- the credential guard resolves for its own
    comparison, and silently canonicalising a user's paths would rewrite the
    box's mount points out from under the file that named them.
    """
    if not isinstance(raw, list) or not all(
        isinstance(e, str) and e.strip() for e in raw
    ):
        raise ValueError(f"{path}: {key} must be an array of non-empty path strings")
    out = []
    for entry in raw:
        # os.pathsep in one entry becomes TWO PATH entries once joined, and the
        # second rides past the coverage check that saw one string. A real path
        # never needs it; refuse it rather than let it smuggle a directory in.
        if os.pathsep in entry:
            raise ValueError(f"{path}: {key} entry {entry!r} contains {os.pathsep!r}")
        try:
            p = Path(entry).expanduser()
        except RuntimeError as e:
            # ~nosuchuser and friends: a RuntimeError bypasses the "ValueError
            # naming the file" contract this loader promises its caller.
            raise ValueError(
                f"{path}: {key} entry {entry!r} cannot be expanded: {e}"
            ) from e
        if not p.is_absolute():
            p = path.parent / p
        out.append(p)
    return out


def _dedup(paths: list[Path]) -> list[Path]:
    """Same paths, first occurrence's order, one entry each. bwrap does not
    want the same source twice, and a duplicate in a user file is far more
    likely an edit-in-place than a wish to mount it again."""
    seen: set[Path] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]
