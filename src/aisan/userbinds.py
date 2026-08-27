# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""User-written bind specs: a four-key TOML file, appended to a preset.

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

`~` expands, and a relative path resolves against the FILE's directory, so a
spec checked into a repo can name its neighbours without knowing where the
checkout lives.

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
reordering, globs, Seal and BindOver.

The credential guard is why `load` takes the egress tuple. Until now every
bind source was audited code plus a snapshot; a user file is the first
input nobody reviewed, and the credential-absence claim is subtraction -- a
spec naming `~` would re-add exactly what the profile exists to keep out.
`load` refuses such a file naming the key it came from, and `Box._sandbox`
enforces the same rule over spec binds at assembly, so neither a user file
nor a hand-composed caller can route around it.

Git-awareness is NOT applied here. For interactive sessions the perimeter
model treats the whole root as the session's, and hook-pinning user repos
would be a different model applied inconsistently; the one measured trap --
a linked worktree bound alone answers `fatal: not a git repository` -- is
recorded in the session scripts, and the fix for that case is `git_binds` in
a hand-composed spec, which `aisan-claude-specs.py` demonstrates.
"""

from __future__ import annotations

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


def load(path: Path, *, egress: tuple[Backend, ...] = ()) -> UserSpec:
    """The mounts and PATH entries a user file names, validated and guarded.

    Raises ValueError naming the file and the offending key for every
    malformed input -- a typo'd key (`wr = [...]`) is a refusal, not a
    silently ignored line in a file somebody believes is in effect.
    FileNotFoundError propagates: a missing spec file is the caller's path
    being wrong, not this file's contents.
    """
    try:
        with path.open("rb") as f:
            doc = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} is not valid TOML: {e}") from e

    unknown = sorted(set(doc) - {*_MOUNT_KEYS, "path"})
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {', '.join(map(repr, unknown))};"
            f" expected any of {', '.join(map(repr, (*_MOUNT_KEYS, 'path')))}"
        )

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

    binds: list[BindSpec] = [Overlay(p) for p in _dedup(mounts["overlay"])]
    binds += [Bind(p, RO, optional=True) for p in _dedup(mounts["ro"])]
    binds += [Bind(p, RW) for p in _dedup(mounts["rw"])]

    dirs = _dedup(_entries(path, doc.get("path", []), "path"))
    mounted = [p for key in _MOUNT_KEYS for p in mounts[key]]
    uncovered = [d for d in dirs if not any(d.is_relative_to(m) for m in mounted)]
    if uncovered:
        raise ValueError(
            f"{path}: path entry {uncovered[0]} is not covered by any ro, rw or"
            " overlay entry in this file -- a PATH directory the box does not"
            " mount resolves nothing"
        )

    exposure = credential_exposure(tuple(binds), egress)
    if exposure is not None:
        src, backend, cred = exposure
        raise ValueError(
            f"{path}: {src} would expose the {backend.name} backend's"
            f" credential at {cred} -- a user bind may not contain it"
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
        p = Path(entry).expanduser()
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
