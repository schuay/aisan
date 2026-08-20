# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""User-written bind specs: a two-key TOML file, appended to a preset.

The launchers' escape hatch until now was "copy the script and edit it",
which forks the template the moment a session wants one more directory. This
is the smaller format for that case: a file naming paths to bind ro and rw,
appended AFTER the preset's own list by the caller's `with_binds` -- later
wins is the model's one precedence rule, so appending is exactly "these
shadow the template", and no composition semantics have to be defined::

    # a user spec, in full
    ro = [                       # optional: absent on this host -> dropped
        "~/ro_repo",             # reference material comes and goes
    ]
    rw = [                       # mandatory: missing -> the launch refuses
        "~/rw_repo",
    ]

`~` expands, and a relative path resolves against the FILE's directory, so a
spec checked into a repo can name its neighbours without knowing where the
checkout lives.

The two optionality defaults are the ones the repo measured rather than
picked: a ro reference bind that has vanished is a bind to skip
(`with_reference_docs`), while a named WRITABLE that silently vanishes is a
box quietly missing what the caller asked for. There is deliberately no
per-path `optional` key: when the defaults are wrong for a case, the escape
hatch is still Python, and a data file that grew the vocabulary for it would
be the matching DSL the library explicitly refuses to bless. Same reasoning
for what is absent on purpose: env keys (capability, not mount), removals,
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
from pathlib import Path

from .egress.base import Backend, credential_exposure
from .sandbox import RO, RW, Bind, BindSpec

__all__ = ["load"]


def load(path: Path, *, egress: tuple[Backend, ...] = ()) -> list[BindSpec]:
    """The bind specs a user file names, validated, expanded and guarded.

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

    unknown = sorted(set(doc) - {"ro", "rw"})
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {', '.join(map(repr, unknown))};"
            " expected 'ro' and/or 'rw'"
        )

    # Absent keys default to empty lists -- a file with only `ro` is a file
    # with no writable binds, not a malformed one.
    ro = _entries(path, doc.get("ro", []), "ro")
    rw = _entries(path, doc.get("rw", []), "rw")
    both = [str(p) for p in ro if p in rw]
    if both:
        # There is no order between two keys to resolve this with, and picking
        # one silently would be a precedence rule nobody asked for.
        raise ValueError(f"{path}: {both[0]} appears in both ro and rw -- pick one")

    binds = [Bind(p, RO, optional=True) for p in _dedup(ro)]
    binds += [Bind(p, RW) for p in _dedup(rw)]

    exposure = credential_exposure(tuple(binds), egress)
    if exposure is not None:
        src, backend, cred = exposure
        raise ValueError(
            f"{path}: {src} would expose the {backend.name} backend's"
            f" credential at {cred} -- a user bind may not contain it"
        )
    return binds


def _entries(path: Path, raw: object, key: str) -> list[Path]:
    """`raw` as a list of expanded absolute Paths, or a ValueError naming it.

    Expanded here rather than at bind time so the both-lists and dedupe
    checks compare paths, not spellings: `~/a` and its expansion are the same
    bind, and a file that names one twice in two spellings has still named it
    twice. NOT symlink-resolved -- the credential guard resolves for its own
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
