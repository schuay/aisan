# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Load user-written bind specs from a six-key TOML format.

The format adds mounts without requiring users to copy and modify a launcher.
Callers add the result to a preset. Every mount from a spec file is plain, and
overlaps resolve as `sandbox` describes: the deeper path wins below it, the
stricter mode wins at one path, and a plain mount at or below a guard the
preset owns is refused::

    # a user spec, in full
    ro = [                       # optional: absent on this host -> dropped
        "~/ro_repo",             # reference material comes and goes
    ]
    rw = [                       # mandatory: missing -> the launch refuses
        "~/rw_repo",
        "~/ro_repo/scratch",     # a writable hole inside a read-only mount
    ]
    overlay = [                  # mandatory: warm to read, writable, discarded
        "~/.cache/some-tool-store",
    ]
    path = [                     # prepended to the box PATH
        "~/tooling",             # must be covered by a mount named above
    ]
    mcp = [                      # host MCP servers this box may start
        "v8-mcp",                # the launcher binary by name ...
        "~/tools/nvim-mcp/bin/nv",  # ... or by full path
    ]
    include = [                  # other spec files, merged with this one
        "./base-userbinds.toml",
    ]

`~` expands to the user's home. Relative paths resolve from the spec file's
directory, allowing checked-in specs to refer to neighboring repositories.
Included files use the same resolution.

Nesting across keys is how a file states a mixed tree: `rw` above `ro` pins a
subtree, `ro` above `rw` opens a hole. Neither the order of keys nor the order
of files carries meaning. Two files that name one path with different modes
get the stricter one; that is not an error because it is what a stricter
include is for.

An explicit include list keeps the composed set visible in the file instead of
the operator's command line. Diamond includes apply a shared file once. Direct
or transitive include cycles are rejected with the full chain.

The repository's measured use cases determine optionality. A missing read-only
reference is skipped, while a missing writable path is an error because the box
would lack a requested output location. Overlays are also mandatory because an
absent shared tool cache can cause an offline rebuild to hang. Cases requiring
different behavior use the Python API; the TOML format has no per-path
`optional` setting.

`mcp` names the host-declared MCP servers the box may start. Without it the
box starts none. A host client config is one shared list that grows as tools
are added for interactive work, and an unattended box should not gain a
channel because of an entry written for an attended one.

An entry matches a declaration by its config name, by the basename of its
command, or by the command's full path. `~` expands; a relative path is
rejected because the spec file's directory is not where a command resolves.
A command admits every declaration that runs it, so a multiplexer such as
`npx` admits each server it launches; name those by config name.
The single entry `*` admits every local stdio declaration, which is the
behavior from before this key existed. Entries union across the include tree,
so a `*` anywhere in it admits everything.

A full path selects, it does not mount. The launcher binds still come from
`session_mcp`, under its tool-root and credential rules, and a spec can only
narrow what the host already declares. A name here therefore carries no
capability that an unfiltered import would not have given.

`path` accepts only directories. Every entry must be covered by a mount from the
merged include tree, so it adds no filesystem
access. It only makes an already mounted tool discoverable without symlinks that
could break argv0-relative bootstrap logic.

The format excludes arbitrary environment values, removals, reordering, globs,
`Seal`, and `BindOver`. Those operations require reviewed Python policy or may
carry capabilities that a mount-only file cannot reveal. Includes name files
explicitly so readers can see the complete set and its order.

`load` accepts the egress tuple so it can reject user mounts that overlap a
backend credential. It rejects the written source even if a later mount would
hide the credential. `Box._sandbox` separately evaluates the finished mount
order, covering both user files and hand-built specs.

The guard also rejects credentials owned by other known backends. For example,
a Claude box must keep `~/.codex` out even when Codex isn't in its egress tuple.
The same rule covers `~/.ssh` and `~/.gnupg`.

This loader does not add Git-specific binds. Interactive sessions treat the
entire root as belonging to the session. Hand-built specs that need a linked
worktree's external Git metadata can add `git_binds`, as demonstrated by
`aisan-claude-specs.py`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .egress import known_credential_paths
from .egress.base import Backend, credential_exposure, credential_overlap
from .sandbox import RO, RW, Bind, BindSpec, Overlay

__all__ = ["UserSpec", "load"]

# Always refuse the host's SSH and GnuPG private-key stores. Neither belongs in
# an agent box; host-side Git or a purpose-specific deploy key handles pushes.
# This is intentionally narrower than a general list of sensitive directories.
_REFUSED_KEY_STORES = (".ssh", ".gnupg")


def _refused_key_stores() -> tuple[Path, ...]:
    """Return key-store paths using the current home directory."""
    return tuple(Path.home() / name for name in _REFUSED_KEY_STORES)


# Keys that produce mounts. `path` adds no mount and must be covered by one of
# these entries.
_MOUNT_KEYS = ("overlay", "ro", "rw")

# Complete set of accepted keys. `include` names other spec files, and `mcp`
# names host MCP servers rather than paths.
_KEYS = (*_MOUNT_KEYS, "path", "include", "mcp")

# Admits every local stdio declaration. The format has no other glob.
WILDCARD = "*"


@dataclass(frozen=True)
class UserSpec:
    """The mounts, covered PATH entries and MCP selectors of one include tree.

    Callers apply the mount fields through `with_binds` and `with_path_prefix`
    after parsing the file once. `mcp` goes to `session_mcp`, which matches it
    against the host declarations.
    """

    binds: list[BindSpec]
    path: tuple[Path, ...]
    mcp: tuple[str, ...] = ()


def load(path: Path, *, egress: tuple[Backend, ...]) -> UserSpec:
    """Load and validate mounts and PATH entries from a user spec.

    Malformed input raises `ValueError` naming the file and key. A missing file
    propagates `FileNotFoundError` because the caller supplied the wrong path.

    `egress` is mandatory because omitting it would disable credential checks.
    Callers without backends pass an explicit empty tuple.
    """
    spec = _load(path, egress, [], set())
    _assert_path_coverage(path, spec)
    return spec


def _assert_path_coverage(path: Path, spec: UserSpec) -> None:
    """Require every PATH directory to lie below an emitted mount.

    Check the merged include tree so a shared include can cover entries from
    either branch. Normalize paths before containment checks to prevent `..`
    components from appearing covered while escaping the mount. User binds mount
    each source at the same path inside the box.
    """
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
    """Load one file while tracking include ancestry and prior expansions.

    `chain` supports cycle detection and diagnostics. `expanded` ensures a
    diamond include applies its shared file once.
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
    inner_mcp: list[str] = []
    for spec_file in _entries(path, doc.get("include", []), "include"):
        # A file expanded through another branch is a duplicate. A file in the
        # active chain must reach `_load` so the cycle is reported.
        target = spec_file.resolve()
        if target in expanded and target not in (*chain, here):
            continue
        if not spec_file.is_file():
            # Name both sides of a missing include relationship.
            raise ValueError(f"{path}: include {spec_file} is not a file")
        inner = _load(spec_file, egress, [*chain, here], expanded)
        inner_binds += inner.binds
        inner_dirs += inner.path
        inner_mcp += inner.mcp

    # Omitted keys represent empty lists.
    mounts = {key: _entries(path, doc.get(key, []), key) for key in _MOUNT_KEYS}
    for (a, first), (b, second) in combinations(mounts.items(), 2):
        both = [str(p) for p in first if p in second]
        if both:
            # One path cannot have two modes within the same file.
            raise ValueError(
                f"{path}: {both[0]} appears in both {a} and {b} -- pick one"
            )

    # Spec mounts are plain: another spec entry may sit below any of them.
    binds: list[BindSpec] = [*inner_binds]
    binds += [Overlay(p, guard=False) for p in _dedup(mounts["overlay"])]
    binds += [Bind(p, RO, optional=True, guard=False) for p in _dedup(mounts["ro"])]
    binds += [Bind(p, RW, guard=False) for p in _dedup(mounts["rw"])]

    dirs = _dedup([*inner_dirs, *_entries(path, doc.get("path", []), "path")])
    # Includes first, like mounts, so one spec can add to what it includes.
    servers = tuple(
        dict.fromkeys([*inner_mcp, *_mcp_entries(path, doc.get("mcp", []))])
    )
    # Check PATH coverage once after merging. Per-file checks would make diamond
    # includes depend on which branch expanded the shared file first.

    exposure = credential_exposure(tuple(binds), egress)
    if exposure is not None:
        src, backend, cred = exposure
        raise ValueError(
            f"{path}: {src} would expose the {backend.name} backend's"
            f" credential at {cred} -- a user bind may not name it, nor"
            " anything containing it or inside it"
        )
    # Also protect credentials belonging to backends absent from this box.
    sources = [
        src
        for spec in binds
        if (src := getattr(spec, "path", None) or getattr(spec, "src", None))
        is not None
    ]
    other = credential_overlap(sources, known_credential_paths())
    if other is not None:
        src, cred = other
        raise ValueError(
            f"{path}: {src} would expose the credential at {cred} -- it belongs"
            " to a backend this box does not carry, but a box that can read it"
            " can use it"
        )
    keys = credential_overlap(sources, _refused_key_stores())
    if keys is not None:
        src, store = keys
        raise ValueError(
            f"{path}: {src} would expose the private key store at {store} --"
            " a user bind may not name it, nor anything containing it or"
            " inside it"
        )
    return UserSpec(binds, tuple(dirs), servers)


def _entries(path: Path, raw: object, key: str) -> list[Path]:
    """Return `raw` as expanded absolute paths, or raise a named `ValueError`.

    Expand paths before overlap and duplicate checks so equivalent spellings
    compare equally. Preserve symlink spelling because it determines the mount
    destination; credential checks resolve aliases separately.
    """
    if not isinstance(raw, list) or not all(
        isinstance(e, str) and e.strip() for e in raw
    ):
        raise ValueError(f"{path}: {key} must be an array of non-empty path strings")
    out = []
    for entry in raw:
        # A path separator would become a second unchecked PATH entry when joined.
        if os.pathsep in entry:
            raise ValueError(f"{path}: {key} entry {entry!r} contains {os.pathsep!r}")
        try:
            p = Path(entry).expanduser()
        except RuntimeError as e:
            # Convert failed user expansion into the loader's named error form.
            raise ValueError(
                f"{path}: {key} entry {entry!r} cannot be expanded: {e}"
            ) from e
        if not p.is_absolute():
            p = path.parent / p
        out.append(p)
    return out


def _mcp_entries(path: Path, raw: object) -> list[str]:
    """Return validated MCP selectors, or raise a named `ValueError`.

    Entries stay as written because `session_mcp` resolves them against the
    host declarations and the box PATH. Only the spelling is checked here: a
    relative path would otherwise look like it resolves from the spec file's
    directory, which is not where a command is found.
    """
    if not isinstance(raw, list) or not all(
        isinstance(e, str) and e.strip() for e in raw
    ):
        raise ValueError(f"{path}: mcp must be an array of non-empty strings")
    for entry in raw:
        if entry == WILDCARD:
            continue
        if WILDCARD in entry:
            raise ValueError(
                f"{path}: mcp entry {entry!r} contains {WILDCARD!r}; the only"
                f" glob is the single entry {WILDCARD!r}, which admits every"
                " local stdio server the host declares"
            )
        if os.pathsep in entry:
            raise ValueError(f"{path}: mcp entry {entry!r} contains {os.pathsep!r}")
        if "/" in entry and not Path(entry).expanduser().is_absolute():
            raise ValueError(
                f"{path}: mcp entry {entry!r} is a relative path -- name the"
                " server, its command, or the command's absolute path"
            )
    return list(raw)


def _dedup(paths: list[Path]) -> list[Path]:
    """Return unique paths in first-occurrence order."""
    seen: set[Path] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]
