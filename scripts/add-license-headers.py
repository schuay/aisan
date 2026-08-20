#!/usr/bin/env python3
# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Stamp the MIT header onto every source file that can carry a comment.

Run over a repo root; idempotent, so it doubles as the tool you re-run after
adding files. Only formats with `#` comments are touched. JSON and Markdown do
not receive inline headers; the top-level LICENSE covers them.

    scripts/add-license-headers.py .. --check     # CI/pre-commit: report only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER = "# Copyright 2026 The aisan developers\n# SPDX-License-Identifier: MIT\n"

# The marker that makes a re-run a no-op. Matching on the SPDX line rather than
# the whole header means a file whose copyright line was later edited by hand
# (a vendored file, a different holder) is left alone instead of accumulating a
# second header. Anchored to a comment at line start because a file that merely
# names the marker in prose has no header -- this one did exactly that and went
# unstamped.
MARKER = re.compile(r"^# SPDX-License-Identifier:")

# How far into a file a header may start (shebang, or an already-correct header).
HEAD_LINES = 4

SUFFIXES = {".py", ".toml", ".sh", ".service"}
NAMES = {"Dockerfile"}

# Generated, so a header would be clobbered on regeneration.
EXCLUDE_NAMES = {"uv.lock"}


def is_candidate(path: Path) -> bool:
    """Whether this file takes a header at all.

    The single home of that rule, so the pre-commit hook can hand over every
    staged path and let this decide rather than re-encoding the format list in
    bash, where it would drift.
    """
    if path.name in EXCLUDE_NAMES:
        return False
    # Submodule contents are the other repo's business -- it gets its own run.
    if "vendor" in path.parts:
        return False
    return path.suffix in SUFFIXES or path.name in NAMES


def problem(path: Path) -> str | None:
    """The header defect in this file, or None if it is correct."""
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines[:HEAD_LINES]):
        if MARKER.match(line):
            if i + 1 < len(lines) and lines[i + 1].strip():
                return "header not separated"
            return None
    return "missing header"


def targets(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(root / p for p in map(Path, out) if is_candidate(p))


def stamp(path: Path) -> bool:
    """Insert the header. Returns False if the file already had a correct one."""
    text = path.read_text()
    # A shebang has to stay on line 1 or the kernel stops honouring it.
    prefix = ""
    if text.startswith("#!"):
        shebang, _, text = text.partition("\n")
        prefix = f"{shebang}\n"

    lines = text.split("\n")
    # Only the head matters: a file that mentions SPDX further down is quoting
    # it (a test fixture, a doc string), not carrying a header.
    for i, line in enumerate(lines[:HEAD_LINES]):
        if not MARKER.match(line):
            continue
        # Already stamped, so the only thing left to enforce is the separator.
        # This branch is what migrates the files stamped before the blank line
        # was the convention; it is a no-op on everything else.
        if i + 1 < len(lines) and lines[i + 1].strip():
            lines.insert(i + 1, "")
            path.write_text(prefix + "\n".join(lines))
            return True
        return False

    path.write_text(f"{prefix}{HEADER}\n{text}" if text else f"{prefix}{HEADER}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="*", help="repo root(s) to stamp")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report files missing a header, write nothing; exit 1 if any",
    )
    ap.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="check only these paths, skipping non-candidates (pre-commit "
        "hook); implies --check. Pass last, after every other flag",
    )
    ap.add_argument(
        "--fix",
        action="store_true",
        help="with --files: stamp the offenders instead of only naming them, "
        "and still exit 1 so the hook can ask for a re-stage",
    )
    args = ap.parse_args()

    if args.files is not None:
        # An empty --files means the caller's paths were swallowed by another
        # flag rather than that there was nothing to check: a greedy nargs="*"
        # option takes nothing when the next token is an option, and the paths
        # then fall through to the `root` positional. Exiting 0 there would make
        # the hook a silent no-op -- the one failure a guard must not have.
        if not args.files:
            ap.error("--files got no paths; pass them last, after every flag")
        bad = 0
        # A staged path can be gone by now (renamed, or a hook run on a partial
        # index), which is not a header defect -- skip rather than crash.
        for path in args.files:
            if not is_candidate(path) or not path.is_file():
                continue
            if defect := problem(path):
                print(f"{defect}: {path}")
                bad += 1
                if args.fix:
                    stamp(path)
        # Nonzero even after --fix: the fix landed in the worktree, not the
        # index, so committing now would still record the unstamped blob.
        return 1 if bad else 0

    if not args.root:
        ap.error("give a repo root, or --files")

    missing = 0
    for base in args.root:
        root = base.resolve()
        changed = 0
        for path in targets(root):
            if args.check:
                if defect := problem(path):
                    print(f"{defect}: {path.relative_to(root)}")
                    missing += 1
            elif stamp(path):
                changed += 1
        if not args.check:
            print(f"{root.name}: stamped {changed} file(s)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
