#!/usr/bin/env python3
# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Add the MIT header to tracked source files that support comments.

The operation is idempotent. JSON and Markdown rely on the top-level LICENSE.

    scripts/add-license-headers.py .. --check     # CI/pre-commit: report only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER = "# Copyright 2026 The aisan developers\n# SPDX-License-Identifier: MIT\n"

# Match the anchored SPDX line so edited copyright holders don't receive a
# duplicate header. Mentions of SPDX elsewhere in the file don't count.
MARKER = re.compile(r"^# SPDX-License-Identifier:")

# Allow room for a shebang before the header.
HEAD_LINES = 4

SUFFIXES = {".py", ".toml", ".sh", ".service"}
NAMES = {"Dockerfile"}

# Generated files lose manual headers during regeneration.
EXCLUDE_NAMES = {"uv.lock"}


def is_candidate(path: Path) -> bool:
    """Return whether ``path`` should carry a header."""
    if path.name in EXCLUDE_NAMES:
        return False
    # Vendored sources retain their upstream headers.
    if "vendor" in path.parts:
        return False
    return path.suffix in SUFFIXES or path.name in NAMES


def problem(path: Path) -> str | None:
    """Return the header defect in ``path``, if any."""
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines[:HEAD_LINES]):
        if MARKER.match(line):
            if i + 1 < len(lines) and lines[i + 1].strip():
                return "header not separated"
            return None
    return "missing header"


def targets(root: Path) -> list[Path]:
    # NUL separation preserves spaces. is_file() drops deleted index entries and
    # submodules.
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return sorted(
        root / p
        for p in map(Path, filter(None, out))
        if is_candidate(p) and (root / p).is_file()
    )


def stamp(path: Path) -> bool:
    """Insert or repair the header and report whether the file changed."""
    text = path.read_text()
    # The kernel requires the shebang on the first line.
    prefix = ""
    if text.startswith("#!"):
        shebang, _, text = text.partition("\n")
        prefix = f"{shebang}\n"

    lines = text.split("\n")
    # SPDX text below the header area may be documentation or test data.
    for i, line in enumerate(lines[:HEAD_LINES]):
        if not MARKER.match(line):
            continue
        # Add the separator required after an existing header.
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
        # Empty --files usually means argparse assigned paths to another greedy
        # option. Refuse instead of silently skipping the hook.
        if not args.files:
            ap.error("--files got no paths; pass them last, after every flag")
        bad = 0
        # Skip staged paths deleted or renamed before the hook runs.
        for path in args.files:
            if not is_candidate(path) or not path.is_file():
                continue
            if defect := problem(path):
                print(f"{defect}: {path}")
                bad += 1
                if args.fix:
                    stamp(path)
        # --fix changes the worktree; fail so callers re-stage the file.
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
