# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The package and domain boundaries, asserted rather than intended.

aisan is standalone and mostly domain-neutral: modules may import only the
standard library and declared dependencies, and only designated modules may
know what V8 is.

The dependency boundary is checked statically over every module's AST, including
imports on code paths no test runs. A separate subprocess verifies that a fresh
import resolves the installed package without relying on the repository cwd.

The V8-shaped names are checked separately because they describe a different
failure: dependencies affect whether the package installs alone, while a domain
name in a generic module is misplaced knowledge.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import aisan

PACKAGE = Path(aisan.__file__).parent

# Every third-party root imported by the package must correspond to a declared
# dependency. `google` belongs to the optional google-auth extra.
_ALLOWED_IMPORT_ROOTS = {
    *sys.stdlib_module_names,
    "aisan",
    "aiohttp",
    "google",
    "h2",
}

# Names that mean aisan has learned the domain.
_DOMAIN_WORDS = ("v8", "d8", "sisoenv", "vpython", "depot_tools", "gerrit", "chromium")

# The modules allowed to name it, each for a stated reason. Everything NOT in
# here is the extraction's API surface -- the spec, the box, the runtime, the
# launcher, the renderer, the git policy -- and a domain name appearing in one
# of those is the regression this guards.
_DOMAIN_OK = {
    # A preset IS the domain-specific half. That is why presets are a directory
    # of pure functions rather than an `if project == "v8"` inside the spec:
    # the knowledge is concentrated where it can be read, replaced, or left
    # behind. This one ships as the worked example.
    "presets/v8_job.py": "the preset is the designated home for domain knowledge",
    "presets/__init__.py": "the registry names its presets",
    # A backend knows its own upstream by construction -- the vertex one knows
    # what a Vertex endpoint is, this one knows siso reads a .sisoenv. Neither
    # is aisan learning the domain; both are shipped implementations of the
    # Backend protocol, which is the thing that is actually generic.
    "egress/reapi.py": "a backend knows the service it is a backend for",
}


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_has_modules_to_check():
    # The rest of this file is a loop over a glob, and an empty glob passes every
    # assertion in it. Cheap insurance against a rename turning this whole file
    # into a no-op that still shows green.
    assert len(_modules()) >= 8


def test_modules_import_only_declared_dependencies():
    """Statically, over every import statement in the package.

    The AST rather than the import graph, so an import on a path no test
    exercises still counts. Relative imports stay within the package; absolute
    imports must resolve to the standard library or a declared dependency.
    """
    bad: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: within the package, fine.
                names = [node.module or ""] if not node.level else []
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    rel = path.relative_to(PACKAGE)
                    bad.append(f"{rel}:{node.lineno}: {name}")
    assert not bad, "aisan imports undeclared dependencies:\n" + "\n".join(bad)


def test_importing_aisan_standalone_resolves_the_installed_package():
    """Dynamically, in a subprocess, through the INSTALLED package.

    A plain `import aisan`, not a spec loaded off a path: the question is what a
    user gets after installing this package, and only the installed spelling
    answers that. Every module is imported, including the two that package
    initialization defers.
    """
    code = (
        "import sys, json\n"
        "import aisan\n"
        "import aisan.explain, aisan.launch\n"
        "from aisan.presets import PRESETS\n"
        "print(json.dumps(aisan.__file__))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        # Not our cwd: `-c` puts the working directory on sys.path, so running
        # from the repo root would let a directory next to the tests answer
        # `import aisan` instead of the installed distribution. Pinned somewhere
        # that holds neither, and the resolved __file__ is then asserted rather
        # than assumed -- a stale editable install pointing at a tree that no
        # longer exists is a real failure mode, and it looked like a missing
        # module rather than a wrong one.
        cwd=Path(sys.prefix),
    )
    assert out.returncode == 0, (
        "aisan does not import standalone -- which is the whole claim:\n" + out.stderr
    )
    resolved = json.loads(out.stdout)
    assert Path(resolved) == PACKAGE / "__init__.py", (
        f"`import aisan` resolved to {resolved}, not the package under test"
    )


def _code_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Every identifier and string literal in a module, minus its prose.

    Docstrings are excluded and comments never reach the AST, which is the
    distinction this rule needs: a comment explaining that a module is NOT
    V8-shaped is the documentation working, while a `sisoenv` parameter is the
    module having learned V8. Matching raw source text cannot tell those apart
    and would push every explanation out of the package to keep a test green.
    """
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            out.append((node.lineno, node.attr))
        elif isinstance(node, ast.arg):
            out.append((node.lineno, node.arg))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.append((node.lineno, node.name))
        elif isinstance(node, ast.alias):
            out.append((getattr(node, "lineno", 0), node.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.lineno, node.module))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                out.append((node.lineno, node.value))
    return out


def test_only_the_named_modules_know_what_v8_is():
    """The domain rule, and the reason it is separate from the layering one.

    A consumer import is a layering violation: the package stops installing on
    its own. A V8 name is a domain violation: everything still resolves, but
    aisan has learned something only its consumer should know. The V8-specific
    half that deliberately stayed BEHIND in the app (the probe's tool list, the
    assembly annotator, the harness view) is what this keeps from creeping back.

    Substring matching, deliberately: `_v8_utils_ro_binds` and `V8Backend` are
    exactly the names that would show up, and a word-boundary match lets both
    through.
    """
    bad: list[str] = []
    for path in _modules():
        rel = str(path.relative_to(PACKAGE))
        if rel in _DOMAIN_OK:
            continue
        tree = ast.parse(path.read_text(), str(path))
        for lineno, name in _code_names(tree):
            lower = name.lower()
            hit = [w for w in _DOMAIN_WORDS if w in lower]
            if hit:
                bad.append(f"{rel}:{lineno}: {name!r} ({', '.join(hit)})")
    assert not bad, (
        "aisan names the domain outside the modules allowed to:\n" + "\n".join(bad)
    )
