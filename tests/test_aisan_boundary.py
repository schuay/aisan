# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import aisan

PACKAGE = Path(aisan.__file__).parent


_ALLOWED_IMPORT_ROOTS = {
    *sys.stdlib_module_names,
    "aisan",
    "aiohttp",
    "google",
    "h2",
}


_DOMAIN_WORDS = ("v8", "d8", "sisoenv", "vpython", "depot_tools", "gerrit", "chromium")


_DOMAIN_OK = {
    "presets/depot_tools_job.py": "the preset is the designated home for domain knowledge",
    "presets/__init__.py": "the registry names its presets",
    "egress/reapi.py": "a backend knows the service it is a backend for",
}


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_has_modules_to_check():

    assert len(_modules()) >= 8


def test_modules_import_only_declared_dependencies():
    bad: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
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
