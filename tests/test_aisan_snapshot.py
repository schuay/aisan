# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from aisan import Box
from aisan.explain import explain, normalise
from aisan.presets import PRESETS
from aisan.presets.depot_tools_job import depot_tools_job

SNAPSHOTS = Path(__file__).parent / "snapshots"


def _checkout(tmp_path: Path) -> Path:
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    for d in ("build", "third_party/icu", ".git/worktrees/wt"):
        (main / d).mkdir(parents=True)
    (wt / "third_party").mkdir(parents=True)
    (wt / "out").mkdir()
    (wt / "build").symlink_to(main / "build")
    (wt / "third_party" / "icu").symlink_to(main / "third_party" / "icu")
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    return wt


def _check(name: str, text: str) -> None:
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), (
        f"no snapshot for {name}; review the profile below, then regenerate with"
        f" UPDATE_SNAPSHOTS=1\n\n{text}"
    )
    assert text == path.read_text(), (
        f"the {name} profile changed. This is the audit diff -- read it and"
        " confirm the box is meant to be able to touch what it now touches,"
        " then regenerate with UPDATE_SNAPSHOTS=1"
    )


@pytest.fixture
def pinned_host(tmp_path, monkeypatch):

    preset = importlib.import_module("aisan.presets.depot_tools_job")

    private = importlib.import_module("aisan.private")
    monkeypatch.setattr(private, "_PRIVATE_ROOT", Path("/tmp/aisan-snapshot"))

    cache = tmp_path / "vpython-cache" / "vpython-root"
    cache.mkdir(parents=True)
    monkeypatch.setattr(preset, "_vpython_cache", lambda: cache)
    depot = tmp_path / "depot_tools"
    depot.mkdir()
    monkeypatch.setattr(preset.shutil, "which", lambda cmd: str(depot / cmd))

    # Pin the launcher layout. These binds vary in number as well as in path
    # across installations, so the snapshot needs a fixed layout rather than an
    # elision in the report.
    launch = importlib.import_module("aisan.launch")
    from aisan.sandbox import RO, Bind

    prefix = tmp_path / "aisan-prefix"
    (prefix / "bin").mkdir(parents=True)
    src = tmp_path / "aisan-src"
    src.mkdir()
    monkeypatch.setattr(
        launch, "launcher_binds", lambda: [Bind(prefix, RO), Bind(src, RO)]
    )

    opencode = importlib.import_module("aisan.presets.opencode")
    catalog = depot.parent / "opencode" / "models.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")
    monkeypatch.setattr(opencode, "_default_catalog", lambda: catalog)
    return depot


def _report(
    wt: Path,
    depot: Path,
    spec,
    preset: str = "depot_tools_job",
    extra_paths: tuple[tuple[Path, str], ...] = (),
) -> str:
    box = Box(spec, box_id="snapshot")
    with box.staged():
        text = explain(box, inputs=(("preset", preset),))
    return normalise(
        text,
        root=wt,
        paths=(
            (wt.parent / "main", "<MAIN>"),
            (depot, "<DEPOT_TOOLS>"),
            (depot.parent / "vpython-cache", "<VPYTHON_CACHE>"),
            (depot.parent / "opencode", "<OPENCODE_CACHE>"),
            (depot.parent / "aisan-prefix", "<AISAN PREFIX>"),
            (depot.parent / "aisan-src", "<AISAN SRC>"),
            *extra_paths,
        ),
    )


def test_depot_tools_job_profile_snapshot(tmp_path, pinned_host):
    wt = _checkout(tmp_path)
    spec = depot_tools_job(wt, depot_tools=pinned_host, unshare_net=True)
    _check("depot_tools_job", _report(wt, pinned_host, spec))


def test_claude_code_profile_snapshot(tmp_path, pinned_host):
    from aisan.egress.anthropic import AnthropicBackend
    from aisan.presets.claude_code import claude_code

    wt = _checkout(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    spec = claude_code(
        wt,
        state=state,
        config=config,
        egress=(AnthropicBackend(credentials=tmp_path / "absent.json"),),
    )
    _check(
        "claude_code",
        _report(
            wt,
            pinned_host,
            spec,
            preset="claude_code",
            extra_paths=((state, "<STATE>"), (config, "<CONFIG>")),
        ),
    )


def test_codex_profile_snapshot(tmp_path, pinned_host):
    from aisan.egress.openai_responses import CodexBackend
    from aisan.presets.codex import codex

    wt = _checkout(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexBackend(credentials=tmp_path / "absent.json")
    spec = codex(wt, state=state, egress=(backend,))
    _check(
        "codex",
        _report(
            wt,
            pinned_host,
            spec,
            preset="codex",
            extra_paths=((state, "<STATE>"),),
        ),
    )


def test_every_registered_preset_has_a_snapshot(tmp_path, pinned_host):
    for name, build in sorted(PRESETS.items()):
        wt = _checkout(tmp_path / name)
        _check(f"registry_{name}", _report(wt, pinned_host, build(wt), preset=name))
