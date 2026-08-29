# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The audit artifact: one snapshot of the resolved profile, per preset.

Everything else in the aisan suite asserts a property somebody thought to state.
This asserts the WHOLE profile, which is the only thing that catches a mount
nobody thought about -- an `extra_ro` that quietly widened, a pin that stopped
landing after the rw bind it guards, a tmpfs that a later bind shadows. A diff
here is not a failure to fix by regenerating: it is the review artifact, and the
question it asks is "did you mean to change what the box can touch".

Over `explain` output rather than over the BoxSpec, because the spec is the
input and the box is what the box ENDS UP with -- order, hoisting, deferred
remounts, optional sources that do or do not exist. Those are exactly where a
confinement bug lives, and none of them is visible in the spec.

The checkout is synthetic and the preset's host-dependent inputs are pinned
(depot_tools, the vpython cache), so what is left in the file is policy. See
`normalise` for the four host facts that are replaced and the reasoning for
each; the rest is verbatim on purpose.

Regenerate with:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_aisan_snapshot.py

and read the resulting diff before committing it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from aisan import Box
from aisan.explain import explain, normalise
from aisan.presets import PRESETS
from aisan.presets.v8_job import v8_job

SNAPSHOTS = Path(__file__).parent / "snapshots"


def _checkout(tmp_path: Path) -> Path:
    """A main checkout + linked worktree, the shape the preset expects.

    Synthetic rather than a real checkout: a real one differs per machine in
    which deps are symlinked and which .git files exist, and a snapshot of that
    would be a snapshot of somebody's disk.
    """
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
    """Compare against the stored snapshot, or write it under UPDATE_SNAPSHOTS.

    A missing snapshot FAILS rather than silently writing itself: a new preset
    landing with an auto-generated profile nobody read is precisely the review
    this file exists to force.
    """
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
    """The preset's host-dependent inputs, made the same everywhere.

    Two of them, both discovered rather than passed: the vpython cache (an
    Overlay, present only on a host that has run `git cl`) and depot_tools
    (found on PATH). Left alone, the snapshot would carry two mounts that appear
    and vanish with the machine -- and the case that matters is the one where
    they are PRESENT, since that is the complete profile.
    """
    # import_module, not `from ... import v8_job`: the presets package re-exports
    # the FUNCTION under its module's name, so the plain import binds a callable
    # with no module attributes to patch.
    preset = importlib.import_module("aisan.presets.v8_job")

    cache = tmp_path / "vpython-cache" / "vpython-root"
    cache.mkdir(parents=True)
    monkeypatch.setattr(preset, "_vpython_cache", lambda: cache)
    depot = tmp_path / "depot_tools"
    depot.mkdir()
    monkeypatch.setattr(preset.shutil, "which", lambda cmd: str(depot / cmd))
    # The opencode preset's provider catalog: an OPTIONAL bind at the host's
    # real cache path, so without pinning it the snapshot would carry one mount
    # on a host that has run opencode and none on one that has not. Present,
    # because the complete profile includes it. Substituted as
    # "<OPENCODE_CACHE>" in _report.
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
    preset: str = "v8_job",
    extra_paths: tuple[tuple[Path, str], ...] = (),
) -> str:
    """The normalised report for `spec`, with this run's own paths named.

    Three beyond the host facts `normalise` knows: the sibling MAIN checkout the
    dep symlinks and .git resolve into, the pinned depot_tools, and the vpython
    cache. On a real host all three sit under $HOME and are covered; under
    tmp_path they are not, and an unnamed one puts a pytest run counter in the
    snapshot -- which fails on the next run for a reason nobody caused.

    `preset` is the name the report leads with. A parameter rather than a
    constant since the registry loop covers more than one: hardcoded, every
    snapshot claimed to be v8_job, and an audit artifact that misnames its own
    subject is worse than one that omits it.
    """
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
            *extra_paths,
        ),
    )


def test_v8_job_profile_snapshot(tmp_path, pinned_host):
    """The preset as the app runs it: worktree, depot_tools, no egress.

    The registry entry (`v8_job_default`) is deliberately egress-less so it can
    be described on a host with no deployment; this calls the preset directly
    with the arguments the app passes, so the snapshot covers the arguments that
    exercise the complete profile rather than the dry-run defaults.
    """
    wt = _checkout(tmp_path)
    spec = v8_job(wt, depot_tools=pinned_host, unshare_net=True)
    _check("v8_job", _report(wt, pinned_host, spec))


def test_claude_code_profile_snapshot(tmp_path, pinned_host):
    """The Claude Code preset with the arguments a real caller passes.

    The registry entry is egress-less and roots its state inside the worktree,
    both so a dry run needs no deployment; this is the other shape -- a state dir
    of its own, a ro config dir, and the Anthropic backend -- so the artifact
    covers the two extra binds and the client env omitted by the dry-run shape.

    The backend is constructed with a credentials path that does not exist. It
    is never read here (nothing runs), and naming the real one would make the
    snapshot depend on the host's login state.
    """
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
            # Both live under this run's tmp_path, which would otherwise put a
            # pytest run counter in the snapshot. Through `normalise` rather than
            # a str.replace afterwards: by then it has already rewritten the
            # temp root, so the raw paths no longer appear to match.
            extra_paths=((state, "<STATE>"), (config, "<CONFIG>")),
        ),
    )


def test_codex_profile_snapshot(tmp_path, pinned_host):
    """The Codex preset with writable isolated state and no config bind."""
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
    """The registry is the list of presets an operator can explain, so it is the
    list that needs an audit artifact. Asserted as a loop over PRESETS rather
    than as one test per preset, because the failure worth catching is a preset
    added WITHOUT a snapshot -- which a hand-written test list cannot notice."""
    for name, build in sorted(PRESETS.items()):
        wt = _checkout(tmp_path / name)
        _check(f"registry_{name}", _report(wt, pinned_host, build(wt), preset=name))
