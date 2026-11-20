# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
from pathlib import Path

import pytest

from aisan import Box
from aisan.explain import explain, normalise
from aisan.presets import PRESETS
from aisan.presets.depot_tools_job import depot_tools_job, sisoenv_path

SNAPSHOTS = Path(__file__).parent / "snapshots_seatbelt"

_PINNED_SYSTEM_SUBPATHS = (
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/System"),
    Path("/Library/Apple"),
    Path("/private/etc/ssl"),
    Path("/private/var/select"),
    Path("/private/var/db/timezone"),
    Path("/dev/fd"),
    Path("/Library/Developer"),
    Path("/opt/homebrew/bin"),
    Path("/opt/homebrew/lib"),
    Path("/opt/homebrew/Cellar"),
)

_PINNED_SYSTEM_LITERALS = (
    Path("/"),
    Path("/dev"),
    Path("/private/etc/hosts"),
    Path("/private/etc/passwd"),
    Path("/private/etc/group"),
    Path("/private/etc/services"),
    Path("/private/etc/protocols"),
    Path("/dev/null"),
    Path("/dev/zero"),
    Path("/dev/random"),
    Path("/dev/urandom"),
    Path("/dev/tty"),
    Path("/dev/ptmx"),
)


def _checkout(tmp_path: Path) -> Path:
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    for d in ("build/config/siso", "third_party/icu", ".git/worktrees/wt"):
        (main / d).mkdir(parents=True)
    (main / "build/config/siso/.sisoenv").write_text(
        "SISO_PROJECT=rbe-chromium-untrusted\nSISO_REAPI_INSTANCE=default_instance\n"
    )
    (wt / "third_party").mkdir(parents=True)
    (wt / "out").mkdir()
    (wt / "build").symlink_to(main / "build")
    (wt / "third_party" / "icu").symlink_to(main / "third_party" / "icu")
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    return wt


def _check(name: str, text: str) -> None:
    path = SNAPSHOTS / f"{name}.darwin.txt"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), (
        f"no Darwin snapshot for {name}; review the profile below, then"
        f" regenerate with UPDATE_SNAPSHOTS=1\n\n{text}"
    )
    assert text == path.read_text(), (
        f"the Darwin {name} profile changed. Read the diff and confirm the"
        " Seatbelt policy is intended, then regenerate with UPDATE_SNAPSHOTS=1"
    )


@pytest.fixture
def pinned_darwin_host(monkeypatch):
    raw_tmp = (
        Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/var/tmp")
    )
    sys_tmp = str(raw_tmp.resolve())
    monkeypatch.setenv("HOME", "/Users/snapshot-user")
    monkeypatch.setattr(tempfile, "tempdir", sys_tmp)

    policy_tree_mod = importlib.import_module("aisan.policy_tree")
    monkeypatch.setattr(policy_tree_mod.sys, "platform", "darwin")
    monkeypatch.setattr(policy_tree_mod, "_f_getpath", lambda _p: None)
    egress_base_mod = importlib.import_module("aisan.egress.base")
    monkeypatch.setattr(egress_base_mod.sys, "platform", "darwin")
    monkeypatch.setattr(egress_base_mod, "_SUN_PATH_BYTES", 103)

    seatbelt_mod = importlib.import_module("aisan.seatbelt")
    monkeypatch.setattr(
        seatbelt_mod, "_active_system_subpaths", lambda: _PINNED_SYSTEM_SUBPATHS
    )
    monkeypatch.setattr(
        seatbelt_mod, "_DARWIN_SYSTEM_LITERALS", _PINNED_SYSTEM_LITERALS
    )
    monkeypatch.setattr(seatbelt_mod, "darwin_controlling_ttys", lambda: ())
    # Stub existence, symlink, and resolution checks on pinned system roots so
    # usrmerge Linux (`/bin -> /usr/bin`, `/dev/fd -> /proc/self/fd`) and macOS
    # hosts without `/opt/homebrew` or `/Library/Developer` render identical SBPL.
    pinned_set = frozenset(_PINNED_SYSTEM_SUBPATHS) | frozenset(_PINNED_SYSTEM_LITERALS)
    real_exists = Path.exists
    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if self in pinned_set else real_exists(self),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: False if self in pinned_set else real_is_symlink(self),
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: (
            self if self in pinned_set else real_resolve(self, *args, **kwargs)
        ),
    )

    # Root both `_PRIVATE_ROOT` and test workspaces under `sys_tmp` (`/private/tmp`
    # on Darwin, `/var/tmp` on Linux) so Seatbelt's ancestor metadata chain has
    # identical depth regardless of `TMPDIR` or pytest `--basetemp`.
    private = importlib.import_module("aisan.private")
    with (
        tempfile.TemporaryDirectory(prefix="aisan-sn-", dir=sys_tmp) as pdir,
        tempfile.TemporaryDirectory(prefix="aisan-ws-", dir=sys_tmp) as wdir,
    ):
        monkeypatch.setattr(private, "_PRIVATE_ROOT", Path(pdir).resolve())
        work = Path(wdir).resolve()

        preset_dt = importlib.import_module("aisan.presets.depot_tools_job")
        cache = work / "vpython-cache" / "vpython-root"
        cache.mkdir(parents=True)
        monkeypatch.setattr(preset_dt, "_vpython_cache", lambda **_: cache)
        monkeypatch.setattr(preset_dt, "_active_xcode_dev_dir", lambda: None)
        depot = work / "depot_tools"
        depot.mkdir()
        monkeypatch.setattr(preset_dt.shutil, "which", lambda cmd: str(depot / cmd))

        preset_codex = importlib.import_module("aisan.presets.codex")
        monkeypatch.setattr(
            preset_codex,
            "_CODEX_APP_CLI",
            Path(
                "/Applications/NonexistentChatGPTForSnapshot.app/Contents/Resources/cli"
            ),
        )

        launch = importlib.import_module("aisan.launch")
        from aisan.sandbox import RO, Bind

        prefix = work / "aisan-prefix"
        (prefix / "bin").mkdir(parents=True)
        fake_py = prefix / "bin" / "python3"
        fake_py.write_text("")
        src = work / "aisan-src"
        src.mkdir()
        monkeypatch.setattr(launch.sys, "executable", str(fake_py))
        monkeypatch.setattr(
            launch, "launcher_binds", lambda **_: [Bind(prefix, RO), Bind(src, RO)]
        )

        opencode = importlib.import_module("aisan.presets.opencode")
        catalog = work / "opencode" / "models.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_text("{}")
        monkeypatch.setattr(opencode, "_default_catalog", lambda: catalog)
        yield depot


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
            (wt.parent, "<CASE_TMP>"),
            (depot.parent, "<TEST_TMP>"),
            *extra_paths,
        ),
    )


def test_darwin_depot_tools_job_snapshot(pinned_darwin_host):
    work = pinned_darwin_host.parent
    wt = _checkout(work)
    spec = depot_tools_job(wt, depot_tools=pinned_darwin_host, unshare_net=True)
    _check("depot_tools_job", _report(wt, pinned_darwin_host, spec))


def test_darwin_depot_tools_job_rbe_snapshot(pinned_darwin_host):
    from aisan.egress.reapi import ReapiBackend

    work = pinned_darwin_host.parent
    wt = _checkout(work)
    backend = ReapiBackend(
        project="rbe-chromium-untrusted",
        sisoenv=sisoenv_path(wt),
        mint=lambda: asyncio.sleep(0, result="tok"),
    )
    spec = depot_tools_job(
        wt,
        depot_tools=pinned_darwin_host,
        egress=(backend,),
        unshare_net=True,
    )
    _check("depot_tools_job_rbe", _report(wt, pinned_darwin_host, spec))


def test_darwin_claude_code_snapshot(pinned_darwin_host):
    from aisan.egress.anthropic import AnthropicBackend
    from aisan.presets.claude_code import claude_code

    work = pinned_darwin_host.parent
    wt = _checkout(work)
    state = work / "state"
    state.mkdir()
    config = work / "config"
    config.mkdir()
    skills = work / "skills"
    skills.mkdir()
    spec = claude_code(
        wt,
        state=state,
        config=config,
        skills=skills,
        egress=(AnthropicBackend(credentials=work / "absent.json"),),
    )
    _check(
        "claude_code",
        _report(
            wt,
            pinned_darwin_host,
            spec,
            preset="claude_code",
            extra_paths=(
                (state, "<STATE>"),
                (config, "<CONFIG>"),
                (skills, "<SKILLS>"),
            ),
        ),
    )


def test_darwin_codex_snapshot(pinned_darwin_host):
    from aisan.egress.openai_responses import CodexBackend
    from aisan.presets.codex import codex

    work = pinned_darwin_host.parent
    wt = _checkout(work)
    state = work / "state"
    state.mkdir()
    backend = CodexBackend(credentials=work / "absent.json", codex_home=state)
    spec = codex(wt, state=state, egress=(backend,))
    _check(
        "codex",
        _report(
            wt,
            pinned_darwin_host,
            spec,
            preset="codex",
            extra_paths=((state, "<STATE>"),),
        ),
    )


def test_darwin_antigravity_snapshot(pinned_darwin_host):
    from aisan.egress.antigravity import AntigravityBackend
    from aisan.egress.antigravity_keyring import KeyringCredential
    from aisan.presets.antigravity import TOKEN_FILE, antigravity

    work = pinned_darwin_host.parent
    wt = _checkout(work)
    state = work / "state"
    state.mkdir()
    backend = AntigravityBackend(
        credential=KeyringCredential(), token_file=state / TOKEN_FILE
    )
    spec = antigravity(wt, state=state, egress=(backend,))
    _check(
        "antigravity",
        _report(
            wt,
            pinned_darwin_host,
            spec,
            preset="antigravity",
            extra_paths=((state, "<STATE>"),),
        ),
    )


def test_darwin_every_registered_preset_has_a_snapshot(pinned_darwin_host):
    work = pinned_darwin_host.parent
    for name, build in sorted(PRESETS.items()):
        wt = _checkout(work / name)
        _check(
            f"registry_{name}",
            _report(
                wt,
                pinned_darwin_host,
                build(wt),
                preset=name,
                extra_paths=((work / name / "main", "<MAIN>"),),
            ),
        )
