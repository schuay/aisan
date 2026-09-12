# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from aisan import Box
from aisan.egress.reapi import ReapiBackend
from aisan.presets.depot_tools_job import (
    depot_tools_job,
    sisoenv_path,
    sisoenv_paths,
    v8_rbe,
    v8_rbe_shared_net,
)
from aisan.proxy.rbe import UPSTREAM_HOST
from aisan.sandbox import RO

PROJECT = "rbe-chromium-untrusted"


def _checkout(tmp_path: Path) -> Path:
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "config").touch()
    wt = tmp_path / "wt"
    (wt / "build" / "config" / "siso").mkdir(parents=True)

    (wt / "build" / "config" / "siso" / ".sisoenv").write_text(
        "SISO_PROJECT=not-the-override\n"
    )
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (main / ".git" / "worktrees" / "wt" / "gitdir").write_text(f"{wt}/.git\n")
    return wt


def _box(wt: Path, tmp_path: Path) -> Box:
    spec = depot_tools_job(
        wt, egress=(_backend(wt),), unshare_net=True, depot_tools=None
    )
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    return Box(spec, box_id=str(tmp_path / "job"))


async def _in_box(box: Box, script: str) -> subprocess.CompletedProcess:
    argv = box.command(["/usr/bin/sh", "-c", script])
    return await asyncio.to_thread(
        subprocess.run, argv, capture_output=True, text=True, timeout=60, check=False
    )


def _backend(wt: Path) -> ReapiBackend:

    return ReapiBackend(
        project=PROJECT,
        sisoenv=sisoenv_path(wt),
        mint=lambda: asyncio.sleep(0, result="fake-token"),
    )


def test_the_sisoenv_instance_is_fully_qualified(tmp_path):

    wt = _checkout(tmp_path)
    rt = tmp_path / "rt"
    rt.mkdir()
    b = _backend(wt)
    b.prepare(rt)
    body = b.sisoenv_path(rt).read_text()
    assert f"SISO_REAPI_INSTANCE=projects/{PROJECT}/instances/default_instance" in body
    assert f"SISO_PROJECT={PROJECT}" in body


def test_the_hosts_override_keeps_the_rest_of_resolution(tmp_path):

    wt = _checkout(tmp_path)
    rt = tmp_path / "rt"
    rt.mkdir()
    b = _backend(wt)
    b.prepare(rt)
    lines = b.hosts_path(rt).read_text().splitlines()
    assert lines[0] == f"127.0.0.1 {UPSTREAM_HOST}"
    assert len(lines) > 1


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
async def test_the_box_actually_sees_both_overridden_files(tmp_path):
    wt = _checkout(tmp_path)
    box = _box(wt, tmp_path)
    sisoenv = sisoenv_path(wt)
    async with box:
        r = await _in_box(box, f"head -1 /etc/hosts; echo ---; cat {sisoenv}")
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    hosts_line, sisoenv_body = r.stdout.split("---")
    assert hosts_line.strip() == f"127.0.0.1 {UPSTREAM_HOST}"

    assert "not-the-override" not in sisoenv_body
    assert f"SISO_REAPI_INSTANCE=projects/{PROJECT}/instances/default_instance" in (
        sisoenv_body
    )


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
async def test_the_box_holds_no_luci_credential(tmp_path):
    wt = _checkout(tmp_path)
    box = _box(wt, tmp_path)
    async with box:
        r = await _in_box(box, "env | grep -icE 'credential|token' || true")

        creds = await _in_box(box, "cat ~/.config/chrome_infra/auth/creds.json")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "0", f"credential env in the box: {r.stdout}"
    assert creds.returncode != 0, f"the box can read a luci token: {creds.stdout!r}"


def test_every_checkout_under_the_root_gets_the_override(tmp_path):

    root = tmp_path / "root"
    dsts = []
    for name in ("main", "wt-a", "wt-b"):
        d = root / name / "build" / "config" / "siso"
        d.mkdir(parents=True)
        (d / ".sisoenv").write_text("SISO_PROJECT=not-the-override\n")
        dsts.append(d / ".sisoenv")

    rt = tmp_path / "rt"
    rt.mkdir()
    backend = ReapiBackend(
        project=PROJECT,
        sisoenv=dsts,
        mint=lambda: asyncio.sleep(0, result="fake-token"),
    )
    backend.prepare(rt)
    overrides = [b for b in backend.box_binds(rt) if b.dst != Path("/etc/hosts")]

    assert [b.dst for b in overrides] == dsts

    assert {b.src for b in overrides} == {backend.sisoenv_path(rt)}


def test_the_profile_finds_the_checkouts_and_dedupes_the_shared_file(tmp_path):

    root = tmp_path / "root"
    main = root / "main" / "build" / "config" / "siso"
    main.mkdir(parents=True)
    (main / ".sisoenv").write_text("SISO_PROJECT=upstream\n")
    other = root / "other" / "build" / "config" / "siso"
    other.mkdir(parents=True)
    (other / ".sisoenv").write_text("SISO_PROJECT=upstream\n")
    (root / "wt").mkdir()
    (root / "wt" / "build").symlink_to(root / "main" / "build")

    assert sisoenv_paths(root) == sorted({main / ".sisoenv", other / ".sisoenv"})
    (backend,) = v8_rbe(root).backends
    assert backend.name == "rbe"

    assert not v8_rbe(tmp_path / "empty")


def test_a_single_sisoenv_may_arrive_as_a_string(tmp_path):

    dst = tmp_path / "build" / "config" / "siso" / ".sisoenv"
    dst.parent.mkdir(parents=True)
    dst.touch()
    rt = tmp_path / "rt"
    rt.mkdir()
    backend = ReapiBackend(
        project=PROJECT,
        sisoenv=str(dst),
        mint=lambda: asyncio.sleep(0, result="fake-token"),
    )
    backend.prepare(rt)
    overrides = [b for b in backend.box_binds(rt) if b.dst != Path("/etc/hosts")]

    assert [b.dst for b in overrides] == [dst]


def test_the_shared_net_profile_mounts_the_credential_and_says_so(
    monkeypatch, tmp_path
):

    store = tmp_path / "chrome_infra"
    (store / "auth").mkdir(parents=True)
    monkeypatch.setattr("aisan.egress.reapi.LUCI_STORE", store)

    profile = v8_rbe_shared_net(tmp_path)

    assert profile.backends == ()
    assert [(b.path, b.mode) for b in profile.binds] == [(store, RO)]
    assert "luci credential" in profile.notice


def test_the_shared_net_profile_needs_a_store_to_mount(monkeypatch, tmp_path):
    monkeypatch.setattr("aisan.egress.reapi.LUCI_STORE", tmp_path / "absent")

    assert not v8_rbe_shared_net(tmp_path)


async def test_concurrent_first_requests_mint_the_token_only_once():
    import asyncio

    from aisan.egress.reapi import RefreshingToken

    calls = {"n": 0}

    async def mint() -> str:
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return "tok"

    tok = RefreshingToken(mint)
    results = await asyncio.gather(*[tok() for _ in range(20)])
    assert results == ["tok"] * 20
    assert calls["n"] == 1
