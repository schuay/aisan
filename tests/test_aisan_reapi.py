# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The RBE backend's two config files, and what a real box makes of them.

`tests/test_proxy_rbe.py` covers the proxy mechanism (allowlist,
credential injection, refusal shape). This covers the backend: the two files it
writes, the exact bytes in them, and -- through a real bwrap -- that they land
where siso reads them.

Both files exist because siso is configured in a way neither knob makes obvious,
and both got their content wrong once in a way that surfaced as something else
entirely (a 404 with an HTML body; a CONSUMER_INVALID that reads like an ACL
problem). So the content assertions are exact, and the placement assertion runs
a process in the box rather than reading argv -- bwrap mounts in order and a
later mount wins, so a correct-looking argv with the wrong ordering yields a box
reading the ORIGINAL files, and that failure is invisible from outside.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from aisan import Box
from aisan.egress.reapi import ReapiBackend
from aisan.presets.v8_job import sisoenv_path, sisoenv_paths, v8_job, v8_rbe
from aisan.proxy.rbe import UPSTREAM_HOST

PROJECT = "rbe-chromium-untrusted"


def _checkout(tmp_path: Path) -> Path:
    """A worktree carrying the .sisoenv the override has to shadow."""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "config").touch()
    wt = tmp_path / "wt"
    (wt / "build" / "config" / "siso").mkdir(parents=True)
    # A DIFFERENT project, so a passing test proves the override WON rather than
    # that both files happened to agree.
    (wt / "build" / "config" / "siso" / ".sisoenv").write_text(
        "SISO_PROJECT=not-the-override\n"
    )
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    return wt


def _box(wt: Path, tmp_path: Path) -> Box:
    """The real job profile, with the cgroup off.

    A transient systemd scope is not available in every test env, and what these
    two tests are about is the mount policy. Through `replace` rather than a
    preset argument because the preset takes the individual caps, not the Limits
    object -- and turning the scope off is not a cap.
    """
    spec = v8_job(wt, egress=(_backend(wt),), unshare_net=True, depot_tools=None)
    spec = dataclasses.replace(
        spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
    )
    return Box(spec, box_id=str(tmp_path / "job"))


async def _in_box(box: Box, script: str) -> subprocess.CompletedProcess:
    """Run `script` in the box, through the argv `command()` composes.

    Through a real Box bracket, so the relays are up and the bind-overs are on
    disk: these tests are about what the assembled box contains, and a
    hand-built argv would be a test of this file's own idea of one.
    """
    argv = box.command(["/usr/bin/sh", "-c", script])
    return await asyncio.to_thread(
        subprocess.run, argv, capture_output=True, text=True, timeout=60, check=False
    )


def _backend(wt: Path) -> ReapiBackend:
    # mint is stubbed: what is under test here is the files and the mounts, and
    # a real luci-auth call would make every one of these tests a credential
    # test that skips on any host without one.
    return ReapiBackend(
        project=PROJECT,
        sisoenv=sisoenv_path(wt),
        mint=lambda: asyncio.sleep(0, result="fake-token"),
    )


def test_the_sisoenv_instance_is_fully_qualified(tmp_path):
    # A bare instance name makes Google unable to attribute the request to a
    # project: it answers CONSUMER_INVALID / "permission denied on resource
    # project default_instance", which reads like an ACL problem rather than a
    # naming one. Measured, and the reason this file is written at all.
    wt = _checkout(tmp_path)
    rt = tmp_path / "rt"
    rt.mkdir()
    b = _backend(wt)
    b.prepare(rt)
    body = b.sisoenv_path(rt).read_text()
    assert f"SISO_REAPI_INSTANCE=projects/{PROJECT}/instances/default_instance" in body
    assert f"SISO_PROJECT={PROJECT}" in body


def test_the_hosts_override_keeps_the_rest_of_resolution(tmp_path):
    # Prepended, not replacing: the box still resolves everything else it
    # legitimately looks up. A bare one-line file would break DNS in the box,
    # which presents as every unrelated lookup failing for no visible reason.
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
    """The binds land where siso reads them -- asserted from INSIDE the box.

    argv inspection cannot answer this: bwrap mounts in order and a later mount
    wins, so a correct-looking argv with the wrong ordering yields a box reading
    the ORIGINAL files. That failure is invisible except from inside.
    """
    wt = _checkout(tmp_path)
    box = _box(wt, tmp_path)
    sisoenv = sisoenv_path(wt)
    async with box:
        r = await _in_box(box, f"head -1 /etc/hosts; echo ---; cat {sisoenv}")
    assert r.returncode == 0, f"OUT={r.stdout!r} ERR={r.stderr!r}"
    hosts_line, sisoenv_body = r.stdout.split("---")
    assert hosts_line.strip() == f"127.0.0.1 {UPSTREAM_HOST}"
    # The override won: the checkout's own value is gone.
    assert "not-the-override" not in sisoenv_body
    assert f"SISO_REAPI_INSTANCE=projects/{PROJECT}/instances/default_instance" in (
        sisoenv_body
    )


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
async def test_the_box_holds_no_luci_credential(tmp_path):
    """The backend's whole claim, checked in the ASSEMBLED box.

    Not in a dict of intended env: the box's environment is the spec's plus each
    backend's client vars plus whatever bwrap's own flags add, and the union is
    the only thing a job actually gets. The token is minted host-side per
    request and attached to a frame the box never sees -- so the box holding
    nothing is not a precaution, it is the design.
    """
    wt = _checkout(tmp_path)
    box = _box(wt, tmp_path)
    async with box:
        r = await _in_box(box, "env | grep -icE 'credential|token' || true")
        # And no token file is reachable either. The luci refresh token lives at
        # ~/.config/chrome_infra and is gerrit + cloud scoped and non-expiring,
        # so it is the one file whose absence is worth naming: the legacy path
        # bound it, which is what the host-side mint replaced.
        creds = await _in_box(box, "cat ~/.config/chrome_infra/auth/creds.json")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "0", f"credential env in the box: {r.stdout}"
    assert creds.returncode != 0, f"the box can read a luci token: {creds.stdout!r}"


def test_every_checkout_under_the_root_gets_the_override(tmp_path):
    # A box rooted above several checkouts is the interactive shape: a gclient
    # root holds the main tree and its worktrees, and worktrees on different DEPS
    # hashes resolve to DIFFERENT shared .sisoenv files. One bind-over would
    # leave the rest reading their own bare instance name, which Google rejects
    # as CONSUMER_INVALID -- a mid-build failure in exactly the trees the
    # operator did not test.
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
    # One source for all of them: the instance is a property of the box, not of
    # which checkout the agent happens to build in.
    assert {b.src for b in overrides} == {backend.sisoenv_path(rt)}


def test_the_profile_finds_the_checkouts_and_dedupes_the_shared_file(tmp_path):
    # Worktrees share the file by symlinking into a checkout: three trees, two
    # distinct .sisoenv. Deduplication is not cosmetic -- two bind-overs of one
    # destination is an ambiguous mount order.
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
    (backend,) = v8_rbe(root)
    assert backend.name == "rbe"
    # Nothing to override is a box without the fast path, not a refusal.
    assert v8_rbe(tmp_path / "empty") == ()
