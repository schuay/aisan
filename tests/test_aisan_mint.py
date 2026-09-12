# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import os

import pytest

from aisan import mint


async def test_rbe_token_gives_luci_auth_no_inherited_stdin(monkeypatch):
    captured: dict = {}

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"a-token\n", b""

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(mint.asyncio, "create_subprocess_exec", fake_exec)

    assert await mint.rbe_token() == "a-token"
    assert captured.get("stdin") is asyncio.subprocess.DEVNULL


async def test_rbe_token_starts_luci_auth_in_an_empty_directory(monkeypatch, tmp_path):
    captured: dict = {}

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"a-token\n", b""

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        captured["entries"] = sorted(os.listdir(kwargs["cwd"]))
        return _Proc()

    monkeypatch.setattr(mint.asyncio, "create_subprocess_exec", fake_exec)
    launcher_cwd = tmp_path / "the-repo-the-agent-edited"
    launcher_cwd.mkdir()
    (launcher_cwd / ".netrc").write_text("machine example.com\n")
    monkeypatch.chdir(launcher_cwd)
    monkeypatch.setenv("TMPDIR", str(launcher_cwd))
    monkeypatch.setenv("OLDPWD", str(launcher_cwd))
    monkeypatch.setenv("INIT_CWD", str(launcher_cwd))

    assert await mint.rbe_token() == "a-token"
    assert captured["cwd"] != str(launcher_cwd)
    assert captured["entries"] == []
    for name in ("PWD", "TMPDIR", "TMP", "TEMP"):
        assert captured["env"][name] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]
    assert "INIT_CWD" not in captured["env"]


async def test_rbe_token_survives_non_utf8_output(monkeypatch):

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"tok\xff\n", b""

    async def fake_exec(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(mint.asyncio, "create_subprocess_exec", fake_exec)

    result = await mint.rbe_token()
    assert result.startswith("tok")


async def test_rbe_token_times_out_instead_of_blocking(monkeypatch, tmp_path):

    fake = tmp_path / "luci-auth"
    fake.write_text("#!/bin/sh\nexec sleep 3\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(mint, "_TOKEN_TIMEOUT_S", 0.3)

    with pytest.raises(RuntimeError, match="timed out"):
        await mint.rbe_token()
