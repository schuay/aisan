# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import os

import pytest

from aisan import mint


async def test_rbe_token_gives_luci_auth_no_inherited_stdin(monkeypatch):
    """luci-auth reauth is interactive, and an inherited stdin is the box's own
    TTY -- it would block the request forever. The mint must hand it DEVNULL so
    a lapsed login fails fast instead of prompting into the sandbox."""
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


async def test_rbe_token_survives_non_utf8_output(monkeypatch):
    """The output is decoded best-effort: a luci-auth that emits a non-UTF-8
    byte must not turn a mint into an opaque UnicodeDecodeError on the request
    path."""

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
    """A luci-auth that never returns (the interactive-reauth case, or any other
    hang) is bounded: the mint raises rather than pinning the request forever."""
    # `exec` so the killed process IS the sleep, as a single-binary luci-auth
    # would be: a forked grandchild would keep the stdout pipe open and gate
    # asyncio's own wait(), which is a fake-harness artifact, not the real tool.
    fake = tmp_path / "luci-auth"
    fake.write_text("#!/bin/sh\nexec sleep 3\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(mint, "_TOKEN_TIMEOUT_S", 0.3)

    with pytest.raises(RuntimeError, match="timed out"):
        await mint.rbe_token()
