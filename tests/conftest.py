# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import pytest

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)

_DARWIN_BASETEMP_ROOT: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Keep Darwin's ``tmp_path`` short enough for ``AF_UNIX`` sockets.

    macOS sets ``TMPDIR`` to ``/var/folders/...``, which resolves below
    ``/private/var/folders/...`` and leaves pytest's per-test directories past
    Darwin's 104-byte ``sun_path`` limit before a socket name is appended.
    """
    global _DARWIN_BASETEMP_ROOT  # noqa: PLW0603
    if sys.platform == "darwin" and config.option.basetemp is None:
        _DARWIN_BASETEMP_ROOT = Path(
            tempfile.mkdtemp(prefix="aisan-pt-", dir="/private/tmp")
        )
        config.option.basetemp = str(_DARWIN_BASETEMP_ROOT / "t")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _DARWIN_BASETEMP_ROOT is not None:
        shutil.rmtree(_DARWIN_BASETEMP_ROOT, ignore_errors=True)


async def run_boxed(command: str, *, sandbox=None, timeout: float = 60.0) -> str:
    argv = [*(sandbox.wrapper() if sandbox else []), "bash", "-lc", command]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout):
            out, _ = await proc.communicate()
    except TimeoutError:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()
        raise AssertionError(
            f"boxed command exceeded {timeout:g}s: {command}"
        ) from None
    return f"{out.decode(errors='replace')}\nexit {proc.returncode}"
