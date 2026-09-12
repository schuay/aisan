# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import os
import signal


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
