# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""One helper the end-to-end tests share: run a command inside a real box.

aisan renders an argv and stops there -- running it is the consumer's, and the
consumer's runner has retry/capture/timeout-hint behaviour that has nothing to do
with confinement. So the tests bring their own spawn rather than importing one,
which is also what keeps the suite honest about the dependency graph: a test
helper that reached for an app's shell runner would make "aisan installs alone"
true of the package and false of its tests.

Deliberately minimal, and the minimum is set by what the assertions read:
combined output (a "Read-only file system" arrives on stderr, and the claim is
about what the box refused, not which stream said so) and the exit status.
"""

from __future__ import annotations

import asyncio
import os
import signal


async def run_boxed(command: str, *, sandbox=None, timeout: float = 60.0) -> str:
    """`bash -lc command` under `sandbox`, as combined output plus "exit N".

    `sandbox` is anything with a `wrapper()` returning an argv prefix -- a
    Sandbox, or a Box through the adapter the preset tests use. None runs
    unconfined, which is how a test states its baseline.

    Its own process group, killed on timeout: a box whose payload hangs holds
    children the test would otherwise leak into the rest of the session.
    """
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
