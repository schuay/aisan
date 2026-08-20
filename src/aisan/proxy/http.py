# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Shared infrastructure for the aiohttp-based host proxies.

Provider modules own request policy, credentials, upstream behavior, and error
envelopes. This module owns only the mechanics they share: request-size and rate
limits plus the Unix-socket server lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

# A prompt plus its history is large but bounded; an exfiltration is not. 32 MiB
# is far above a normal request and far below a useful bulk-smuggling channel.
MAX_BODY_BYTES = 32 << 20


@dataclass
class RateLimit:
    """Token bucket, so a runaway client burns its allowance, not the quota.

    Deliberately crude: this bounds a loop that has gone wrong, it is not a
    fairness mechanism. `monotonic` avoids handing out tokens after a wall-clock
    step.
    """

    per_minute: int = 120
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)

    def allow(self) -> bool:
        now = time.monotonic()
        if self._last == 0.0:
            self._tokens, self._last = float(self.per_minute), now
        self._tokens = min(
            float(self.per_minute),
            self._tokens + (now - self._last) * self.per_minute / 60.0,
        )
        self._last = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


async def serve(socket_path: Path, app: web.Application) -> web.AppRunner:
    """Serve `app` on `socket_path`; the caller owns the runner's lifetime.

    The socket lives in a box's control directory, which is already mounted into
    the box. A stale socket from a crashed run would make bind fail, so it is
    removed first. Runtime paths are per box, so this cannot unlink a live peer.
    """
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    # The policy modules log decisions explicitly. aiohttp's access log would
    # add one uninformative INFO line per model request and bury refusals.
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.UnixSite(runner, str(socket_path)).start()
    # The box runs as the same uid; keep the socket off-limits to anyone else.
    socket_path.chmod(0o600)
    return runner


async def run_forever(socket_path: Path, app: web.Application) -> None:
    """Serve until cancelled, then clean up the runner and socket."""
    runner = await serve(socket_path, app)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        socket_path.unlink(missing_ok=True)
