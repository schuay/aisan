# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Shared infrastructure for the aiohttp-based host proxies.

Provider modules own request policy, credentials, upstream behavior, and error
envelopes. This module owns only the mechanics they share: request-size and rate
limits, constant-time token primitives, and UNIX/TCP server lifecycle.
"""

from __future__ import annotations

import asyncio
import secrets
import socket
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


async def serve_tcp(app: web.Application) -> tuple[web.AppRunner, int]:
    """Serve `app` on one explicit IPv4 loopback socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    runner = web.AppRunner(app, access_log=None)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(socket.SOMAXCONN)
        sock.setblocking(False)
        assigned = int(sock.getsockname()[1])
        await runner.setup()
        await web.SockSite(runner, sock).start()
    except BaseException:
        await runner.cleanup()
        sock.close()
        raise
    else:
        return runner, assigned


def token_matches(presented: str | None, expected: str | None) -> bool:
    """Whether a client-presented proxy token is valid for this mode."""
    if expected is None:
        return True
    return presented is not None and secrets.compare_digest(presented, expected)


def bearer_token(request: web.Request) -> str | None:
    """Return one exact bearer credential, rejecting duplicates and malformed input."""
    values = request.headers.getall("Authorization", [])
    if len(values) != 1:
        return None
    scheme, separator, value = values[0].partition(" ")
    if not separator or scheme.lower() != "bearer" or not value or " " in value:
        return None
    return value


async def run_forever(socket_path: Path, app: web.Application) -> None:
    """Serve until cancelled, then clean up the runner and socket."""
    runner = await serve(socket_path, app)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        socket_path.unlink(missing_ok=True)
