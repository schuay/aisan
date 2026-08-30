# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""In-box loopback -> UNIX socket splice. Holds no secret, makes no decisions.

The sandbox has no network (`--unshare-net`) but keeps a working loopback, and a
UNIX socket bound in from the host crosses the namespace because it is a
filesystem object. This is the seam. The model client cannot use it directly:
`rest_asyncio` constructs its own `AsyncAuthorizedSession` internally with no
place to hand in an aiohttp `UnixConnector`, so reaching the socket would mean
patching a library private. These ~30 lines are the honest alternative.

Deliberately dumb: it copies bytes. Every decision -- what is allowed, which
credential is attached -- belongs to the host side, where the box cannot reach
it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# A generous ceiling on concurrent spliced connections. Each one opens a
# host-side UNIX connection to the proxy, so an in-box loop could otherwise
# exhaust host file descriptors -- http.py rate-limits REQUESTS, not the
# connections underneath them. Far above what siso (a handful) or a model
# client (one) opens, so nothing legitimate is refused.
_MAX_CONNECTIONS = 256


async def _splice(
    src: asyncio.StreamReader, dst: asyncio.StreamWriter, direction: str
) -> None:
    """Copy until EOF, then half-close so the peer sees the end of the body.

    Per-chunk with a drain, never read-until-complete: a response that is
    accumulated before forwarding is still byte-correct, so buffering hides in
    correctness tests and surfaces only as a model that appears to think for
    several seconds and then answers at once.
    """
    try:
        while chunk := await src.read(65536):
            dst.write(chunk)
            await dst.drain()
    except (ConnectionResetError, BrokenPipeError) as e:
        # Ordinary: one side tore the connection down while the other was still
        # going. Not a warning, but not nothing either -- both directions are
        # spliced concurrently, so without the label a line cannot say which leg
        # broke, and "host proxy -> box" repeating across jobs means something
        # very different from the box hanging up on its own.
        #
        # Direction, not the peer that died: either the read or the write in the
        # loop above can raise, so a single name would be wrong half the time.
        # The clean hangup does not come through here at all -- a peer that goes
        # away after writing delivers EOF, which ends the loop normally.
        log.debug("relay: %s: %s", direction, type(e).__name__)
    finally:
        with contextlib.suppress(OSError):
            dst.write_eof()


async def serve(socket_path: Path, port: int, *, host: str = "127.0.0.1"):
    """Serve loopback `host:port`, forwarding each connection to `socket_path`.

    Returns the asyncio server; the caller owns its lifetime. Bind failures
    propagate -- a relay that is not listening means the agent's very first model
    call fails with a connection error, and that is far easier to diagnose at
    startup than mid-turn.
    """

    gate = asyncio.Semaphore(_MAX_CONNECTIONS)

    async def handle(
        client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter
    ) -> None:
        if gate.locked():
            # At the ceiling: refuse this one rather than pile on. A box that
            # opens connections without draining them is the only way here.
            client_w.close()
            with contextlib.suppress(OSError):
                await client_w.wait_closed()
            return
        async with gate:
            try:
                up_r, up_w = await asyncio.open_unix_connection(str(socket_path))
            except OSError:
                client_w.close()
                with contextlib.suppress(OSError):
                    await client_w.wait_closed()
                return
            try:
                # return_exceptions: a non-Connection error in one splice must
                # not leave its sibling running on a transport this `finally` is
                # about to close -- an orphaned task whose exception is never
                # retrieved. Both are awaited to completion, then both close.
                await asyncio.gather(
                    _splice(client_r, up_w, "box -> host proxy"),
                    _splice(up_r, client_w, "host proxy -> box"),
                    return_exceptions=True,
                )
            finally:
                for w in (up_w, client_w):
                    w.close()
                    with contextlib.suppress(OSError, asyncio.CancelledError):
                        await w.wait_closed()

    return await asyncio.start_server(handle, host, port)


async def run_command(socket_path: Path, port: int, cmd: list[str]) -> int:
    """Serve the relay for the duration of one subprocess command."""
    server = await serve(socket_path, port)
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()
    finally:
        server.close()
        with contextlib.suppress(Exception):
            await server.wait_closed()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: python -m aisan.proxy.relay <socket_path> <port> -- <cmd...>"""
    import sys

    usage = "usage: python -m aisan.proxy.relay <socket_path> <port> -- <cmd...>"
    args = sys.argv[1:] if argv is None else argv
    # sep >= 2 so args[0] (socket) and args[1] (port) are real, not the `--` or
    # past the end: `relay -- cmd` used to index the separator as the socket and
    # int() the command as the port, tracebacking instead of printing usage.
    sep = args.index("--") if "--" in args else -1
    if sep < 2:
        sys.exit(usage)
    sock, port = Path(args[0]), int(args[1])
    cmd = args[sep + 1 :]
    if not cmd:
        sys.exit(usage)
    code = asyncio.run(run_command(sock, port, cmd))
    sys.exit(code)


if __name__ == "__main__":
    main()
