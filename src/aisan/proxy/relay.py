# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Splice in-box loopback connections to a host UNIX socket.

The sandbox has no network (``--unshare-net``) but retains loopback. A bound UNIX
socket crosses the namespace as a filesystem object. ``rest_asyncio`` creates
its own ``AsyncAuthorizedSession`` and can't accept an aiohttp ``UnixConnector``,
so model clients reach the socket through this small TCP relay.

The relay only copies bytes. The host proxy decides what to allow and which
credential to attach.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Bound concurrent connections to protect host file descriptors. Request rate
# limits in http.py don't cover idle connections. Normal clients use far fewer
# than this ceiling.
_MAX_CONNECTIONS = 256


async def _splice(
    src: asyncio.StreamReader, dst: asyncio.StreamWriter, direction: str
) -> None:
    """Copy until EOF, then half-close the destination.

    Drain each chunk so streaming responses reach the client immediately.
    """
    try:
        while chunk := await src.read(65536):
            dst.write(chunk)
            await dst.drain()
    except (ConnectionResetError, BrokenPipeError) as e:
        # Log the direction because either its read or write can fail. Clean
        # hangups deliver EOF and don't reach this branch.
        log.debug("relay: %s: %s", direction, type(e).__name__)
    finally:
        with contextlib.suppress(OSError):
            dst.write_eof()


async def serve(socket_path: Path, port: int, *, host: str = "127.0.0.1"):
    """Serve loopback `host:port`, forwarding each connection to `socket_path`.

    Return the asyncio server so the caller controls its lifetime. Propagate bind
    failures at startup instead of letting the first model call fail later.
    """

    gate = asyncio.Semaphore(_MAX_CONNECTIONS)

    async def handle(
        client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter
    ) -> None:
        if gate.locked():
            # Refuse connections above the ceiling instead of exhausting the host.
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
                # Retrieve exceptions from both tasks before closing their shared
                # transports, so neither task is orphaned.
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
