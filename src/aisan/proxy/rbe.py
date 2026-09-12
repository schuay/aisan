# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Proxy Remote Build Execution without giving the box a LUCI token.

The network-isolated box speaks plaintext HTTP/2 to this proxy. The proxy checks
gRPC authority and method, adds a cloud bearer, and forwards opaque protobuf
bodies over TLS. A prototype linked V8's ``d8`` in 170 seconds through 1,268
remote executions without putting a credential in the build process.

Siso requires ``-reapi_insecure`` for the local connection and then refuses to
attach OAuth credentials. It must dial the real service name through an in-box
hosts entry because Google routes on ``:authority``. The proxy verifies that
authority as well as the method. The fully qualified instance must be passed via
``-reapi_instance``; tests found that ``SISO_REAPI_INSTANCE`` was ignored.

The method allowlist includes reads, execution, and CAS writes required by real
builds. It excludes action-cache updates, which could poison results shared with
other users. Remaining risks include quota abuse and using CAS as a covert
channel within the authority already granted to the build.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import h2.config
import h2.connection
import h2.events
import h2.exceptions

from .http import LogGate
from .policy import permits as policy_permits

log = logging.getLogger(__name__)

UPSTREAM_HOST = "remotebuildexecution.googleapis.com"
UPSTREAM_PORT = 443

# Methods observed in a complete proxied V8 build. Include ByteStream/Write for
# blobs above the batch size even though that build didn't use it.
ALLOWED_METHODS = frozenset(
    {
        "/build.bazel.remote.execution.v2.Capabilities/GetCapabilities",
        "/build.bazel.remote.execution.v2.ActionCache/GetActionResult",
        "/build.bazel.remote.execution.v2.ContentAddressableStorage/FindMissingBlobs",
        "/build.bazel.remote.execution.v2.ContentAddressableStorage/BatchReadBlobs",
        "/build.bazel.remote.execution.v2.ContentAddressableStorage/BatchUpdateBlobs",
        "/build.bazel.remote.execution.v2.ContentAddressableStorage/GetTree",
        "/build.bazel.remote.execution.v2.Execution/Execute",
        "/build.bazel.remote.execution.v2.Execution/WaitExecution",
        "/google.longrunning.Operations/GetOperation",
        "/google.longrunning.Operations/WaitOperation",
        "/google.bytestream.ByteStream/Read",
        "/google.bytestream.ByteStream/Write",
    }
)

# ActionCache/UpdateActionResult stays blocked because it can poison shared
# results. Siso doesn't need it; remote execution updates the cache server-side.

# Forward only measured request headers. Box-controlled ``x-goog-*`` fields could
# change routing, billing, or quota under the injected cloud bearer.
#
# A complete build with grpc-go 1.83.1 sent content-type, te, user-agent,
# grpc-timeout, grpc-accept-encoding, and REAPI RequestMetadata. Its user agent
# identifies grpc-go without a host fingerprint.
#
# The gRPC specification reserves the ``grpc-`` prefix for transport metadata.
# Passing the family preserves compression and tracing fields. RequestMetadata
# carries tool and invocation IDs for build-event correlation.
_FORWARDED_HEADERS = frozenset(
    {
        b"content-type",
        b"te",
        b"user-agent",
        b"build.bazel.remote.execution.v2.requestmetadata-bin",
    }
)


def _forwarded_header(name: bytes) -> bool:
    """Return whether a lowercase request header may reach Google.

    The caller handles pseudo-headers, rewrites authority and host, and replaces
    authorization with the host bearer.
    """
    return name in _FORWARDED_HEADERS or name.startswith(b"grpc-")


# Return application errors as gRPC trailers so clients don't retry them as
# transport failures.
_GRPC_PERMISSION_DENIED = "7"

# Distinguish a missing host credential from a policy denial. gRPC retries
# neither status.
_GRPC_UNAUTHENTICATED = "16"

# Limit repeated logs during a credential outage.
_MINT_FAILURE_LOG_S = 60.0

# Resolve the bearer for every request because builds outlive 30-minute tokens.
TokenSource = Callable[[], Awaitable[str]]


@dataclass
class _Stream:
    """Map a client stream to its upstream stream."""

    upstream_id: int


class _Pending:
    """Hold data until the destination's flow-control window opens.

    Keep each chunk's source-side flow-control debt. Acknowledge the source only
    after forwarding the bytes, so a slow destination applies backpressure
    instead of growing this queue without limit.
    """

    def __init__(self) -> None:
        self.chunks: deque[tuple[bytes, int]] = deque()
        self.offset = 0  # bytes of chunks[0] already sent
        self.end = False  # end_stream seen; owed once the queue drains


class Session:
    """Map one inbound HTTP/2 connection to one upstream TLS connection.

    Siso keeps a small number of connections open, so pooling adds no measured
    benefit.
    """

    def __init__(self, token: TokenSource, allow: frozenset[str]) -> None:
        self._token = token
        self._allow = allow
        self._c2u: dict[int, int] = {}
        self._u2c: dict[int, int] = {}
        # Key both directional queues by client stream ID.
        self._to_up: dict[int, _Pending] = {}
        self._to_down: dict[int, _Pending] = {}
        self._lock = asyncio.Lock()
        # Track mint failures per session to avoid global test state.
        self._mint_failed_at = float("-inf")
        # Bound warnings from loops on denied methods.
        self._warn = LogGate()

    async def run(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            up_r, up_w, up_conn = await self._connect_upstream()
        except Exception as e:
            log.warning("rbe proxy: upstream connect failed: %s", e)
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            return
        down = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        down.initiate_connection()
        writer.write(down.data_to_send())
        await writer.drain()
        # End the session when either direction closes. Waiting for both could
        # leave the other pump blocked forever on a read.
        pumps = [
            asyncio.create_task(self._pump_down(reader, writer, down, up_w, up_conn)),
            asyncio.create_task(self._pump_up(up_r, writer, down, up_w, up_conn)),
        ]
        try:
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in pumps:
                t.cancel()
            for t in pumps:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            for w in (up_w, writer):
                w.close()
                with contextlib.suppress(OSError, asyncio.CancelledError):
                    await w.wait_closed()

    async def _connect_upstream(self):
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(["h2"])
        reader, writer = await asyncio.open_connection(
            UPSTREAM_HOST, UPSTREAM_PORT, ssl=ctx, server_hostname=UPSTREAM_HOST
        )
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        writer.write(conn.data_to_send())
        await writer.drain()
        return reader, writer, conn

    async def _flush(self, writer, conn) -> None:
        if data := conn.data_to_send():
            writer.write(data)
            await writer.drain()

    def _queue(
        self, pending: dict[int, _Pending], key: int, data: bytes, flow_len: int
    ) -> None:
        """Queue stream data until the destination can accept it."""
        pending.setdefault(key, _Pending()).chunks.append((data, flow_len))

    def _drain(self, pending, key: int, conn, sid: int, src_conn, src_sid: int):
        """Send queued data within the peer's flow-control limits.

        h2 doesn't buffer or split oversized DATA frames. A 256 KiB CAS blob
        exceeded the default 64 KiB window and previously ended the session with
        ``FlowControlError``. Limit writes by both the current window and
        ``max_outbound_frame_size``.

        Acknowledge source bytes only after forwarding them so a stalled peer
        stops the sender instead of increasing memory use.
        """
        p = pending.get(key)
        if p is None:
            return
        try:
            while p.chunks:
                data, flow_len = p.chunks[0]
                remaining = len(data) - p.offset
                if remaining <= 0:
                    # Retire zero-length DATA frames immediately. They consume no
                    # window but can otherwise keep end_stream queued forever.
                    p.chunks.popleft()
                    p.offset = 0
                    if src_conn is not None and flow_len:
                        with contextlib.suppress(Exception):
                            src_conn.acknowledge_received_data(flow_len, src_sid)
                    continue
                window = conn.local_flow_control_window(sid)
                if window <= 0:
                    return  # peer is full; a WindowUpdated event resumes us
                take = min(window, conn.max_outbound_frame_size, remaining)
                conn.send_data(sid, data[p.offset : p.offset + take], end_stream=False)
                p.offset += take
                if p.offset >= len(data):
                    p.chunks.popleft()
                    p.offset = 0
                    # Acknowledge the source after forwarding the complete chunk.
                    if src_conn is not None and flow_len:
                        with contextlib.suppress(Exception):
                            src_conn.acknowledge_received_data(flow_len, src_sid)
            if p.end and not p.chunks:
                with contextlib.suppress(Exception):
                    conn.end_stream(sid)
                pending.pop(key, None)
        except (h2.exceptions.StreamClosedError, h2.exceptions.NoSuchStreamError):
            # Release source flow-control credit for an abandoned queue.
            self._discard(pending, key, src_conn, src_sid)

    def _drain_all(self, pending, conn, to_upstream: bool, src_conn) -> None:
        """Retry stalled streams after a flow-control window update.

        A connection-level update uses stream ID zero and may unblock any stream,
        so retry each small queue.
        """
        for key in list(pending):
            sid = self._c2u.get(key) if to_upstream else key
            src_sid = key if to_upstream else self._c2u.get(key)
            if sid is None or src_sid is None:
                continue
            self._drain(pending, key, conn, sid, src_conn, src_sid)

    def _discard(self, pending, key: int, src_conn, src_sid) -> None:
        """Drop a queue and repay its source-side flow-control debt.

        Without the acknowledgements, reset streams permanently consume the
        connection window. About 64 KiB of accumulated debt can stall the peer.
        h2 restores the connection window even when the stream has closed.
        """
        p = pending.pop(key, None)
        if p is None or src_conn is None or src_sid is None:
            return
        for _data, flow_len in p.chunks:
            if flow_len:
                with contextlib.suppress(Exception):
                    src_conn.acknowledge_received_data(flow_len, src_sid)

    def _refuse(
        self,
        down,
        stream_id: int,
        message: str,
        status: str = _GRPC_PERMISSION_DENIED,
    ) -> None:
        """Return a gRPC status trailer instead of an HTTP error.

        Siso treats an HTTP error as a retryable transport failure. A gRPC status
        reports the policy or authentication failure and stops retries. The
        message is written for the build log rather than naming internal code.
        """
        down.send_headers(
            stream_id,
            [
                (b":status", b"200"),
                (b"content-type", b"application/grpc"),
                (b"grpc-status", status.encode()),
                (b"grpc-message", message.encode()),
            ],
            end_stream=True,
        )

    def _note_mint_failure(self, exc: Exception) -> None:
        """Log a sustained credential outage at a bounded rate.

        One build may issue thousands of requests, all with the same failure.
        """
        # Measure from the last logged failure. Updating on every request would
        # suppress a sustained outage forever.
        now = time.monotonic()
        if now - self._mint_failed_at > _MINT_FAILURE_LOG_S:
            log.warning(
                "rbe proxy: no credential; refusing REAPI (builds go local): %s", exc
            )
            self._mint_failed_at = now

    async def _pump_down(self, reader, writer, down, up_w, up_conn) -> None:
        """Forward box traffic after enforcing policy and adding credentials.

        Process request events before flushing HTTP/2 housekeeping. Flushing the
        downstream SETTINGS acknowledgement first could fail after parsing a
        complete request but before forwarding it. Traces reproduced this on
        about five percent of connections under load.
        """
        try:
            while data := await reader.read(65536):
                async with self._lock:
                    for ev in down.receive_data(data):
                        await self._on_down_event(ev, down, up_conn)
                    await self._flush(up_w, up_conn)
                    await self._flush(writer, down)
        except (ConnectionResetError, BrokenPipeError) as e:
            # Session closure is expected, but keep a debug record for diagnosing
            # requests that never reach the upstream.
            log.debug("rbe proxy: downstream closed: %s", type(e).__name__)
        # Keep one bad connection from stopping the proxy.
        except Exception as e:
            log.warning("rbe proxy: downstream: %s: %s", type(e).__name__, e)

    async def _on_down_event(self, ev, down, up_conn) -> None:
        if isinstance(ev, h2.events.RequestReceived):
            headers = dict(ev.headers)
            # Replace invalid UTF-8 so a legal binary header value becomes a clean
            # policy refusal instead of terminating the connection.
            path = headers.get(b":path", b"").decode(errors="replace")
            # Convert exceptions from custom allowlist implementations to denials.
            if not policy_permits(lambda: path in self._allow, subject=path):
                self._warn.warning(log, "rbe proxy: refused %s", path)
                self._refuse(down, ev.stream_id, f"method not permitted: {path}")
                return
            # Google routes generic ByteStream and Operations methods by
            # ``:authority``. Verify it before attaching the cloud bearer.
            #
            # Refuse a different host instead of hiding it through rewriting.
            # Permit the relay's numeric loopback port in the authority value.
            #
            # ``isdigit`` accepts non-ASCII numerals, so check both properties.
            authority = headers.get(b":authority", b"").decode(errors="replace")
            host, sep, port = authority.partition(":")
            if host != UPSTREAM_HOST or (
                sep and not (port.isascii() and port.isdigit())
            ):
                self._warn.warning(log, "rbe proxy: refused authority %r", authority)
                self._refuse(
                    down, ev.stream_id, f"authority not permitted: {authority}"
                )
                return
            # Return missing credentials as a per-stream refusal. A disconnected
            # session made Siso retry ten times for about 40 seconds before local
            # fallback. The next request can use a newly refreshed login.
            try:
                bearer = await self._token()
            # All mint failures have the same client-facing meaning.
            except Exception as e:
                self._note_mint_failure(e)
                self._refuse(
                    down,
                    ev.stream_id,
                    f"no RBE credential available: {e}",
                    status=_GRPC_UNAUTHENTICATED,
                )
                return
            # Normalize the accepted authority to the upstream host. The
            # loopback relay port has no meaning on the HTTPS connection to 443.
            # Replace pseudo-headers in place because HTTP/2 requires them before
            # regular fields.
            # Rewrite ``host`` too because h2 rejects disagreement with
            # ``:authority``.
            out: list[tuple[bytes, bytes]] = []
            dropped: list[str] = []
            for k, v in ev.headers:
                lk = k.lower()
                if lk in (b":authority", b"host"):
                    out.append((k, UPSTREAM_HOST.encode()))
                elif lk == b"authorization":
                    continue  # replaced with the injected bearer below
                elif lk.startswith(b":") or _forwarded_header(lk):
                    out.append((k, v))
                else:
                    dropped.append(lk.decode("latin1"))
            out.append((b"authorization", f"Bearer {bearer}".encode()))
            if dropped:
                # Record dropped headers at debug level for client upgrades.
                log.debug("rbe proxy: dropped box header(s): %s", ", ".join(dropped))
            uid = up_conn.get_next_available_stream_id()
            self._c2u[ev.stream_id] = uid
            self._u2c[uid] = ev.stream_id
            up_conn.send_headers(uid, out, end_stream=False)
        elif isinstance(ev, h2.events.DataReceived):
            if (uid := self._c2u.get(ev.stream_id)) is not None:
                # Queue against the upstream window and acknowledge after sending.
                self._queue(
                    self._to_up, ev.stream_id, ev.data, ev.flow_controlled_length
                )
                self._drain(self._to_up, ev.stream_id, up_conn, uid, down, ev.stream_id)
            else:
                # Release credit immediately for data on a refused stream.
                down.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
        elif isinstance(ev, h2.events.WindowUpdated):
            # A box window update unblocks response data queued for the box.
            self._drain_all(self._to_down, down, False, up_conn)
        elif isinstance(ev, h2.events.StreamEnded):
            if (uid := self._c2u.get(ev.stream_id)) is not None:
                # End the stream only after queued request data has been sent.
                p = self._to_up.setdefault(ev.stream_id, _Pending())
                p.end = True
                self._drain(self._to_up, ev.stream_id, up_conn, uid, down, ev.stream_id)
        elif isinstance(ev, h2.events.StreamReset):
            if (uid := self._c2u.pop(ev.stream_id, None)) is not None:
                self._u2c.pop(uid, None)
                # Release flow-control credit on both sides before dropping queues.
                self._discard(self._to_up, ev.stream_id, down, ev.stream_id)
                self._discard(self._to_down, ev.stream_id, up_conn, uid)
                with contextlib.suppress(Exception):
                    up_conn.reset_stream(uid)

    async def _pump_up(self, up_r, writer, down, up_w, up_conn) -> None:
        """Forward Google responses to the box frame by frame.

        Process response events before flushing upstream housekeeping, mirroring
        the request-side ordering. Otherwise a failed upstream write can discard
        a response already received from Google.
        """
        try:
            while data := await up_r.read(65536):
                async with self._lock:
                    for ev in up_conn.receive_data(data):
                        self._on_up_event(ev, down, up_conn)
                    await self._flush(writer, down)
                    await self._flush(up_w, up_conn)
        except (ConnectionResetError, BrokenPipeError) as e:
            log.debug("rbe proxy: upstream closed: %s", type(e).__name__)
        # Keep one bad connection from stopping the proxy.
        except Exception as e:
            log.warning("rbe proxy: upstream: %s: %s", type(e).__name__, e)

    def _on_up_event(self, ev, down, up_conn) -> None:
        cid = self._u2c.get(getattr(ev, "stream_id", -1))
        if isinstance(ev, h2.events.ResponseReceived):
            if cid is not None:
                with contextlib.suppress(
                    h2.exceptions.StreamClosedError, h2.exceptions.NoSuchStreamError
                ):
                    down.send_headers(cid, ev.headers, end_stream=False)
        elif isinstance(ev, h2.events.TrailersReceived):
            if cid is not None:
                with contextlib.suppress(
                    h2.exceptions.StreamClosedError, h2.exceptions.NoSuchStreamError
                ):
                    down.send_headers(cid, ev.headers, end_stream=True)
        elif isinstance(ev, h2.events.DataReceived):
            if cid is not None:
                # CAS blobs routinely exceed the box's 64 KiB window.
                self._queue(self._to_down, cid, ev.data, ev.flow_controlled_length)
                self._drain(self._to_down, cid, down, cid, up_conn, ev.stream_id)
            else:
                up_conn.acknowledge_received_data(
                    ev.flow_controlled_length, ev.stream_id
                )
        elif isinstance(ev, h2.events.WindowUpdated):
            # An upstream window update unblocks queued request data.
            self._drain_all(self._to_up, up_conn, True, down)
        elif isinstance(ev, h2.events.StreamEnded):
            if cid is not None:
                p = self._to_down.setdefault(cid, _Pending())
                p.end = True
                self._drain(self._to_down, cid, down, cid, up_conn, ev.stream_id)
        elif isinstance(ev, h2.events.StreamReset):
            if cid is not None:
                self._c2u.pop(cid, None)
                self._u2c.pop(ev.stream_id, None)
                self._discard(self._to_up, cid, down, cid)
                self._discard(self._to_down, cid, up_conn, ev.stream_id)
                with contextlib.suppress(Exception):
                    down.reset_stream(cid)


def _handler(token: TokenSource, allow: frozenset[str]):
    """Create one REAPI session for each accepted connection."""

    async def handle(reader, writer) -> None:
        await Session(token, allow).run(reader, writer)

    return handle


async def serve_unix(
    socket_path: Path,
    token: TokenSource,
    *,
    allow: frozenset[str] = ALLOWED_METHODS,
) -> asyncio.Server:
    """Serve authenticated REAPI through a private UNIX socket.

    An in-box TCP relay connects Siso to this socket. Policy and credentials stay
    in the host process. Remove stale per-box sockets before binding and restrict
    the new socket to its owner.
    """
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(_handler(token, allow), str(socket_path))
    socket_path.chmod(0o600)
    return server


def hosts_file(tmp_dir: Path, port_host: str = UPSTREAM_HOST) -> Path:
    """Create an in-box hosts file that resolves the RBE service locally.

    Siso then sends Google's service name as ``:authority`` while its connection
    reaches the loopback relay. Dialing 127.0.0.1 directly makes Google return a
    misleading HTML 404.
    """
    path = tmp_dir / "hosts"
    existing = Path("/etc/hosts").read_text() if Path("/etc/hosts").exists() else ""
    path.write_text(f"127.0.0.1 {port_host}\n{existing}")
    return path
