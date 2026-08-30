# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side RBE proxy: remote builds for a box that holds no luci token.

The counterpart to `vertex.py`, and the reason the sandbox can keep remote
builds under `--unshare-net`. Measured working before this existed: a real V8
`d8` linked in 170s with 1268 remote executions through the prototype, while the
build process held no credential.

REAPI is gRPC, so this is an HTTP/2 proxy rather than the HTTP relay Vertex
uses. It parses HPACK but never the payload, which is why it stays small: the
protobuf bodies are forwarded as opaque bytes.

Four facts drive the design, none of them guessable from documentation:

- siso with `-reapi_insecure` speaks PLAINTEXT h2 to a local address, so the box
  side needs no TLS and no certificates.
- In that mode gRPC-Go REFUSES to attach its OAuth token ("transport: cannot
  send secure credentials on an insecure connection"). That is the property we
  want -- the box cannot leak a credential it never holds -- but it means the
  proxy must inject `authorization` itself.
- `:authority` reaches the Google frontend as sent, and it routes on it. The box
  must therefore dial a NAME that resolves to this proxy (an `/etc/hosts` entry
  bound into the sandbox), not `127.0.0.1`, or Google answers 404. Because that
  header is what selects the SERVICE, it is checked against UPSTREAM_HOST
  alongside the `:path` allowlist -- gating the method alone would leave the
  generic methods (ByteStream, longrunning Operations) usable against whatever
  else the frontend fronts.
- The instance must be fully qualified via siso's `-reapi_instance` FLAG. The
  `SISO_REAPI_INSTANCE` env var was measured to be silently ignored, and a bare
  instance name makes Google reject the request as CONSUMER_INVALID.

The allowlist is the security boundary. CAS writes ARE permitted, deliberately:
remote execution cannot work without them (the prototype build issued 1260
BatchUpdateBlobs calls), so this surface is genuinely wider than Vertex's. What
bounds it is that the same authority a legitimate build already exercises on the
shared `rbe-chromium-untrusted` instance -- the residual is quota abuse and CAS
as a covert channel, not credential theft.
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

# The gRPC methods a real build actually calls, measured from a full proxied V8
# build rather than read off the REAPI spec -- an allowlist built from the spec
# would be wider than the need. ByteStream/Write is included although that build
# did not reach it: blobs above the batch-size limit stream instead, so omitting
# it would turn a large-artifact build into a confusing mid-build failure.
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

# Deliberately NOT allowlisted, and worth naming so the omission reads as a
# decision: ActionCache/UpdateActionResult. A build that can write the action
# cache can poison it for every other consumer of the shared instance, and siso
# does not need it (remote execution updates the cache server-side).

# Request headers forwarded upstream. Everything the box sends rides the
# cloud-scoped, Gerrit-capable bearer this proxy injects, so a header the box
# CHOSE that reaches Google under that bearer is a capability the allowlist
# never vetted -- `x-goog-user-project` (billing/quota attribution),
# `x-goog-request-params` (routing), whatever a prompt-injected build adds next.
# Forwarding all-but-`authorization` made "whatever the box sent is data" false.
#
# Measured from a full proxied V8 build (siso over grpc-go 1.83.1,
# GetCapabilities through Execute/ByteStream): siso sends exactly content-type,
# te, user-agent, grpc-timeout, grpc-accept-encoding, and the REAPI
# RequestMetadata below -- no `x-goog-*` among them. user-agent is `grpc-go/1.x`,
# tool identity with no host fingerprint (unlike the model proxies' x-stainless
# set), so it is kept rather than dropped.
#
# The `grpc-` family is passed by PREFIX, not enumerated: the gRPC spec reserves
# that prefix for transport metadata (grpc-timeout, grpc-encoding,
# grpc-accept-encoding, grpc-previous-rpc-attempts, the grpc-*-bin tracing
# pair), none of which carries authorization or reroutes a request -- and
# hard-coding only the two a plain build happens to send would strip a
# compressed build's `grpc-encoding` and leave the upstream unable to decode.
# requestmetadata-bin is box-chosen data the upstream expects (tool and
# invocation ids for build-event correlation), inert like the protobuf body.
_FORWARDED_HEADERS = frozenset(
    {
        b"content-type",
        b"te",
        b"user-agent",
        b"build.bazel.remote.execution.v2.requestmetadata-bin",
    }
)


def _forwarded_header(name: bytes) -> bool:
    """Whether a lowercased box request header may reach the upstream.

    Pseudo-headers (`:method`, `:path`, `:scheme`, `:authority`) are the request
    itself and handled by the caller, which also rewrites `:authority`/`host`;
    `authorization` is dropped and replaced with the injected bearer there too.
    """
    return name in _FORWARDED_HEADERS or name.startswith(b"grpc-")


# gRPC signals application errors in trailers with a 200 status, so a refusal
# must be a trailer too -- an HTTP error status makes the client report a
# transport failure and retry, hiding the reason.
_GRPC_PERMISSION_DENIED = "7"

# UNAUTHENTICATED, for the case where the proxy is healthy but has no credential
# to attach (luci-auth can require interactive reauthentication). A DISTINCT
# code from the allowlist's
# PERMISSION_DENIED because they mean opposite things to an operator reading a
# log: one is the boundary doing its job, the other is a machine that needs a
# human. Neither is retried by gRPC, which is the point -- see _refuse.
_GRPC_UNAUTHENTICATED = "16"

# How often a sustained credential outage is logged. One line per outage rather
# than per request: a build issues thousands, and every one of them refuses.
_MINT_FAILURE_LOG_S = 60.0

# Returns the current bearer. Called per REQUEST, never captured: a build runs
# far longer than the ~30 min a luci token lives, so a value read once would
# strand it mid-build.
TokenSource = Callable[[], Awaitable[str]]


@dataclass
class _Stream:
    """One client stream mapped onto its upstream twin."""

    upstream_id: int


class _Pending:
    """Payload accepted from one side that the other side's window cannot take yet.

    Each chunk keeps the SOURCE's flow-control debt (`flow_controlled_length`)
    alongside its bytes, and that debt is only repaid -- acknowledge_received_data
    -- once the chunk has actually gone out the far side. That is what makes a
    slow peer slow the fast one down instead of silently growing this buffer:
    unacknowledged bytes hold the source's window shut.
    """

    def __init__(self) -> None:
        self.chunks: deque[tuple[bytes, int]] = deque()
        self.offset = 0  # bytes of chunks[0] already sent
        self.end = False  # end_stream seen; owed once the queue drains


class Session:
    """One inbound connection, one upstream TLS connection, streams mapped 1:1.

    Per connection rather than pooled: siso opens a handful and keeps them, so
    pooling would add lifetime bookkeeping for no measured gain.
    """

    def __init__(self, token: TokenSource, allow: frozenset[str]) -> None:
        self._token = token
        self._allow = allow
        self._c2u: dict[int, int] = {}
        self._u2c: dict[int, int] = {}
        # Queued payload per direction, both keyed by CLIENT stream id (the
        # upstream twin is one _c2u lookup away), so one key names a stream
        # whichever way its data happens to be flowing.
        self._to_up: dict[int, _Pending] = {}
        self._to_down: dict[int, _Pending] = {}
        self._lock = asyncio.Lock()
        # When this session last failed to mint, for rate-limiting the log. Per
        # session rather than global: siso keeps a handful of connections, so
        # this is a handful of lines per outage rather than thousands, without
        # module state that a test would have to reset between cases.
        self._mint_failed_at = float("-inf")
        # Same reasoning for refusal warnings, which a box loop on a denied
        # method would otherwise emit once per request, unbounded.
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
        # FIRST_COMPLETED, not gather: the two directions end independently, and
        # a session is over as soon as either does. gather() waits for both, so a
        # box that disconnects mid-build would leave the upstream pump blocked on
        # a read that may never return -- pinning a TLS socket and a task for the
        # life of the daemon. Whichever finishes first, the other is cancelled.
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
        """Accept payload for a stream without sending it yet."""
        pending.setdefault(key, _Pending()).chunks.append((data, flow_len))

    def _drain(self, pending, key: int, conn, sid: int, src_conn, src_sid: int):
        """Send as much of one stream's queue as the peer's window allows.

        The bug this exists for: DATA was forwarded with a bare `send_data`, as
        if h2 buffered. It does not -- it raises FlowControlError when the frame
        exceeds the peer's window or the max frame size, and that exception left
        the pump, ending the session. Every byte of the response was dropped and
        the connection died. A 256KiB CAS blob against the default 64KiB window
        reproduced it exactly, which is what the failure log showed: a build streaming
        blobs bigger than one window.

        Two separate limits, so both are respected here rather than assumed
        away: the peer's remaining flow-control window, and max_outbound_frame_size
        (16KiB by default -- h2 does NOT split for us).

        The source's flow-control debt is repaid only as bytes leave, which is
        what turns this from a buffer into backpressure: a stalled peer stops
        the sender rather than accumulating in memory here.
        """
        p = pending.get(key)
        if p is None:
            return
        try:
            while p.chunks:
                data, flow_len = p.chunks[0]
                remaining = len(data) - p.offset
                if remaining <= 0:
                    # A zero-length DATA frame, which gRPC does send. It costs no
                    # window, so it must be retired here rather than measured against
                    # one: treating it as "no room" deadlocked the stream -- the
                    # queue never emptied, so end_stream was never forwarded and the
                    # response hung with its last bytes undelivered.
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
                    # Repay the source now that this chunk is fully forwarded.
                    if src_conn is not None and flow_len:
                        with contextlib.suppress(Exception):
                            src_conn.acknowledge_received_data(flow_len, src_sid)
            if p.end and not p.chunks:
                with contextlib.suppress(Exception):
                    conn.end_stream(sid)
                pending.pop(key, None)
        except (h2.exceptions.StreamClosedError, h2.exceptions.NoSuchStreamError):
            # The destination stream is gone; the queue will never forward, so
            # repay its debt to the source rather than dropping it silently.
            self._discard(pending, key, src_conn, src_sid)

    def _drain_all(self, pending, conn, to_upstream: bool, src_conn) -> None:
        """Retry every stalled stream in one direction after a WindowUpdated.

        A connection-level window update carries stream_id 0 and lifts the whole
        connection, so keying off the event's stream id would leave every other
        stream stalled forever. Cheap to just retry them all: siso keeps a
        handful of streams, and a stream whose window is still shut returns from
        _drain immediately.
        """
        for key in list(pending):
            sid = self._c2u.get(key) if to_upstream else key
            src_sid = key if to_upstream else self._c2u.get(key)
            if sid is None or src_sid is None:
                continue
            self._drain(pending, key, conn, sid, src_conn, src_sid)

    def _discard(self, pending, key: int, src_conn, src_sid) -> None:
        """Drop a stream's queue, repaying the source's flow-control debt.

        A queued chunk holds the SOURCE's window shut until it is forwarded (see
        `_drain`, which repays only as bytes leave). If the stream dies first --
        a reset, a closed twin -- those bytes never leave, so the ack never
        happens and the source's CONNECTION-level window shrinks by that much
        for good. A few resets mid-build silently strand the peer: once ~64KiB
        of unrepaid debt accumulates it can send nothing more, and `refused`
        stays None so the offline fallback never fires. Repaying here makes a
        dropped chunk cost no window.

        `acknowledge_received_data` restores the connection window even for an
        already-closed stream (it processes the connection manager before
        looking the stream up), which is the window that leaks; the suppress is
        for a `src_sid` whose twin is gone.
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
        """Refuse in gRPC's own terms: 200 + a status trailer.

        An HTTP 403 would surface to siso as a transport error and be retried;
        a gRPC status is reported as what it is and stops. That "and stops" is
        what makes this the right answer for a missing credential too, so
        `status` selects which refusal this is (see _GRPC_UNAUTHENTICATED).

        `message` is a MODEL-FACING surface, not an internal note: it reaches the
        box, which means it reaches a build log an agent reads and summarises. So
        it names the policy and what was refused ("method not permitted: <path>"),
        never the hook that decided -- a reason that says "denied by _on_down_event"
        tells the one reader who can do something about it nothing they can act on.
        sandbox-runtime uses the same user-facing refusal rule.
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
        """Log a credential failure once per outage, not once per request.

        A build issues thousands of requests and every one of them refuses, so
        the unconditional line would bury the log in copies of a single fact.
        What an operator needs is one legible "no RBE credential, builds are
        local"; the rest is noise that hides the next real failure.
        """
        # The reset is INSIDE the window check: updating it on every failure
        # (a build issues them every few ms) kept `now - _mint_failed_at` under
        # the window forever, so a sustained outage logged once ever instead of
        # once per window. Measuring from the last LOG restores "once per window".
        now = time.monotonic()
        if now - self._mint_failed_at > _MINT_FAILURE_LOG_S:
            log.warning(
                "rbe proxy: no credential; refusing REAPI (builds go local): %s", exc
            )
            self._mint_failed_at = now

    async def _pump_down(self, reader, writer, down, up_w, up_conn) -> None:
        """Box -> Google. Enforces the allowlist and injects the credential.

        Forwarding comes BEFORE writing anything back to the box, and that order
        is load-bearing. What h2 wants written downstream after a read is
        housekeeping -- the SETTINGS ACK, window updates -- while the events from
        the same read may carry the request itself. Flushing downstream first
        meant a box that had gone away took a fully received, fully parsed
        request with it: the write raised, the loop unwound, and _on_down_event
        never ran, so the request was dropped after the point where it was
        entirely in our hands. Measured at ~5% of connections under load, with a
        trace showing RequestReceived parsed and then discarded. HTTP/2 permits
        the ACK to trail by the microseconds this costs; losing a request it had
        already accepted is not something a proxy may do.
        """
        try:
            while data := await reader.read(65536):
                async with self._lock:
                    for ev in down.receive_data(data):
                        await self._on_down_event(ev, down, up_conn)
                    await self._flush(up_w, up_conn)
                    await self._flush(writer, down)
        except (ConnectionResetError, BrokenPipeError) as e:
            # The ordinary end of a session -- the box exited, or siso closed a
            # pooled connection -- so it is not a warning. But it is not nothing
            # either: silence here is what hid the ordering bug above, since the
            # symptom was a request that simply never arrived upstream.
            log.debug("rbe proxy: downstream closed: %s", type(e).__name__)
        # Broad on purpose: one bad connection must not kill the proxy.
        except Exception as e:
            log.warning("rbe proxy: downstream: %s: %s", type(e).__name__, e)

    async def _on_down_event(self, ev, down, up_conn) -> None:
        if isinstance(ev, h2.events.RequestReceived):
            headers = dict(ev.headers)
            # errors="replace": a non-UTF-8 :path (0x80-0xff is legal in an h2
            # header value) used to raise out of here, through _pump_down's broad
            # except, tearing the connection down with no gRPC status. A replaced
            # path cannot equal an ASCII REAPI method, so it is refused cleanly --
            # the decision is safe on the replaced string.
            path = headers.get(b":path", b"").decode(errors="replace")
            # Through `policy_permits`, so an `allow` that is not the plain
            # frozenset this ships with -- a consumer's own matcher, a set-like
            # object with a __contains__ that raises -- denies rather than
            # unwinding the session, which siso reads as a flaky transport and
            # retries. Same reason the mint failure below is a refusal.
            if not policy_permits(lambda: path in self._allow, subject=path):
                self._warn.warning(log, "rbe proxy: refused %s", path)
                self._refuse(down, ev.stream_id, f"method not permitted: {path}")
                return
            # The allowlist gates :path, but the frontend routes on :authority,
            # and three allowlisted methods (ByteStream/Read, /Write, the
            # longrunning Operations) are GENERIC across Google services rather
            # than REAPI-specific. A permitted :path with someone else's
            # :authority would therefore carry our bearer to a service the
            # allowlist never vetted -- and that bearer is cloud-scoped and
            # Gerrit-capable while it lives (see mint.py).
            #
            # Refused, not silently rewritten: the only legitimate sender is
            # siso dialing the name from the bound /etc/hosts, so a mismatch is
            # either a bug in that bind or the box trying something. Rewriting
            # would hide both; this way it lands in the log.
            # A port is allowed here because siso dials one: the in-box relay
            # offers this host name on a loopback port, and both HTTP/2 and gRPC
            # carry it in :authority, so an exact match against the bare host
            # refused every request.
            #
            # isascii() guards isdigit(), which alone is true for superscripts and
            # for Arabic-Indic and fullwidth digits -- isdecimal() is no better,
            # still accepting the latter two. Harmless either way (the value only
            # gates this decision; the upstream target is the constant below), but
            # a check whose comment says "digits" should mean ASCII digits.
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
            # No credential is a REFUSAL, not a dropped connection. Letting the
            # mint error escape tore the session down with no headers ever sent,
            # and siso reads a silent drop as a flaky transport: it retried ten
            # times (~40s) per build before falling back to a local one. Since
            # luci-auth here needs an interactive reauth -- the steady state, not
            # an incident -- that cost was paid by every build.
            #
            # Answered per stream so the proxy stays up and stays serving: the
            # token is re-minted on demand, so a reauth landing mid-run is picked
            # up by the very next request with no restart and no cached verdict.
            try:
                bearer = await self._token()
            # Broad on purpose: any mint failure reads the same to the client.
            except Exception as e:
                self._note_mint_failure(e)
                self._refuse(
                    down,
                    ev.stream_id,
                    f"no RBE credential available: {e}",
                    status=_GRPC_UNAUTHENTICATED,
                )
                return
            # Our own headers: whatever the box sent is data, not instruction.
            #
            # :authority is normalized to the bare host, because the port the box
            # dialled is the RELAY's loopback port -- an artifact of this hop that
            # means nothing upstream. The connection this rides on is https to
            # :443, whose normal-form authority omits the port (RFC 9110 4.2.2,
            # 4.2.3), and that is exactly what a direct siso would have sent. Not
            # in tension with refusing a foreign host above: the host is the
            # authorization decision and is never rewritten, while the port is
            # transport hygiene on a request already accepted.
            #
            # Replaced IN PLACE, not filtered and re-appended: pseudo-headers must
            # precede regular fields, and appending one after them is a protocol
            # error h2 rejects outright.
            #
            # `host` is rewritten to the same value, not left alone: h2 refuses to
            # send a block whose `host` and `:authority` disagree, and the box may
            # send a `host` header of its choosing. Rewriting only :authority left
            # the box-supplied host in place, so a single crafted request tripped
            # that check; the ProtocolError escaped into `_pump_down`'s broad
            # except and tore the whole connection down with no gRPC status -- the
            # retry-storm-then-local-fallback this module exists to avoid, at will.
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
                # Not a refusal -- the request still goes -- but the breadcrumb
                # for the one case this can break: a future siso sending a header
                # a real build needs that is not yet measured here. Debug, so it
                # is there when a build misbehaves without flooding a healthy one.
                log.debug("rbe proxy: dropped box header(s): %s", ", ".join(dropped))
            uid = up_conn.get_next_available_stream_id()
            self._c2u[ev.stream_id] = uid
            self._u2c[uid] = ev.stream_id
            up_conn.send_headers(uid, out, end_stream=False)
        elif isinstance(ev, h2.events.DataReceived):
            if (uid := self._c2u.get(ev.stream_id)) is not None:
                # Queued, not sent: upstream's window may be smaller than this
                # frame, and the ack is deferred until it actually goes out so
                # a slow upstream throttles the box instead of piling up here.
                self._queue(
                    self._to_up, ev.stream_id, ev.data, ev.flow_controlled_length
                )
                self._drain(self._to_up, ev.stream_id, up_conn, uid, down, ev.stream_id)
            else:
                # No upstream twin (a refused method): nothing will ever forward
                # this, so repay the window now or the box stalls on its own quota.
                down.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
        elif isinstance(ev, h2.events.WindowUpdated):
            # The BOX opened its window, so what unblocks is the data waiting to
            # go TO the box. (Getting this backwards deadlocks: the response
            # fills the initial 64KiB, the box acks, and nothing ever drains.)
            self._drain_all(self._to_down, down, False, up_conn)
        elif isinstance(ev, h2.events.StreamEnded):
            if (uid := self._c2u.get(ev.stream_id)) is not None:
                # end_stream must not overtake queued payload, or the request
                # body is truncated. _drain closes it once the queue empties.
                p = self._to_up.setdefault(ev.stream_id, _Pending())
                p.end = True
                self._drain(self._to_up, ev.stream_id, up_conn, uid, down, ev.stream_id)
        elif isinstance(ev, h2.events.StreamReset):
            if (uid := self._c2u.pop(ev.stream_id, None)) is not None:
                self._u2c.pop(uid, None)
                # Repay both queues before dropping them: the box-bound queue
                # owes the box, the upstream-bound queue owes the upstream.
                self._discard(self._to_up, ev.stream_id, down, ev.stream_id)
                self._discard(self._to_down, ev.stream_id, up_conn, uid)
                with contextlib.suppress(Exception):
                    up_conn.reset_stream(uid)

    async def _pump_up(self, up_r, writer, down, up_w, up_conn) -> None:
        """Google -> box. Forwarded verbatim, per frame.

        Same ordering rule as _pump_down and for the same reason: translate the
        events first, then flush. Flushing the upstream housekeeping ahead of the
        forward put a write to GOOGLE between a received response and its
        delivery to the box, so a dropped upstream connection discarded a
        response that had already arrived -- the mirror of the request-dropping
        bug, and the one that would strand a build waiting on an action result.
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
        # Broad on purpose: see _pump_down.
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
                # The direction the failure log showed: a CAS blob is routinely
                # larger than the box's 64KiB window, so this must queue.
                self._queue(self._to_down, cid, ev.data, ev.flow_controlled_length)
                self._drain(self._to_down, cid, down, cid, up_conn, ev.stream_id)
            else:
                up_conn.acknowledge_received_data(
                    ev.flow_controlled_length, ev.stream_id
                )
        elif isinstance(ev, h2.events.WindowUpdated):
            # Mirror image: upstream opened its window, so the queued REQUEST
            # body is what moves.
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
    """One REAPI session per accepted connection."""

    async def handle(reader, writer) -> None:
        await Session(token, allow).run(reader, writer)

    return handle


async def serve_unix(
    socket_path: Path,
    token: TokenSource,
    *,
    allow: frozenset[str] = ALLOWED_METHODS,
) -> asyncio.Server:
    """Listen on a UNIX socket, forwarding authenticated REAPI to Google.

    The only transport, and the same seam the Vertex proxy uses: a socket is a
    filesystem object, so a bind mount carries it into a box that has no route
    anywhere. siso cannot dial a socket -- it takes a host:port -- so an in-box
    relay offers the port and splices it here. That relay holds no credential
    and makes no decisions; the allowlist and the bearer stay on this side,
    which is the whole point.

    Mode 0600, mirroring the Vertex proxy: the box runs as the same uid, and the
    socket is the only thing standing between any other local user and RBE calls
    on our credential. A stale socket from a killed job would make bind fail
    forever, so it is removed first -- the path is per-job (a digest of the job's
    control dir), so this cannot unlink a live peer's socket.
    """
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(_handler(token, allow), str(socket_path))
    socket_path.chmod(0o600)
    return server


def hosts_file(tmp_dir: Path, port_host: str = UPSTREAM_HOST) -> Path:
    """Write an /etc/hosts to bind into the box so the real name resolves local.

    This is what makes :authority carry the name Google routes on while the
    connection lands on the proxy. Without it siso dials 127.0.0.1 literally,
    the frontend sees that as the authority, and answers 404 with an HTML body
    that surfaces as an unrelated-looking content-type error.
    """
    path = tmp_dir / "hosts"
    existing = Path("/etc/hosts").read_text() if Path("/etc/hosts").exists() else ""
    path.write_text(f"127.0.0.1 {port_host}\n{existing}")
    return path
