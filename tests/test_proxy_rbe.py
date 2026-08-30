# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The RBE proxy: method allowlist, credential injection, gRPC-shaped refusals.

The allowlist is the security boundary and its surface is wider than Vertex's --
CAS writes are permitted, because remote execution cannot work without them. So
what matters most here is that the boundary is exactly where we think it is:
the methods a real build uses, and nothing else.

The permitted set was measured from a full proxied V8 build rather than read off
the REAPI spec (see rbe.py), so these tests pin the measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

import h2.config
import h2.connection
import h2.events
import pytest

from aisan.proxy import RBE_ALLOWED_METHODS, rbe, serve_rbe_unix
from aisan.proxy.rbe import Session, hosts_file

CAS = "/build.bazel.remote.execution.v2.ContentAddressableStorage"
EXEC = "/build.bazel.remote.execution.v2.Execution"


async def _token() -> str:
    return "real-bearer"


def test_allowlist_covers_what_a_real_build_calls():
    # Every method the measured 170s V8 build issued. If one is dropped the
    # build fails mid-way with a permission error, so they are pinned by name.
    for m in (
        "/build.bazel.remote.execution.v2.Capabilities/GetCapabilities",
        "/build.bazel.remote.execution.v2.ActionCache/GetActionResult",
        f"{CAS}/FindMissingBlobs",
        f"{CAS}/BatchReadBlobs",
        f"{CAS}/BatchUpdateBlobs",
        f"{EXEC}/Execute",
        f"{EXEC}/WaitExecution",
        "/google.longrunning.Operations/GetOperation",
        "/google.longrunning.Operations/WaitOperation",
        "/google.bytestream.ByteStream/Read",
    ):
        assert m in RBE_ALLOWED_METHODS, m


def test_cas_writes_are_permitted_deliberately():
    # Not an oversight: remote execution issues BatchUpdateBlobs constantly (1260
    # times in the measured build), so a read-only allowlist would mean no remote
    # builds at all. Pinned so that removing it is a decision, not a drift.
    assert f"{CAS}/BatchUpdateBlobs" in RBE_ALLOWED_METHODS
    assert "/google.bytestream.ByteStream/Write" in RBE_ALLOWED_METHODS


def test_action_cache_writes_are_not_permitted():
    # A box that can write the action cache can poison it for every other
    # consumer of the shared rbe-chromium-untrusted instance. siso does not need
    # it (remote execution updates the cache server-side), so it stays out.
    assert not [m for m in RBE_ALLOWED_METHODS if "UpdateActionResult" in m]


@pytest.mark.parametrize(
    "method",
    [
        "/build.bazel.remote.execution.v2.ActionCache/UpdateActionResult",
        "/google.devtools.build.v1.PublishBuildEvent/PublishLifecycleEvent",
        "/google.longrunning.Operations/CancelOperation",
        f"{CAS}/BatchUpdateBlobsExtra",  # suffix smuggling
        "/build.bazel.remote.execution.v2.Execution/Execute/../../evil",
        "",
    ],
)
def test_allowlist_refuses(method):
    assert method not in RBE_ALLOWED_METHODS


def test_hosts_file_points_the_real_name_at_loopback(tmp_path):
    """The :authority mechanism: siso must dial a NAME that lands here.

    Dialing 127.0.0.1 makes the Google frontend see that as the authority and
    answer 404 with an HTML body -- which surfaces as a content-type error and
    reads like anything but a routing problem.
    """
    p = hosts_file(tmp_path)
    first = p.read_text().splitlines()[0]
    assert first == "127.0.0.1 remotebuildexecution.googleapis.com"
    # The rest of the host's resolution must survive, or the box loses DNS for
    # everything else it legitimately resolves.
    assert len(p.read_text().splitlines()) > 1


async def test_refusal_is_a_grpc_status_not_an_http_error():
    """A refused method must come back as grpc-status 7, with HTTP 200.

    gRPC reports a non-200 as a TRANSPORT failure and retries it, so an HTTP 403
    would surface to siso as a flaky connection and be retried until the build
    gives up -- hiding the actual reason. The status trailer is reported as what
    it is and stops immediately.
    """
    sent: list = []

    class _FakeConn:
        def send_headers(self, stream_id, headers, end_stream=False):
            sent.append((stream_id, dict(headers), end_stream))

    s = Session(_token, RBE_ALLOWED_METHODS)
    s._refuse(_FakeConn(), 1, "method not permitted: /evil")
    (_sid, headers, end) = sent[0]
    assert headers[b":status"] == b"200"
    assert headers[b"grpc-status"] == b"7"
    assert b"not permitted" in headers[b"grpc-message"]
    assert end is True


async def _h2_request(
    sock: Path,
    path: str,
    authority: str = rbe.UPSTREAM_HOST,
    host: str | None = None,
    extra: list[tuple[bytes, bytes]] | None = None,
):
    """Speak real HTTP/2 to the proxy and return (headers, trailers).

    A hand-rolled client rather than a gRPC one: the payload is opaque to the
    proxy, so what needs exercising is the frame and header handling, and this
    keeps the test free of a grpc dependency.

    Over the UNIX socket because that is the only transport the proxy has: the
    box has its own netns, so a host loopback port would be unreachable from it.
    What is under test here is the session, which is identical either way.

    `host` sends an explicit Host field alongside :authority. h2 refuses to send
    a block whose two disagree, so a caller exercising the rewrite passes the
    same value it puts in `authority`.
    """
    reader, writer = await asyncio.open_unix_connection(str(sock))
    conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
    conn.initiate_connection()
    sid = conn.get_next_available_stream_id()
    conn.send_headers(
        sid,
        [
            (b":method", b"POST"),
            (b":path", path.encode()),
            (b":scheme", b"http"),
            (b":authority", authority.encode()),
            *([(b"host", host.encode())] if host is not None else []),
            (b"content-type", b"application/grpc"),
            (b"te", b"trailers"),
            *(extra or []),
        ],
        end_stream=False,
    )
    # End on the HEADERS frame. A refusal closes the stream at headers, so a
    # following DATA frame would be sent on a half-closed stream -- h2 raises,
    # the helper dies before reading, and the test hangs instead of reporting.
    # The body is irrelevant to the proxy anyway: it never parses the payload.
    conn.end_stream(sid)
    writer.write(conn.data_to_send())
    await writer.drain()

    headers: dict = {}
    done = False  # NOT per-read: StreamEnded can arrive in a later frame batch
    try:
        async with asyncio.timeout(10):
            while not done:
                data = await reader.read(65536)
                if not data:
                    break
                for ev in conn.receive_data(data):
                    if isinstance(
                        ev, h2.events.ResponseReceived | h2.events.TrailersReceived
                    ):
                        headers.update(dict(ev.headers))
                    if isinstance(
                        ev, h2.events.StreamEnded | h2.events.ConnectionTerminated
                    ):
                        done = True
                if out := conn.data_to_send():
                    writer.write(out)
                    await writer.drain()
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
    return headers


async def test_refusal_happens_on_the_wire_before_the_credential(monkeypatch, tmp_path):
    """A refused method is answered with grpc-status 7 and never forwarded.

    Asserted through a real HTTP/2 exchange rather than on the allowlist set:
    the set being right proves nothing if the handler consults it too late, or
    forwards first and refuses after.

    The upstream leg is faked at the FRAME level (not by making connect raise),
    because the proxy dials upstream per connection, before any header is seen --
    a raising connect would abort the session early and make this pass whether
    or not the allowlist is enforced. That version of this test did exactly that.
    """
    from aisan.proxy import rbe

    forwarded: list[str] = []
    minted = 0

    async def counting_token() -> str:
        nonlocal minted
        minted += 1
        return "real-bearer"

    class _RecordingConn:
        """Stands in for the upstream h2 connection, recording what is sent."""

        def __init__(self):
            self._next = 1

        def get_next_available_stream_id(self):
            self._next += 2
            return self._next

        def send_headers(self, sid, headers, end_stream=False):
            forwarded.append(dict(headers).get(b":path", b"").decode())

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never  # upstream never speaks; only the request path matters

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _RecordingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, counting_token)
    denied = f"{CAS}/../../evil"
    try:
        headers = await _h2_request(sock, denied)
    finally:
        server.close()
        await server.wait_closed()

    assert headers.get(b"grpc-status") == b"7", headers
    assert denied not in forwarded, "a refused method must not be forwarded"
    assert minted == 0, "a refused method must not mint a credential"


@pytest.mark.parametrize(
    "authority",
    [
        "storage.googleapis.com",  # a real service the generic methods fit
        "storage.googleapis.com:8712",
        "127.0.0.1",  # what siso sends without the bound /etc/hosts
        "127.0.0.1:8712",
        f"{rbe.UPSTREAM_HOST}:invalid_port",
        # Test DATA, not prose: a non-ASCII "digit" that str.isdigit accepts (as
        # does str.isdecimal), so the port check needs an isascii guard to mean
        # what it says. U+0667 ARABIC-INDIC DIGIT SEVEN.
        f"{rbe.UPSTREAM_HOST}:٧",
        f"{rbe.UPSTREAM_HOST}:8712:9",  # not a port at all
        "",  # absent header
    ],
)
async def test_a_foreign_authority_is_refused_before_the_credential(
    monkeypatch, authority, tmp_path
):
    """The allowlist gates :path, but the frontend routes on :authority.

    Three allowlisted methods are generic across Google services (ByteStream
    Read/Write, longrunning Operations), so a permitted :path aimed at another
    authority would carry our cloud-scoped bearer somewhere the allowlist never
    vetted. The method here is deliberately a PERMITTED one: that is what makes
    this test about the authority check and not a second reading of the path
    check.
    """
    from aisan.proxy import rbe

    forwarded: list[tuple[str, str]] = []
    minted = 0

    async def counting_token() -> str:
        nonlocal minted
        minted += 1
        return "real-bearer"

    class _RecordingConn:
        def __init__(self):
            self._next = 1

        def get_next_available_stream_id(self):
            self._next += 2
            return self._next

        def send_headers(self, sid, headers, end_stream=False):
            h = dict(headers)
            forwarded.append(
                (
                    h.get(b":path", b"").decode(),
                    h.get(b":authority", b"").decode(),
                )
            )

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _RecordingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, counting_token)
    allowed = f"{CAS}/BatchReadBlobs"  # permitted method, foreign authority
    try:
        headers = await _h2_request(sock, allowed, authority=authority)
    finally:
        server.close()
        await server.wait_closed()

    assert headers.get(b"grpc-status") == b"7", headers
    assert forwarded == [], "a foreign authority must not be forwarded"
    assert minted == 0, "a foreign authority must not mint a credential"


async def test_an_allowlist_that_raises_denies(monkeypatch, tmp_path, caplog):
    """The fail-closed contract, asserted by breaking the policy.

    `allow` is a parameter, so the frozenset this ships with is only the default;
    a consumer passing its own container is the case that makes this reachable.
    Left to propagate the exception unwinds the session with no headers sent --
    and a silent drop is precisely what siso retries ten times before falling
    back (see rbe.py's mint-failure comment), so a policy BUG would look like a
    flaky transport and get a second chance at a request it never approved.

    The method is a PERMITTED one, so nothing but the raising container can
    explain the refusal.
    """
    from aisan.proxy import rbe

    forwarded: list[str] = []
    minted = 0

    async def counting_token() -> str:
        nonlocal minted
        minted += 1
        return "real-bearer"

    class Exploding:
        def __contains__(self, item) -> bool:
            raise RuntimeError("the policy is broken")

    class _RecordingConn:
        def __init__(self):
            self._next = 1

        def get_next_available_stream_id(self):
            self._next += 2
            return self._next

        def send_headers(self, sid, headers, end_stream=False):
            forwarded.append(dict(headers).get(b":path", b"").decode())

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _RecordingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, counting_token, allow=Exploding())
    allowed = f"{CAS}/BatchReadBlobs"
    try:
        with caplog.at_level(logging.ERROR, logger="aisan.proxy.policy"):
            headers = await _h2_request(sock, allowed)
    finally:
        server.close()
        await server.wait_closed()

    assert headers.get(b"grpc-status") == b"7", (
        f"a raising policy must refuse on the wire, not drop: {headers}"
    )
    assert forwarded == [], "a request the policy never approved must not forward"
    assert minted == 0, "a request the policy never approved must not mint"
    # Loud every time: this is a bug in the boundary, not a repeating outage.
    assert any(r.exc_info for r in caplog.records), "the bug must reach the log"


@pytest.mark.parametrize(
    "authority",
    [
        rbe.UPSTREAM_HOST,
        f"{rbe.UPSTREAM_HOST}:8712",
        f"{rbe.UPSTREAM_HOST}:443",
    ],
)
async def test_a_permitted_method_is_forwarded_with_the_credential(
    monkeypatch, authority, tmp_path
):
    """The converse, so the refusal test cannot pass by refusing everything."""
    from aisan.proxy import rbe

    seen: list[tuple[str, str]] = []

    class _RecordingConn:
        def __init__(self):
            self._next = 1

        def get_next_available_stream_id(self):
            self._next += 2
            return self._next

        def send_headers(self, sid, headers, end_stream=False):
            h = dict(headers)
            # The NAME ORDER is recorded, not asserted here: this runs inside the
            # proxy's request handler, whose broad except swallows an
            # AssertionError into a log line, so a failed assert would leave the
            # test green. Carry it out as data and judge it in the test body.
            seen.append(
                (
                    h.get(b":path", b"").decode(),
                    h.get(b"authorization", b"").decode(),
                    h.get(b":authority", b"").decode(),
                    [k for k, _ in headers],
                )
            )

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _RecordingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, _token)
    allowed = f"{EXEC}/Execute"
    try:
        # No reply is expected: the fake upstream never answers, and what is
        # under test is what the proxy FORWARDED, not what came back.
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(
                _h2_request(sock, allowed, authority=authority), timeout=3
            )
    finally:
        server.close()
        await server.wait_closed()

    assert seen, "a permitted method must be forwarded"
    path, auth, fwd_authority, names = seen[0]
    assert path == allowed
    # The box sent no Authorization; the proxy supplied one.
    assert auth == "Bearer real-bearer"
    # Pseudo-headers must all precede regular fields. Rewriting :authority by
    # filtering it out and re-appending puts it last, which h2 rejects outright
    # ("pseudo-header field out of sequence") -- so the rewrite has to replace the
    # value in place. A dict() of the headers cannot see this.
    pseudo = [i for i, k in enumerate(names) if k.startswith(b":")]
    assert pseudo == list(range(len(pseudo))), names
    # Whatever port the box dialled, upstream sees the bare host: that port is
    # the in-box relay's, an artifact of this hop, and the upstream connection is
    # https to :443, whose normal-form authority carries no port. Forwarding
    # ":8712" to Google would be the same class of bug as the 127.0.0.1 authority
    # the /etc/hosts bind exists to prevent.
    assert fwd_authority == rbe.UPSTREAM_HOST


async def test_a_host_header_is_rewritten_alongside_the_authority(
    monkeypatch, tmp_path
):
    """h2 refuses a header block whose Host and :authority disagree, and the box
    may send a Host field of its own. The proxy strips the box's loopback port
    from :authority; leaving Host untouched made the two disagree on the
    UPSTREAM send, and the resulting ProtocolError escaped the session's broad
    except and tore the whole connection down with no gRPC status -- the
    retry-storm-then-local-fallback this module exists to avoid, on demand from
    any in-box process. Host must be rewritten to the same bare value.

    The upstream connection here is a REAL client-side h2 (unlike the recording
    stub the forward test uses), because the mismatch is caught in `send_headers`
    and a non-validating fake would let the bug through green."""
    from aisan.proxy import rbe

    seen: list[list[tuple[bytes, bytes]]] = []

    class _ValidatingConn:
        def __init__(self):
            self._h2 = h2.connection.H2Connection(
                h2.config.H2Configuration(client_side=True)
            )
            self._h2.initiate_connection()

        def get_next_available_stream_id(self):
            return self._h2.get_next_available_stream_id()

        def send_headers(self, sid, headers, end_stream=False):
            # A real h2 send: a Host that disagrees with :authority raises here,
            # exactly as the upstream would. Recorded only once accepted.
            self._h2.send_headers(sid, headers, end_stream=end_stream)
            seen.append(list(headers))

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _ValidatingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, _token)
    allowed = f"{EXEC}/Execute"
    # The box dials the bound name on the relay's loopback port and sends a
    # matching Host, which is what the downstream h2 accepts on receipt; the
    # port is stripped only when :authority is normalized for the upstream.
    dialled = f"{rbe.UPSTREAM_HOST}:8712"
    try:
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(
                _h2_request(sock, allowed, authority=dialled, host=dialled),
                timeout=3,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert seen, "the request must reach upstream, not tear the session down"
    forwarded = dict(seen[0])
    assert forwarded[b":authority"] == rbe.UPSTREAM_HOST.encode()
    assert forwarded[b"host"] == rbe.UPSTREAM_HOST.encode()


async def test_a_client_disconnect_ends_the_session(monkeypatch, tmp_path):
    """A box that goes away must not pin an upstream socket forever.

    The two directions end independently, so waiting for BOTH (gather) leaves
    the upstream pump blocked on a read that may never return -- one leaked TLS
    connection and task per abandoned build, for the life of the daemon. Found
    because the fake upstream in these tests never speaks, which is exactly the
    half-open case.
    """
    from aisan.proxy import rbe

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never  # upstream never speaks, and never closes

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        class _C:
            def get_next_available_stream_id(self):
                return 1

            def send_headers(self, *a, **k):
                pass

            def data_to_send(self):
                return b""

            def initiate_connection(self):
                pass

        return _R(), _W(), _C()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, _token)
    try:
        _reader, writer = await asyncio.open_unix_connection(str(sock))
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        writer.write(conn.data_to_send())
        await writer.drain()
        # The box drops the connection mid-session.
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        # The session must wind up on its own. Without FIRST_COMPLETED this
        # never resolves and the server cannot close.
        async with asyncio.timeout(5):
            server.close()
            await server.wait_closed()
    finally:
        with contextlib.suppress(Exception):
            server.close()


async def test_a_response_larger_than_one_window_is_relayed_whole(
    monkeypatch, tmp_path
):
    """Flow control under sustained load.

    DATA used to be forwarded with a bare `send_data`, as if h2 buffered. It does
    not: it raises FlowControlError when a frame exceeds the peer's window or the
    max frame size, that exception escaped the pump, and the session ended. The
    the log filled with "Cannot send 8192 bytes, flow control window is 4581"
    and every byte of those responses was dropped -- a build streaming CAS blobs
    bigger than one 64KiB window.

    The fake upstream respects windows itself (h2 splits for nobody), so what is
    measured here is the PROXY's relaying, not the harness's.
    """
    from aisan.proxy import rbe

    blob = b"\0" * (256 * 1024)  # four windows' worth

    async def fake_connect(self):
        up = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        up.initiate_connection()
        peer = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        peer.initiate_connection()
        up.receive_data(peer.data_to_send())
        q: asyncio.Queue = asyncio.Queue()
        state = {"left": b"", "sid": None}

        def push():
            if (sid := state["sid"]) is None:
                return
            while state["left"]:
                win = min(
                    peer.local_flow_control_window(sid), peer.max_outbound_frame_size
                )
                if win <= 0:
                    break
                peer.send_data(sid, state["left"][:win], end_stream=False)
                state["left"] = state["left"][win:]
            if not state["left"]:
                peer.end_stream(sid)
                state["sid"] = None
            if d := peer.data_to_send():
                q.put_nowait(d)

        class _R:
            async def read(self, n):
                return await q.get()

        class _W:
            def write(self, d):
                for ev in peer.receive_data(d):
                    if isinstance(ev, h2.events.RequestReceived):
                        peer.send_headers(
                            ev.stream_id,
                            [
                                (b":status", b"200"),
                                (b"content-type", b"application/grpc"),
                            ],
                        )
                        state["left"], state["sid"] = blob, ev.stream_id
                        push()
                    elif isinstance(ev, h2.events.WindowUpdated):
                        push()
                if d2 := peer.data_to_send():
                    q.put_nowait(d2)

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), up

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, _token)
    try:
        r, w = await asyncio.open_unix_connection(str(sock))
        c = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        c.initiate_connection()
        w.write(c.data_to_send())
        await w.drain()
        sid = c.get_next_available_stream_id()
        c.send_headers(
            sid,
            [
                (b":method", b"POST"),
                (b":scheme", b"http"),
                (b":authority", rbe.UPSTREAM_HOST.encode()),
                (b":path", f"{CAS}/BatchReadBlobs".encode()),
                (b"content-type", b"application/grpc"),
            ],
            end_stream=True,
        )
        w.write(c.data_to_send())
        await w.drain()

        got, ended = 0, False
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            while not ended:
                data = await asyncio.wait_for(r.read(65536), timeout=5)
                if not data:
                    break  # the session died -- the bug's signature
                for ev in c.receive_data(data):
                    if isinstance(ev, h2.events.DataReceived):
                        got += len(ev.data)
                        c.acknowledge_received_data(
                            ev.flow_controlled_length, ev.stream_id
                        )
                    elif isinstance(ev, h2.events.StreamEnded):
                        ended = True
                if d := c.data_to_send():
                    w.write(d)
                    await w.drain()
        # Every byte, and a clean end: a partial relay is the same lost build.
        assert got == len(blob), f"relayed {got} of {len(blob)} bytes"
        assert ended, "the stream must be closed, not left hanging"
    finally:
        with contextlib.suppress(Exception):
            server.close()


def test_drain_closed_stream_prunes_queue_without_raising():
    """A closed/reset stream in _drain must prune its queue and not raise."""
    s = Session(_token, RBE_ALLOWED_METHODS)
    conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
    conn.initiate_connection()
    # sid 1 is not open / closed on conn.
    pending = {1: rbe._Pending()}
    pending[1].chunks.append((b"hello", 5))

    # _drain should catch StreamClosedError/NoSuchStreamError and prune pending[1]
    s._drain(pending, 1, conn, 1, None, 1)
    assert 1 not in pending


async def test_a_received_request_is_forwarded_even_if_the_box_vanishes(monkeypatch):
    """A request the proxy has fully parsed must reach Google regardless.

    The bug: the loop flushed h2's downstream housekeeping (the SETTINGS ACK)
    BEFORE translating the events from that same read. When the box had already
    gone, that write raised, the loop unwound, and the RequestReceived parsed one
    line earlier was dropped -- a request lost after the point where it was
    entirely in the proxy's hands. Measured at ~5% of connections under load.

    Simulated by a downstream writer that always raises, which is what a dead
    peer looks like from here. Order is the whole assertion: with the flush
    first, nothing is forwarded; with it after, the request goes upstream and the
    unwritable ACK is irrelevant.
    """
    forwarded: list[str] = []

    class _Conn:
        def __init__(self):
            self._n = 1

        def get_next_available_stream_id(self):
            self._n += 2
            return self._n

        def send_headers(self, sid, headers, end_stream=False):
            forwarded.append(dict(headers).get(b":path", b"").decode())

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, d):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    class _DeadWriter:
        """The box is gone: every write to it fails, as the kernel reports."""

        def write(self, d):
            raise ConnectionResetError("box went away")

        async def drain(self):
            raise ConnectionResetError("box went away")

        def close(self):
            pass

        async def wait_closed(self):
            pass

    class _OneShotReader:
        """Delivers the whole request in one read, then blocks forever."""

        def __init__(self, payload: bytes):
            self._payload = payload

        async def read(self, n):
            if self._payload:
                d, self._payload = self._payload, b""
                return d
            # Never returns: an unresolved future is how this reader "blocks
            # forever" after the one-shot payload, standing in for a peer that
            # has sent everything and not closed.
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable: the future above never resolves")

    # A real client's bytes: preface, SETTINGS, then HEADERS for an allowed
    # method -- built with h2 so this cannot drift from the wire format.
    client = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
    client.initiate_connection()
    client.send_headers(
        1,
        [
            (b":method", b"POST"),
            (b":path", f"{EXEC}/Execute".encode()),
            (b":scheme", b"http"),
            (b":authority", rbe.UPSTREAM_HOST.encode()),
            (b"content-type", b"application/grpc"),
        ],
        end_stream=True,
    )
    payload = client.data_to_send()

    session = Session(_token, RBE_ALLOWED_METHODS)
    down = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
    down.initiate_connection()
    await session._pump_down(
        _OneShotReader(payload), _DeadWriter(), down, _DeadWriter(), _Conn()
    )

    assert forwarded == [f"{EXEC}/Execute"], (
        "a fully received request was dropped because the box could not be "
        f"written to; forwarded={forwarded}"
    )


async def test_no_credential_is_refused_fast_not_dropped(monkeypatch, tmp_path):
    """A proxy with no credential must ANSWER, not vanish.

    Letting the mint error escape tore the session down with no headers ever
    sent, and siso reads a silent drop as a flaky transport: ten retries, ~40s,
    on every build. Since luci-auth here needs an interactive reauth -- the
    steady state rather than an incident -- that was the cost of every single
    build.

    UNAUTHENTICATED rather than the allowlist's PERMISSION_DENIED, because the
    two mean opposite things: one is the boundary working, the other is a
    machine that needs a human. Neither is retried, which is what makes this
    fast.
    """

    async def dead_mint() -> str:
        raise RuntimeError("luci-auth token failed (exit 1): interactive login")

    # The upstream leg is stubbed, as in every sibling test: without it this
    # dials the real remotebuildexecution.googleapis.com, and the answer under
    # test -- the mint refusal -- is only reached on a connection that opened.
    from aisan.proxy import rbe

    async def silent_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never  # upstream never speaks; this path never forwards

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        class _Conn:
            def data_to_send(self):
                return b""

        return _R(), _W(), _Conn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", silent_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, dead_mint)
    try:
        headers = await asyncio.wait_for(
            _h2_request(sock, f"{EXEC}/Execute"), timeout=10
        )
    finally:
        server.close()
        with contextlib.suppress(Exception):
            await server.wait_closed()

    # The client got a real answer, not a closed socket.
    assert headers, "the proxy dropped the connection instead of refusing"
    assert headers[b":status"] == b"200"  # a transport error would be retried
    assert headers[b"grpc-status"] == b"16"  # UNAUTHENTICATED
    assert b"no RBE credential" in headers[b"grpc-message"]


async def test_box_chosen_headers_are_dropped_from_the_forward(tmp_path, monkeypatch):
    """H5: everything the box sends rides the injected cloud bearer, so a header
    the box CHOSE that reaches Google under it is a capability the allowlist
    never vetted. `x-goog-user-project` (billing attribution) and
    `x-goog-request-params` (routing) are the sharp ones. Only the measured
    transport headers -- content-type, te, user-agent, the grpc- family, the
    REAPI RequestMetadata -- and the pseudo-headers reach the upstream.
    """
    seen: list[list[bytes]] = []

    class _RecordingConn:
        def __init__(self):
            self._next = 1

        def get_next_available_stream_id(self):
            self._next += 2
            return self._next

        def send_headers(self, sid, headers, end_stream=False):
            seen.append([k.lower() for k, _ in headers])

        def send_data(self, *a, **k):
            pass

        def end_stream(self, *a, **k):
            pass

        def reset_stream(self, *a, **k):
            pass

        def receive_data(self, data):
            return []

        def data_to_send(self):
            return b""

        def initiate_connection(self):
            pass

        def acknowledge_received_data(self, *a, **k):
            pass

    async def fake_connect(self):
        loop = asyncio.get_running_loop()
        never = loop.create_future()

        class _R:
            async def read(self, n):
                await never

        class _W:
            def write(self, d):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return _R(), _W(), _RecordingConn()

    monkeypatch.setattr(rbe.Session, "_connect_upstream", fake_connect)
    sock = tmp_path / "rbe.sock"
    server = await serve_rbe_unix(sock, _token)
    extra = [
        (b"user-agent", b"grpc-go/1.83.1"),
        (b"grpc-timeout", b"60S"),
        (b"grpc-encoding", b"gzip"),
        (b"build.bazel.remote.execution.v2.requestmetadata-bin", b"\x0a\x01x"),
        (b"x-goog-user-project", b"attacker-project"),
        (b"x-goog-request-params", b"instance=victim"),
        (b"x-forwarded-for", b"10.0.0.1"),
    ]
    try:
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(
                _h2_request(sock, f"{EXEC}/Execute", extra=extra), timeout=3
            )
    finally:
        server.close()
        await server.wait_closed()

    assert seen, "a permitted method must be forwarded"
    names = set(seen[0])
    # Dropped: box-chosen headers that would ride our bearer.
    assert b"x-goog-user-project" not in names
    assert b"x-goog-request-params" not in names
    assert b"x-forwarded-for" not in names
    # Forwarded: the measured transport set, the grpc- family, and the bearer.
    for kept in (
        b"content-type",
        b"te",
        b"user-agent",
        b"grpc-timeout",
        b"grpc-encoding",
        b"build.bazel.remote.execution.v2.requestmetadata-bin",
        b"authorization",
        b":path",
        b":authority",
    ):
        assert kept in names, kept


def _reset_event(stream_id: int) -> h2.events.StreamReset:
    return h2.events.StreamReset(stream_id=stream_id)


async def test_a_reset_stream_repays_its_queued_flow_control_debt():
    """M1: a queued chunk holds the SOURCE's window shut until it is forwarded.
    A reset drops the queue, so without repayment the source's connection-level
    window shrinks for good -- ~64KiB of it strands the peer silently, with
    `refused` None so the offline fallback never fires. Both queues are repaid
    to their own source: the box-bound queue owes the box, the upstream-bound
    queue owes the upstream.
    """
    s = Session(_token, RBE_ALLOWED_METHODS)
    s._c2u[1] = 2
    s._u2c[2] = 1
    s._queue(s._to_up, 1, b"x" * 100, 100)
    s._queue(s._to_up, 1, b"y" * 40, 40)
    s._queue(s._to_down, 1, b"z" * 200, 200)

    repaid: list[tuple[str, int, int]] = []

    class _Conn:
        def __init__(self, tag):
            self.tag = tag

        def acknowledge_received_data(self, size, sid):
            repaid.append((self.tag, size, sid))

        def reset_stream(self, sid):
            pass

    down, up = _Conn("down"), _Conn("up")
    await s._on_down_event(_reset_event(1), down, up)

    # Box-owed chunks repaid to the box at the client stream id.
    assert ("down", 100, 1) in repaid
    assert ("down", 40, 1) in repaid
    # Upstream-owed chunk repaid to the upstream at its twin stream id.
    assert ("up", 200, 2) in repaid
    # Queues cleared.
    assert 1 not in s._to_up
    assert 1 not in s._to_down


async def test_a_drained_closed_stream_repays_the_source():
    """The other drop site: _drain hits StreamClosedError on the destination and
    prunes the queue. The queued bytes still owe the source, so the same
    repayment applies -- a closed twin must not leak the box's window either.
    """
    s = Session(_token, RBE_ALLOWED_METHODS)
    dest = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
    dest.initiate_connection()  # stream 1 is not open on it -> StreamClosedError

    repaid: list[tuple[int, int]] = []

    class _Src:
        def acknowledge_received_data(self, size, sid):
            repaid.append((size, sid))

    pending = {1: rbe._Pending()}
    pending[1].chunks.append((b"hello", 5))
    pending[1].chunks.append((b"world!", 6))

    s._drain(pending, 1, dest, 1, _Src(), 7)

    assert 1 not in pending
    assert (5, 7) in repaid and (6, 7) in repaid


async def test_a_non_utf8_path_is_refused_not_a_torn_connection():
    """L3: a :path with 0x80-0xff bytes is legal at the h2 layer. Decoding it
    strict used to raise out of the handler, through the broad except, tearing
    the connection down with no gRPC status -- which siso reads as a flaky
    transport and retries. It is refused cleanly now (a replaced path cannot be
    an ASCII REAPI method)."""
    s = Session(_token, RBE_ALLOWED_METHODS)
    refusals: list[tuple] = []

    class _Down:
        def send_headers(self, sid, headers, end_stream=False):
            refusals.append((sid, dict(headers)))

    ev = h2.events.RequestReceived(
        stream_id=1,
        headers=[
            (b":method", b"POST"),
            (b":path", b"/build.bazel.\xff\xfe/X"),
            (b":scheme", b"http"),
            (b":authority", rbe.UPSTREAM_HOST.encode()),
        ],
    )
    # Must not raise, and must answer a gRPC PERMISSION_DENIED trailer.
    await s._on_down_event(ev, _Down(), object())
    assert refusals, "the request was dropped instead of refused"
    trailer = refusals[0][1]
    assert trailer[b"grpc-status"] == b"7"


def test_a_sustained_mint_outage_logs_once_per_window_not_once_ever(
    caplog, monkeypatch
):
    """W4: the reset used to run on every failure, so `now - _mint_failed_at`
    stayed under the window forever and a sustained outage logged once ever.
    Measuring from the last LOG restores once-per-window."""
    s = Session(_token, RBE_ALLOWED_METHODS)
    real = time.monotonic()
    clock = {"t": real}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    with caplog.at_level(logging.WARNING, logger="aisan.proxy.rbe"):
        for _ in range(50):  # a burst inside one window
            s._note_mint_failure(RuntimeError("no creds"))
            clock["t"] += 1  # a second between requests
        first = len([r for r in caplog.records if "no credential" in r.getMessage()])
        clock["t"] += rbe._MINT_FAILURE_LOG_S + 1  # past the window
        s._note_mint_failure(RuntimeError("no creds"))
    total = len([r for r in caplog.records if "no credential" in r.getMessage()])
    assert first == 1, f"logged {first} times inside one window"
    assert total == 2, "the window did not reopen"
