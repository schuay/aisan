# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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

    assert f"{CAS}/BatchUpdateBlobs" in RBE_ALLOWED_METHODS
    assert "/google.bytestream.ByteStream/Write" in RBE_ALLOWED_METHODS


def test_action_cache_writes_are_not_permitted():

    assert not [m for m in RBE_ALLOWED_METHODS if "UpdateActionResult" in m]


@pytest.mark.parametrize(
    "method",
    [
        "/build.bazel.remote.execution.v2.ActionCache/UpdateActionResult",
        "/google.devtools.build.v1.PublishBuildEvent/PublishLifecycleEvent",
        "/google.longrunning.Operations/CancelOperation",
        f"{CAS}/BatchUpdateBlobsExtra",
        "/build.bazel.remote.execution.v2.Execution/Execute/../../evil",
        "",
    ],
)
def test_allowlist_refuses(method):
    assert method not in RBE_ALLOWED_METHODS


def test_hosts_file_points_the_real_name_at_loopback(tmp_path):
    p = hosts_file(tmp_path)
    first = p.read_text().splitlines()[0]
    assert first == "127.0.0.1 remotebuildexecution.googleapis.com"

    assert len(p.read_text().splitlines()) > 1


async def test_refusal_is_a_grpc_status_not_an_http_error():
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

    conn.end_stream(sid)
    writer.write(conn.data_to_send())
    await writer.drain()

    headers: dict = {}
    done = False
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
    from aisan.proxy import rbe

    forwarded: list[str] = []
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
        "storage.googleapis.com",
        "storage.googleapis.com:8712",
        "127.0.0.1",
        "127.0.0.1:8712",
        f"{rbe.UPSTREAM_HOST}:invalid_port",
        f"{rbe.UPSTREAM_HOST}:٧",
        f"{rbe.UPSTREAM_HOST}:8712:9",
        "",
    ],
)
async def test_a_foreign_authority_is_refused_before_the_credential(
    monkeypatch, authority, tmp_path
):
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
    allowed = f"{CAS}/BatchReadBlobs"
    try:
        headers = await _h2_request(sock, allowed, authority=authority)
    finally:
        server.close()
        await server.wait_closed()

    assert headers.get(b"grpc-status") == b"7", headers
    assert forwarded == [], "a foreign authority must not be forwarded"
    assert minted == 0, "a foreign authority must not mint a credential"


async def test_an_allowlist_that_raises_denies(monkeypatch, tmp_path, caplog):
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

    assert auth == "Bearer real-bearer"

    pseudo = [i for i, k in enumerate(names) if k.startswith(b":")]
    assert pseudo == list(range(len(pseudo))), names

    assert fwd_authority == rbe.UPSTREAM_HOST


async def test_a_host_header_is_rewritten_alongside_the_authority(
    monkeypatch, tmp_path
):
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
    from aisan.proxy import rbe

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

        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()

        async with asyncio.timeout(5):
            server.close()
            await server.wait_closed()
    finally:
        with contextlib.suppress(Exception):
            server.close()


async def test_a_response_larger_than_one_window_is_relayed_whole(
    monkeypatch, tmp_path
):
    from aisan.proxy import rbe

    blob = b"\0" * (256 * 1024)

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
                    break
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

        assert got == len(blob), f"relayed {got} of {len(blob)} bytes"
        assert ended, "the stream must be closed, not left hanging"
    finally:
        with contextlib.suppress(Exception):
            server.close()


def test_drain_closed_stream_prunes_queue_without_raising():
    s = Session(_token, RBE_ALLOWED_METHODS)
    conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
    conn.initiate_connection()

    pending = {1: rbe._Pending()}
    pending[1].chunks.append((b"hello", 5))

    s._drain(pending, 1, conn, 1, None, 1)
    assert 1 not in pending


async def test_a_received_request_is_forwarded_even_if_the_box_vanishes(monkeypatch):
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
        def write(self, d):
            raise ConnectionResetError("box went away")

        async def drain(self):
            raise ConnectionResetError("box went away")

        def close(self):
            pass

        async def wait_closed(self):
            pass

    class _OneShotReader:
        def __init__(self, payload: bytes):
            self._payload = payload

        async def read(self, n):
            if self._payload:
                d, self._payload = self._payload, b""
                return d

            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable: the future above never resolves")

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

    async def dead_mint() -> str:
        raise RuntimeError("luci-auth token failed (exit 1): interactive login")

    from aisan.proxy import rbe

    async def silent_connect(self):
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

    assert headers, "the proxy dropped the connection instead of refusing"
    assert headers[b":status"] == b"200"
    assert headers[b"grpc-status"] == b"16"
    assert b"no RBE credential" in headers[b"grpc-message"]


async def test_box_chosen_headers_are_dropped_from_the_forward(tmp_path, monkeypatch):
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

    assert b"x-goog-user-project" not in names
    assert b"x-goog-request-params" not in names
    assert b"x-forwarded-for" not in names

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

    assert ("down", 100, 1) in repaid
    assert ("down", 40, 1) in repaid

    assert ("up", 200, 2) in repaid

    assert 1 not in s._to_up
    assert 1 not in s._to_down


async def test_a_drained_closed_stream_repays_the_source():
    s = Session(_token, RBE_ALLOWED_METHODS)
    dest = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
    dest.initiate_connection()

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

    await s._on_down_event(ev, _Down(), object())
    assert refusals, "the request was dropped instead of refused"
    trailer = refusals[0][1]
    assert trailer[b"grpc-status"] == b"7"


def test_a_sustained_mint_outage_logs_once_per_window_not_once_ever(
    caplog, monkeypatch
):
    s = Session(_token, RBE_ALLOWED_METHODS)
    real = time.monotonic()
    clock = {"t": real}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    with caplog.at_level(logging.WARNING, logger="aisan.proxy.rbe"):
        for _ in range(50):
            s._note_mint_failure(RuntimeError("no creds"))
            clock["t"] += 1
        first = len([r for r in caplog.records if "no credential" in r.getMessage()])
        clock["t"] += rbe._MINT_FAILURE_LOG_S + 1
        s._note_mint_failure(RuntimeError("no creds"))
    total = len([r for r in caplog.records if "no credential" in r.getMessage()])
    assert first == 1, f"logged {first} times inside one window"
    assert total == 2, "the window did not reopen"
