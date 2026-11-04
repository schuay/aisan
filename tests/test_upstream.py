# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Route upstream traffic through an intercepting proxy.

The fake upstream serves a certificate from a test CA, and the proxy tunnels
CONNECT requests to it, which is what a client sees from an intercepting proxy
that mints certificates under its own CA.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import ipaddress
import ssl
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import ClientSession, ClientTimeout, UnixConnector, web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from aisan.egress.anthropic import AnthropicBackend
from aisan.egress.base import PreflightError
from aisan.proxy import openai_compat, openai_responses
from aisan.proxy.anthropic import make_app, serve
from aisan.session import LaunchRefused, upstream_transport
from aisan.upstream import CA_ENV, DIRECT, PROXY_ENV, Transport


@dataclass(frozen=True)
class _Pki:
    ca_pem: Path
    ca_der: Path
    server: ssl.SSLContext


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


@pytest.fixture
def pki(tmp_path) -> _Pki:
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name("aisan test CA"))
        .issuer_name(_name("aisan test CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("127.0.0.1"))
        .issuer_name(ca.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = tmp_path / "ca.pem"
    ca_pem.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    ca_der = tmp_path / "ca.der"
    ca_der.write_bytes(ca.public_bytes(serialization.Encoding.DER))
    chain = tmp_path / "server.pem"
    chain.write_bytes(
        leaf.public_bytes(serialization.Encoding.PEM)
        + leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server.load_cert_chain(chain)
    return _Pki(ca_pem=ca_pem, ca_der=ca_der, server=server)


@contextlib.asynccontextmanager
async def _https_upstream(pki: _Pki, seen: list[web.Request]):
    async def handler(request: web.Request) -> web.Response:
        seen.append(request)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=pki.server).start()
    try:
        yield f"https://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionError):
        while chunk := await src.read(65536):
            dst.write(chunk)
            await dst.drain()
    with contextlib.suppress(OSError):
        dst.write_eof()


@contextlib.asynccontextmanager
async def _connect_proxy(tunnels: list[str]):
    """Serve an HTTP CONNECT proxy and record each tunnel target."""

    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        head = await r.readuntil(b"\r\n\r\n")
        method, target, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
        if method != "CONNECT":
            w.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
            w.close()
            return
        tunnels.append(target)
        host, port = target.rsplit(":", 1)
        up_r, up_w = await asyncio.open_connection(host, int(port))
        w.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await w.drain()
        await asyncio.gather(_pump(r, up_w), _pump(up_r, w))
        for writer in (w, up_w):
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()


async def _key() -> dict[str, str]:
    return {"x-api-key": "real-key"}


async def _post_through(tmp_path: Path, upstream: str, transport: Transport) -> int:
    app = make_app(authorization=_key, upstream=upstream, transport=transport)
    return await _request(tmp_path, app, "POST", "/v1/messages", {})


async def _request(
    tmp_path: Path, app: web.Application, method: str, path: str, body=None
) -> int:
    sock = tmp_path / "proxy.sock"
    runner = await serve(sock, app)
    try:
        async with (
            ClientSession(connector=UnixConnector(path=str(sock))) as box,
            box.request(method, f"http://proxy.invalid{path}", json=body) as resp,
        ):
            return resp.status
    finally:
        await runner.cleanup()


async def _codex_credential() -> tuple[str, str]:
    return "real-key", "real-account"


def _openai_compat_app(upstream: str, transport: Transport) -> web.Application:
    return openai_compat.make_app(
        authorization=_key,
        upstream=upstream,
        paths=openai_compat.PathAllowlist(routes=(("GET", "/models"),)),
        empty_body_routes=frozenset({("GET", "/models")}),
        transport=transport,
    )


def _responses_app(upstream: str, transport: Transport) -> web.Application:
    return openai_responses.make_app(
        credential=_codex_credential, upstream=upstream, transport=transport
    )


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (_openai_compat_app, "/models"),
        (_responses_app, "/models?client_version=1.0.0"),
    ],
)
async def test_openai_proxies_use_their_transport(tmp_path, pki, factory, path):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels) as proxy:
        routed = factory(up, Transport(proxy=proxy, ca=pki.ca_pem))
        assert await _request(tmp_path, routed, "GET", path) == 200
        assert len(tunnels) == 1
        assert len(seen) == 1
        direct = factory(up, DIRECT)
        assert await _request(tmp_path, direct, "GET", path) == 502
        assert len(seen) == 1


@pytest.mark.parametrize("form", ["pem", "der"])
async def test_proxy_requests_reach_the_upstream_through_the_tunnel(
    tmp_path, pki, form
):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    ca = pki.ca_pem if form == "pem" else pki.ca_der
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels) as proxy:
        status = await _post_through(tmp_path, up, Transport(proxy=proxy, ca=ca))
    assert status == 200
    assert tunnels == [up.removeprefix("https://")]
    assert [r.headers.get("x-api-key") for r in seen] == ["real-key"]


async def test_an_untrusted_interception_certificate_is_refused(tmp_path, pki):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels) as proxy:
        status = await _post_through(tmp_path, up, Transport(proxy=proxy))
    assert status == 502
    assert len(tunnels) == 1
    assert seen == []


async def test_a_ca_without_a_proxy_connects_directly(tmp_path, pki):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels):
        status = await _post_through(tmp_path, up, Transport(ca=pki.ca_pem))
    assert status == 200
    assert tunnels == []
    assert len(seen) == 1


async def test_preflight_uses_the_backend_transport(pki):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels) as proxy:
        routed = AnthropicBackend(
            api_key="k", upstream=up, transport=Transport(proxy=proxy, ca=pki.ca_der)
        )
        await routed.preflight()
        assert len(tunnels) == 1
        with pytest.raises(PreflightError):
            await AnthropicBackend(api_key="k", upstream=up).preflight()


async def test_the_backend_serves_through_its_transport(tmp_path, pki):
    seen: list[web.Request] = []
    tunnels: list[str] = []
    async with _https_upstream(pki, seen) as up, _connect_proxy(tunnels) as proxy:
        backend = AnthropicBackend(
            api_key="k", upstream=up, transport=Transport(proxy=proxy, ca=pki.ca_pem)
        )
        async with (
            backend.serve(tmp_path),
            ClientSession(
                connector=UnixConnector(path=str(backend.socket_path(tmp_path)))
            ) as box,
            box.post("http://proxy.invalid/v1/messages", json={}) as resp,
        ):
            assert resp.status == 200
    assert len(tunnels) == 1
    assert [r.headers.get("x-api-key") for r in seen] == ["k"]


async def test_a_ca_file_that_is_not_a_certificate_is_refused(tmp_path):
    bad = tmp_path / "ca.der"
    bad.write_bytes(b"not a certificate")
    with pytest.raises(ValueError, match="not a PEM or DER certificate"):
        Transport(ca=bad).session(ClientTimeout(total=1))


def test_an_empty_environment_connects_directly():
    assert Transport.from_env({}) == DIRECT
    assert Transport.from_env({PROXY_ENV: "", CA_ENV: ""}) == DIRECT
    assert DIRECT.describe() is None


def test_the_environment_names_the_proxy_and_ca(tmp_path):
    ca = tmp_path / "burp.der"
    t = Transport.from_env({PROXY_ENV: "http://127.0.0.1:8080/", CA_ENV: str(ca)})
    assert t == Transport(proxy="http://127.0.0.1:8080", ca=ca)
    assert t.describe() == f"via http://127.0.0.1:8080 trusting {ca}"


@pytest.mark.parametrize(
    "proxy",
    [
        "https://127.0.0.1:8080",
        "http://127.0.0.1:8080/path",
        "http://user:pw@127.0.0.1:8080",
        "socks5://127.0.0.1:1080",
        "127.0.0.1:8080",
    ],
)
def test_a_proxy_other_than_plain_http_host_port_is_refused(proxy):
    with pytest.raises(ValueError, match=PROXY_ENV):
        Transport.from_env({PROXY_ENV: proxy})


def test_the_launcher_refuses_a_malformed_proxy(monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "https://127.0.0.1:8080")
    with pytest.raises(LaunchRefused, match=PROXY_ENV):
        upstream_transport()


def test_explain_reports_a_routed_backend(tmp_path):
    import dataclasses

    from aisan import Box
    from aisan.explain import explain
    from aisan.presets.claude_code import claude_code

    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    ca = tmp_path / "burp.der"

    def report(transport: Transport) -> str:
        backend = AnthropicBackend(api_key="k", transport=transport)
        spec = claude_code(wt, state=state, egress=(backend,))
        spec = dataclasses.replace(
            spec, limits=dataclasses.replace(spec.limits, use_cgroup=False)
        )
        box = Box(spec, box_id="t")
        with box.staged():
            return explain(box, inputs=(), argv=False)

    routed = report(Transport(proxy="http://127.0.0.1:8080", ca=ca))
    assert "REFUSED" not in routed
    assert f"upstream via http://127.0.0.1:8080 trusting {ca}" in routed
    assert "upstream via" not in report(DIRECT)
