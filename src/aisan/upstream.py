# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Validate host routes and build the client that follows them.

Host proxies connect to their upstreams directly by default. Two environment
variables route that traffic through an intercepting proxy such as Burp Suite:
``AISAN_UPSTREAM_PROXY`` names an HTTP proxy, which tunnels HTTPS with CONNECT,
and ``AISAN_UPSTREAM_CA`` adds a trusted root, in PEM or DER form, to the
system store. The intercepting proxy sees the real upstream credentials.
"""

from __future__ import annotations

import os
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from yarl import URL

PROXY_ENV = "AISAN_UPSTREAM_PROXY"
CA_ENV = "AISAN_UPSTREAM_CA"


def literal(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty literal string")
    if any(c in value for c in ("{env:", "{file:", "${")):
        raise ValueError(f"{field}: interpolation is unsupported")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field}: control characters are unsupported")
    return value


def validate_upstream(value: object, field: str = "upstream") -> str:
    value = literal(value, field)
    if any(c.isspace() for c in value) or any(c in value for c in "\\?#"):
        raise ValueError(
            f"{field}: whitespace, backslashes, queries and fragments are unsupported"
        )
    authority = value.partition("://")[2].split("/", 1)[0]
    if "@" in authority:
        raise ValueError(f"{field}: userinfo is unsupported")
    try:
        url = URL(value)
        valid = (
            url.scheme in {"http", "https"}
            and bool(url.host)
            and url.user is None
            and url.password is None
            and url.port is not None
            and 0 < url.port <= 65535
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError(
            f"{field}: expected an absolute HTTP(S) URL with host and valid port, without userinfo"
        )
    return str(url).rstrip("/")


def upstream_origin(value: str) -> str:
    return str(URL(value).origin())


@dataclass(frozen=True)
class Transport:
    """The route and trust roots for host-to-upstream connections.

    The default connects directly and verifies against the system store.
    """

    proxy: str | None = None
    ca: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Transport:
        proxy = environ.get(PROXY_ENV) or None
        if proxy is not None:
            proxy = validate_upstream(proxy, PROXY_ENV)
            url = URL(proxy)
            if url.scheme != "http" or url.raw_path not in {"", "/"}:
                raise ValueError(f"{PROXY_ENV}: expected http://HOST:PORT")
        ca = environ.get(CA_ENV) or None
        return cls(proxy=proxy, ca=None if ca is None else Path(ca).expanduser())

    def describe(self) -> str | None:
        """Return the non-default route for reports, or ``None`` if direct."""
        parts = []
        if self.proxy is not None:
            parts.append(f"via {self.proxy}")
        if self.ca is not None:
            parts.append(f"trusting {self.ca}")
        return " ".join(parts) or None

    def _ssl_context(self) -> ssl.SSLContext:
        assert self.ca is not None
        data = self.ca.read_bytes()
        ctx = ssl.create_default_context()
        try:
            if b"-----BEGIN" in data:
                ctx.load_verify_locations(cadata=data.decode("ascii"))
            else:
                ctx.load_verify_locations(cadata=data)
        except (ssl.SSLError, UnicodeDecodeError) as e:
            raise ValueError(
                f"upstream CA {self.ca} is not a PEM or DER certificate: {e}"
            ) from e
        return ctx

    def session(self, timeout: ClientTimeout) -> ClientSession:
        """Open an upstream client session; call it inside the event loop."""
        connector = None if self.ca is None else TCPConnector(ssl=self._ssl_context())
        return ClientSession(timeout=timeout, proxy=self.proxy, connector=connector)


DIRECT = Transport()
