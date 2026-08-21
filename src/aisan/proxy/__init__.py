# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side egress proxies for a sandboxed process.

The box holds no upstream credential. With `Sandbox.unshare_net`, it reaches
the outside world only through these proxies over UNIX sockets bound into the
box. Opted-in interactive boxes sharing the host network instead use
authenticated host-loopback TCP. The configured credential stays host-side in
both modes; the proxy route remains narrowed by its allowlist.

The isolated path has two parts: a host-side proxy listening on a UNIX socket,
and `relay`, the in-box half that offers a loopback port and splices it to that
socket. The shared-network HTTP path binds an authenticated random TCP port on
host loopback and needs no relay. HTTP provider modules handle model protocols;
`rbe` handles plaintext HTTP/2 REAPI traffic and remains isolated-only.
"""

from .http import RateLimit, run_forever, serve
from .rbe import ALLOWED_METHODS as RBE_ALLOWED_METHODS
from .rbe import hosts_file as rbe_hosts_file
from .rbe import serve_unix as serve_rbe_unix
from .vertex import Allowlist, make_app


def __getattr__(name: str):
    if name in ("run_with_relay", "serve_relay"):
        from . import relay

        return relay.run_command if name == "run_with_relay" else relay.serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RBE_ALLOWED_METHODS",
    "Allowlist",
    "RateLimit",
    "make_app",
    "rbe_hosts_file",
    "run_forever",
    "run_with_relay",
    "serve",
    "serve_rbe_unix",
    "serve_relay",
]
