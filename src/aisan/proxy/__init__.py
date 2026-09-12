# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side egress proxies for sandboxed processes.

The box holds no upstream credential. With `Sandbox.unshare_net`, it reaches
the outside world only through these proxies over UNIX sockets bound into the
box. Opted-in interactive boxes sharing the host network instead use
authenticated host-loopback TCP. The configured credential stays host-side in
both modes; the proxy route remains narrowed by its allowlist.

For isolated networks, an in-box relay connects a loopback port to the proxy's
Unix socket. Shared-network HTTP proxies listen on an authenticated random port
on host loopback. RBE uses unauthenticated HTTP/2 and supports only isolated
networks.
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
