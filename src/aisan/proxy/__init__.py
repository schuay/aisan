# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side egress proxies for a sandboxed process.

The box holds no credential and, with `Sandbox.unshare_net`, no network. It
reaches the outside world only through these proxies, over UNIX sockets bound
into the box. The credential stays host-side; the box gets a capability,
narrowed by an allowlist, and nothing it can steal.

Every path has the same two-part shape: a host-side proxy listening on a Unix
socket, and `relay`, the in-box half that offers a loopback port and splices it
to that socket. The socket crosses the network namespace as a filesystem object;
the relay lets clients that only dial host:port reach it. HTTP provider modules
handle model protocols; `rbe` handles plaintext HTTP/2 REAPI traffic.
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
