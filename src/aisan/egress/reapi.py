# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Configure remote builds through the host-side RBE proxy.

``aisan.proxy.rbe`` checks methods, adds credentials, and forwards HTTP/2 frames.
This backend mints the credential and creates two files needed by Siso.

An in-box hosts entry resolves Google's real service name to loopback, preserving
the ``:authority`` used for routing. A per-box ``.sisoenv`` supplies the fully
qualified instance because tests showed that the checkout file overrides
``SISO_REAPI_INSTANCE``.

The host mints and attaches each LUCI token. The box never receives one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..mint import rbe_token as mint_rbe_token
from ..proxy import serve_rbe_unix
from ..proxy.rbe import UPSTREAM_HOST, hosts_file
from ..sandbox import BindOver, BindSpec
from .base import Backend, PreflightError

log = logging.getLogger(__name__)

HOSTS_FILE = "hosts"
SISOENV_FILE = "sisoenv"

# LUCI tokens last at most 30 minutes. Refresh with 20 minutes remaining because
# a cold V8 build can outlast one token.
_REFRESH_MARGIN_S = 1200

PORT = 8712

# The durable LUCI credential store. Mount checks keep it out of the box because
# any reader could mint its own token.
LUCI_STORE = Path.home() / ".config" / "chrome_infra"

# Preflight reports this interactive fix before the box takes over the terminal.
_FIX = "luci-auth login -scopes-cloud"


class RefreshingToken:
    """Mint on demand, cache until close to expiry, and record failures.

    Builds can outlive LUCI tokens, so the proxy checks this provider for each
    request.

    ``refused`` lets callers select offline fallback without matching Siso error
    text that may change across releases.
    """

    def __init__(self, mint) -> None:
        self._mint = mint
        self._token = ""
        self._expiry = datetime.fromtimestamp(0, UTC)
        # Serialize the first request burst so one LUCI subprocess mints the token.
        self._lock = asyncio.Lock()
        # Keep the first outage visible even if the build later recovers. A new
        # per-box backend starts with no recorded failure.
        self.refused: Exception | None = None

    async def __call__(self) -> str:
        if self._fresh(datetime.now(UTC)):
            return self._token
        async with self._lock:
            now = datetime.now(UTC)
            if self._fresh(now):  # Another waiter may have refreshed the token.
                return self._token
            try:
                self._token = await self._mint()
            except Exception as e:
                self.refused = e
                raise
            self._expiry = now + timedelta(seconds=1800)
        return self._token

    def _fresh(self, now: datetime) -> bool:
        return bool(self._token) and (self._expiry - now).total_seconds() >= (
            _REFRESH_MARGIN_S
        )


class ReapiBackend(Backend):
    """RBE through a host-side proxy on loopback `PORT`."""

    name = "rbe"
    port = PORT

    # Plaintext RBE has no client authentication. Sharing host loopback would
    # expose the proxy to every local process, so this backend only supports
    # isolated mode.
    # The unsafe shared-network profile mounts LUCI_STORE directly. Credential
    # checks prevent combining that profile with this proxy.
    supports_shared_net = False

    def __init__(
        self,
        *,
        project: str,
        sisoenv: Path | Iterable[Path],
        instance: str = "default_instance",
        mint=None,
    ) -> None:
        self._project = project
        self._instance = instance
        # Checkout-specific destinations for the generated ``.sisoenv``.
        #
        # A gclient root and worktrees on different DEPS revisions may resolve to
        # different shared files, so bind over every destination.
        #
        # Treat a string as one path instead of an iterable of characters.
        one = isinstance(sisoenv, str | Path)
        self._sisoenv_dsts = (Path(sisoenv),) if one else tuple(sisoenv)
        self.credentials = (LUCI_STORE,)
        self._token = RefreshingToken(mint or mint_rbe_token)

    def client_env(self) -> dict[str, str]:
        # Use plaintext gRPC to the relay. The bound .sisoenv supplies the
        # instance because the checkout file overrides the environment.
        return {
            "SISO_REAPI_ADDRESS": f"{UPSTREAM_HOST}:{self.port}",
            "RBE_service_no_security": "true",
        }

    def hosts_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / HOSTS_FILE

    def sisoenv_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / SISOENV_FILE

    def prepare(self, runtime_dir: Path) -> None:
        """Write the two files `box_binds` names as bind-over sources."""
        # Resolve the Google service name to the in-box loopback relay.
        hosts_file(runtime_dir)
        # Google rejects a bare instance name with CONSUMER_INVALID.
        self.sisoenv_path(runtime_dir).write_text(
            f"SISO_PROJECT={self._project}\n"
            f"SISO_REAPI_INSTANCE=projects/{self._project}/instances/{self._instance}\n"
        )

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        """Return bind-overs for the hosts file and each ``.sisoenv``.

        The hosts file preserves Google's routing authority while reaching
        loopback. ``.sisoenv`` may live in a dependency cache shared by several
        checkouts, so each box overlays it instead of editing it in place.
        """
        return [
            BindOver(self.hosts_path(runtime_dir), Path("/etc/hosts")),
            *(
                BindOver(self.sisoenv_path(runtime_dir), dst)
                for dst in self._sisoenv_dsts
            ),
        ]

    @property
    def refused(self) -> Exception | None:
        return self._token.refused

    async def preflight(self) -> None:
        try:
            await self._token()
        except Exception as e:
            raise PreflightError(
                self.name, f"cannot mint a luci token: {e}", _FIX
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        """Serve RBE for the duration of the box.

        Raise during startup so builds don't silently fall back to an offline
        mode that may take 10 to 30 minutes longer.
        """
        sock = self.socket_path(runtime_dir)
        sock.unlink(missing_ok=True)
        server = await serve_rbe_unix(sock, self._token)
        log.info("rbe proxy: listening on %s for %s", sock, UPSTREAM_HOST)
        try:
            yield
        finally:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
            sock.unlink(missing_ok=True)
