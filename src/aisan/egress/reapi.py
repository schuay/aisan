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
import shlex
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..mint import rbe_token as mint_rbe_token
from ..proxy import serve_rbe_unix
from ..proxy.rbe import UPSTREAM_HOST, hosts_file
from ..sandbox import BindOver, BindSpec, host_platform
from ..seatbelt import seatbelt_lexpath, seatbelt_path
from ..statedir import write_sealed
from .base import Backend, PreflightError

log = logging.getLogger(__name__)

HOSTS_FILE = "hosts"
SISOENV_FILE = "sisoenv"
SISO_WRAPPER_FILE = "siso-wrapper"
# The box path the hosts bind-over replaces, before symlink resolution.
HOSTS_DST = Path("/etc/hosts")

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
    supports_direct_unix_socket = True

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

    def direct_unix_client_env(self, runtime_dir: Path) -> dict[str, str]:
        """Configure Siso on Darwin to dial `rbe.sock` directly without BindOver."""
        return {
            "SISO_REAPI_ADDRESS": f"unix://{seatbelt_lexpath(self.socket_path(runtime_dir))}",
            "SISO_PATH": seatbelt_path(self.siso_wrapper_path(runtime_dir)),
            "RBE_service_no_security": "true",
        }

    def hosts_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / HOSTS_FILE

    def sisoenv_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / SISOENV_FILE

    def siso_wrapper_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / SISO_WRAPPER_FILE

    def _fallback_siso_binary(self) -> Path:
        for dst in self._sisoenv_dsts:
            if len(dst.parents) >= 4:
                root = dst.parents[3]
                for rel in ("third_party/siso/cipd/siso", "third_party/siso/siso"):
                    cand = root / rel
                    if cand.is_file():
                        return cand
        if self._sisoenv_dsts and len(self._sisoenv_dsts[0].parents) >= 4:
            return self._sisoenv_dsts[0].parents[3] / "third_party/siso/cipd/siso"
        return Path("/usr/bin/false")

    def prepare(self, runtime_dir: Path) -> None:
        """Write runtime override files for Linux BindOver and Darwin SISO_PATH."""
        # Resolve the Google service name to the in-box loopback relay.
        hosts_file(runtime_dir)
        # Google rejects a bare instance name with CONSUMER_INVALID.
        full_instance = f"projects/{self._project}/instances/{self._instance}"
        self.sisoenv_path(runtime_dir).write_text(
            f"SISO_PROJECT={self._project}\nSISO_REAPI_INSTANCE={full_instance}\n"
        )
        fallback = shlex.quote(seatbelt_path(self._fallback_siso_binary()))
        wrapper = (
            "#!/bin/sh\n"
            f"export SISO_PROJECT={shlex.quote(self._project)}\n"
            f"export SISO_REAPI_INSTANCE={shlex.quote(full_instance)}\n"
            'd="$PWD"\n'
            'while [ "$d" != "/" ]; do\n'
            '  if [ -x "$d/third_party/siso/cipd/siso" ]; then\n'
            '    exec "$d/third_party/siso/cipd/siso" "$@"\n'
            "  fi\n"
            '  if [ -x "$d/third_party/siso/siso" ]; then\n'
            '    exec "$d/third_party/siso/siso" "$@"\n'
            "  fi\n"
            '  d=$(dirname "$d")\n'
            "done\n"
            f'exec {fallback} "$@"\n'
        )
        write_sealed(self.siso_wrapper_path(runtime_dir), wrapper, mode=0o700)

    def cleanup_prepared(self, runtime_dir: Path) -> None:
        for p in (
            self.hosts_path(runtime_dir),
            self.sisoenv_path(runtime_dir),
            self.siso_wrapper_path(runtime_dir),
        ):
            p.unlink(missing_ok=True)

    def box_binds_for_platform(
        self, runtime_dir: Path, platform: str
    ) -> list[BindSpec]:
        if platform == "darwin":
            return []
        return [
            BindOver(self.hosts_path(runtime_dir), HOSTS_DST.resolve()),
            *(
                BindOver(self.sisoenv_path(runtime_dir), dst)
                for dst in self._sisoenv_dsts
            ),
        ]

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        """Return bind-overs for Linux `bwrap`, or empty on Darwin (`SISO_PATH`)."""
        return self.box_binds_for_platform(runtime_dir, host_platform())

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
        if host_platform() == "darwin":
            server = await serve_rbe_unix(sock, self._token, allow_unix_localhost=True)
        else:
            server = await serve_rbe_unix(sock, self._token)
        log.info("rbe proxy: listening on %s for %s", sock, UPSTREAM_HOST)
        try:
            yield
        finally:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
            sock.unlink(missing_ok=True)
