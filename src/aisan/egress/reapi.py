# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The RBE backend: a remote build service the box reaches without a luci token.

`aisan.proxy.rbe` owns the mechanism (method allowlist, credential
injection, frame forwarding); this owns the backend -- the upstream, the
credential source, and the two files the box needs in order to dial it.

Two files rather than none, because siso is configured differently from a model
client and neither knob is the obvious one:

- an `/etc/hosts` line, so the box dials the REAL hostname and it resolves to
  loopback. `:authority` reaches the Google frontend as sent and it routes on
  that name, so dialing `127.0.0.1` earns a 404 with an HTML body -- which
  surfaces as a content-type error and reads like anything but a routing bug.
- a `.sisoenv`, so the instance is fully qualified. The checkout ships one at a
  fixed path and it WINS over the environment (measured: `SISO_REAPI_INSTANCE`
  has no effect, even fully qualified), so the box binds over it. That also
  keeps the agent's build command untouched -- it types `autoninja` itself, so
  there is no call site to pass a flag through.

The credential is minted here, host-side (`mint.rbe_token`), and attached to a
request the box never sees: it never leaves this process. That IS the feature --
the build reaches RBE without ever holding a luci token of its own.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..mint import rbe_token as mint_rbe_token
from ..proxy import serve_rbe_unix
from ..proxy.rbe import UPSTREAM_HOST
from ..sandbox import BindOver, BindSpec
from .base import Backend, PreflightError

log = logging.getLogger(__name__)

HOSTS_FILE = "hosts"
SISOENV_FILE = "sisoenv"

# luci-auth caps a token at 30 minutes, and a cold V8 build can outlast that, so
# the margin is generous: re-mint once a third of the life is gone rather than
# discovering the expiry mid-build.
_REFRESH_MARGIN_S = 1200

PORT = 8712

# Where luci-auth keeps the durable credential `mint.rbe_token` mints from.
# Never read here -- named so the Box refuses a spec that binds it: a box that
# can read this store can mint its own token, and this backend's whole claim is
# that it cannot.
LUCI_STORE = Path.home() / ".config" / "chrome_infra"

# What `luci-auth token` tells an operator to run when it has nothing to mint
# from. Interactive, which is why preflight exists: once bwrap is up the box
# owns the TTY and this has nowhere to prompt.
_FIX = "luci-auth login -scopes-cloud"


class RefreshingToken:
    """Mints on demand, caches until nearly expired, and remembers failures.

    Read per request by the proxy rather than captured: a build runs far longer
    than a luci token lives, so a value fetched once would strand it mid-build.
    The prototype had no refresh at all, which is why this exists.

    `refused` is the structured answer to "did this build fail because we had no
    credential". The alternative -- and what the offline fallback did first -- is
    to match siso's stderr for phrases like "code = Unauthenticated". That works
    until Google or depot_tools rewords a message, and then it fails SILENTLY in
    the worst direction: an unrecognised credential failure becomes a dead job
    instead of a local build. We own both ends here, so the question is answered
    by the object that actually knows rather than inferred from prose.
    """

    def __init__(self, mint) -> None:
        self._mint = mint
        self._token = ""
        self._expiry = datetime.fromtimestamp(0, UTC)
        # The last mint failure, or None. Never cleared by a later success: what
        # the caller asks is "did a credential problem happen during this build",
        # and a build that started refused and recovered halfway still explains a
        # partial local build. Cleared only by starting a new backend, which is
        # per box.
        self.refused: Exception | None = None

    async def __call__(self) -> str:
        now = datetime.now(UTC)
        if not self._token or (self._expiry - now).total_seconds() < _REFRESH_MARGIN_S:
            try:
                self._token = await self._mint()
            except Exception as e:
                self.refused = e
                raise
            self._expiry = now + timedelta(seconds=1800)
        return self._token


class ReapiBackend(Backend):
    """RBE through a host-side proxy on loopback `PORT`."""

    name = "rbe"
    port = PORT

    # False by transport rather than by omission: under --net the relay's
    # loopback IS the host's, so the port would be an unauthenticated credential
    # capability for anything on the machine, and the insecure mode this module's
    # docstring describes leaves the box nothing to prove itself with.
    #
    # A shared-net box takes the other route instead: the `v8-rbe-with-net-unsafe`
    # egress profile mounts LUCI_STORE and lets siso authenticate as the
    # operator. Naming the store below is what keeps the two mutually exclusive
    # -- asking for both refuses on the credential guard rather than serving a
    # proxy to a box that already holds the credential.
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
        # Where THIS box's checkouts keep the file we bind over. A constructor
        # argument because where a checkout keeps it is a property of the
        # checkout, not of RBE -- and taking it here rather than in box_binds
        # keeps that method's signature the one every backend has.
        #
        # Several, because a box is not always rooted at one checkout: a gclient
        # root holds the main tree and its worktrees, and worktrees on different
        # DEPS hashes resolve to DIFFERENT shared .sisoenv files. Binding one of
        # them leaves the rest reading their own bare instance name.
        #
        # A str is one path, not an iterable of them: tuple() over it yields a
        # destination per CHARACTER, and bwrap would be handed fifteen mounts
        # nobody asked for rather than a refusal.
        one = isinstance(sisoenv, str | Path)
        self._sisoenv_dsts = (Path(sisoenv),) if one else tuple(sisoenv)
        self.credentials = (LUCI_STORE,)
        self._token = RefreshingToken(mint or mint_rbe_token)

    def client_env(self) -> dict[str, str]:
        # The address siso dials, and the flag that stops gRPC-Go from trying to
        # attach a credential the box does not have. The instance is NOT here:
        # it comes from the bound .sisoenv, because the checkout's copy of that
        # file wins over the environment (measured).
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
        etc = Path("/etc/hosts")
        existing = etc.read_text() if etc.exists() else ""
        # Prepended, not replacing: the box still needs the host's own resolution
        # for everything else it legitimately looks up.
        self.hosts_path(runtime_dir).write_text(
            f"127.0.0.1 {UPSTREAM_HOST}\n{existing}"
        )
        # Fully qualified: with a bare instance name Google cannot attribute the
        # request to a project and answers CONSUMER_INVALID.
        self.sisoenv_path(runtime_dir).write_text(
            f"SISO_PROJECT={self._project}\n"
            f"SISO_REAPI_INSTANCE=projects/{self._project}/instances/{self._instance}\n"
        )

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        """Bind-overs for the two files, neither of which can be written where
        it is needed:

        - /etc/hosts, so the box dials the real hostname and it lands on
          loopback. The Google frontend routes on :authority, so a literal
          127.0.0.1 is answered with a 404 and an HTML body.
        - the checkout's .sisoenv, once per destination. It lives in a SHARED
          deps cache (checkouts on one DEPS hash resolve to the same file, and
          some resolve to the main checkout), so editing it in place would
          corrupt concurrent boxes. Overriding per box is the only safe form.
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
        """Serve the host half.

        Raises on a failed start rather than degrading -- but note the failure
        mode differs from Vertex's. A dead model proxy means every turn is dead;
        a dead RBE proxy means the build falls back to offline (`autoninja -o`),
        which is slower but correct. It still raises, because starting a job
        whose builds will silently take 10-30 min each is worth surfacing at the
        start rather than discovering in the timings.
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
