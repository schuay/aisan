# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Anthropic backend: a model route for a box holding only a placeholder key.

`aisan.proxy.anthropic` owns the mechanism (path allowlist, header allowlist,
credential injection, forwarding); this owns the backend -- where the credential
is read from, which upstream is dialled, and what the box's client has to be told
to find the relay.

**This backend reads a credential. It never writes one, and it never refreshes.**
The file is the HOST's own Claude Code credential (0600), which that process
owns and rewrites on its own schedule. A refresh from here would race it over a
file neither side locks, and losing that race logs the USER out -- a sandbox that
breaks the thing it is sandboxing. So an expired token is a REFUSAL with the
command to fix it, not something aisan repairs. That is a settled decision, not
a shortcut.

What the box gets instead of the credential is `ANTHROPIC_API_KEY` set to a
placeholder. That is enough, and it is enough by measurement rather than by
hope: with a key set, Claude Code sends it as `x-api-key` and reaches for
nothing else -- no OAuth file, no keychain. The proxy drops that header (see the
transport's allowlist) and attaches the real bearer. So the box's key is a
string whose only job is to stop the client looking for a real one, and its
value is deliberately meaningless.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ..proxy.anthropic import make_app
from ..proxy.http import RateLimit
from ..proxy.http import serve as serve_proxy
from .base import PLACEHOLDER_KEY, Backend, PreflightError

log = logging.getLogger(__name__)

PORT = 8713

# Re-exported for the callers and tests that import it from here: the constant
# moved to `base` when the second client-carrying backend grew, and a moved
# constant that silently vanished from its old home is an import error waiting
# for the next consumer.
__all__ = ["PLACEHOLDER_KEY", "AnthropicBackend"]

# The default credential location, which is also where the host's Claude Code
# keeps it. A constructor argument so a test never needs the real one.
DEFAULT_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"

# The standard Anthropic API endpoint. Callers can override it for a compatible
# proxy without changing the route exposed inside the box.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Refuse a token that expires within this window. A `claude -p` turn is seconds
# and an interactive session is not, so this is not a lifetime guarantee -- it is
# the difference between failing at preflight, where there is a fix command and a
# terminal to read it on, and failing three turns in with an opaque 401.
_EXPIRY_MARGIN_S = 300

_FIX_EXPIRED = "claude auth login   (on the HOST -- aisan never writes this file)"
_FIX_MISSING = "claude auth login   (no readable credential at that path)"


class AnthropicBackend(Backend):
    """Model calls through a host-side proxy on loopback `PORT`."""

    name = "anthropic"
    port = PORT

    def __init__(
        self,
        *,
        credentials: Path | None = None,
        upstream: str = DEFAULT_UPSTREAM,
        rpm: int = 120,
    ) -> None:
        self._credentials = credentials or DEFAULT_CREDENTIALS
        # For the Box's credential-exposure refusal: the one file this backend
        # reads is the one a spec bind must not contain.
        self.credentials = (self._credentials,)
        self._upstream = upstream
        self._rpm = rpm

    def client_env(self) -> dict[str, str]:
        """What the box's Claude Code reads to find the relay, and nothing real.

        Both variables matter and for different reasons: the base URL is the
        route, and the key is what stops the client looking for a credential the
        box does not have. Setting only the first leaves an unauthenticated
        client that reads `~/.claude/.credentials.json` -- which is absent in the
        box, so it would try to log in interactively instead of failing.
        """
        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.port}",
            "ANTHROPIC_API_KEY": PLACEHOLDER_KEY,
        }

    async def _bearer(self) -> str:
        """The current access token. Read per request, never cached.

        Not cached deliberately: the host's Claude Code rewrites this file when
        it refreshes, and a value captured at preflight would go stale in a
        session that outlives one token -- while a re-read picks the new one up
        for free. The file is small and local; the read is not worth optimising
        against a correctness property.
        """
        return _read_bearer(self._credentials)

    async def preflight(self) -> None:
        """Fail before bwrap owns the terminal, with the right fix for each case.

        Three things can be wrong, and they need different commands -- a
        PreflightError whose fix is wrong is worse than none:

        - no readable credential -> the user is not logged in HERE (or the path
          is wrong), so `claude auth login`;
        - a token already expired (or expiring within the margin) -> same
          command, different reason, and stated as the reason so an operator who
          IS logged in does not read "no credential" and go looking for a file
          that exists;
        - a dead upstream -> nothing to log into. The user's local proxy is not
          running, and the fix is to start it.

        No mint, by construction. This checks what is on disk and what answers;
        it does not perform an OAuth refresh -- see the module docstring.
        """
        try:
            _read_bearer(self._credentials)
            expires_ms = _read_expiry_ms(self._credentials)
        except OSError as e:
            raise PreflightError(
                self.name, f"cannot read {self._credentials}: {e}", _FIX_MISSING
            ) from e
        except (KeyError, ValueError) as e:
            raise PreflightError(
                self.name, f"credential file is not usable: {e}", _FIX_MISSING
            ) from e

        remaining = expires_ms / 1000.0 - time.time()
        if remaining < _EXPIRY_MARGIN_S:
            # The number, never the token. How long is left is diagnostic; what
            # is left is the secret.
            raise PreflightError(
                self.name,
                f"the access token in {self._credentials} expires in"
                f" {int(remaining)}s -- aisan reads this file and never refreshes"
                " it, so the host has to",
                _FIX_EXPIRED,
            )

        await self._check_upstream()

    async def _check_upstream(self) -> None:
        """Does the configured upstream answer at all.

        A connect, not a model call: the question is whether the far end of the
        chain exists, and asking it anything real would spend a turn's quota to
        learn that. Any HTTP answer counts, including a 4xx -- an upstream
        refusing THIS request is still an upstream, and only a transport failure
        means the hop is missing.
        """
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(f"{self._upstream.rstrip('/')}/api/hello"),
            ):
                pass
        except ClientError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} does not answer: {e}",
                f"start the service at {self._upstream}, or point the backend"
                " at a different upstream",
            ) from e
        except TimeoutError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} timed out",
                f"start the service at {self._upstream}, or point the backend"
                " at a different upstream",
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        # `serve_proxy` removes a leftover socket from a killed box before it
        # binds -- without that, bind() fails EADDRINUSE forever on a path this
        # box derived and nothing else owns.
        app = make_app(
            token=self._bearer,
            upstream=self._upstream,
            rate=RateLimit(per_minute=self._rpm),
        )
        runner = await serve_proxy(sock, app)
        log.info("anthropic proxy: listening on %s for 127.0.0.1:%d", sock, self.port)
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)


def _read_bearer(path: Path) -> str:
    """The OAuth access token out of a Claude Code credential file.

    Raises KeyError/ValueError/OSError naming the PATH and never the contents:
    everything under `claudeAiOauth` is either a secret or a timestamp, and a
    message that quoted the document to explain a missing key would put the
    secret wherever that message is logged.
    """
    data = _read_oauth(path)
    token = data.get("accessToken")
    if not isinstance(token, str) or not token:
        raise KeyError(f"{path} has no claudeAiOauth.accessToken")
    return token


def _read_expiry_ms(path: Path) -> int:
    """`claudeAiOauth.expiresAt`, epoch milliseconds. Same rules on messages."""
    value = _read_oauth(path).get("expiresAt")
    if not isinstance(value, int):
        raise KeyError(f"{path} has no integer claudeAiOauth.expiresAt")
    return value


def _read_oauth(path: Path) -> dict:
    import json

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON") from e
    section = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        raise KeyError(f"{path} has no claudeAiOauth section")
    return section
