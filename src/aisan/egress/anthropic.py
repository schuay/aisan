# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Anthropic backend: a model route for a box holding only a placeholder key.

`aisan.proxy.anthropic` owns the mechanism (path allowlist, header allowlist,
credential injection, forwarding); this owns the backend -- where the credential
is read from, which upstream is dialled, and what the box's client has to be told
to find the relay.

**This backend reads a credential. It never writes one.** The file is the HOST's
own Claude Code credential (0600), which Claude owns and replaces during OAuth
refresh. When the access token is near expiry, aisan delegates that operation to
the host Claude executable. Its inference URL is an ephemeral loopback sink, so
the process can run Claude's supported, lock-aware refresh path without a model
request reaching Anthropic. Success is determined only by rereading the file.

What the box gets instead of the credential is a per-box relay token, in the
environment variable that matches the credential kind the proxy will attach --
`CLAUDE_CODE_OAUTH_TOKEN` for a subscription, `ANTHROPIC_API_KEY` for a static
key. That is enough, and it is enough by measurement rather than by hope: with
either set, Claude Code authenticates with it and reaches for nothing else -- no
OAuth file, no keychain, and (measured for both dresses) no path outside the
transport's allowlist. The proxy drops that header and attaches the real
credential. So the box's token is a string whose only job is to stop the client
looking for a real one, plus to name itself to the relay; its value is
meaningless anywhere else.

Why the variable tracks the credential kind rather than being fixed: see
`_Credential` below. Short version -- the client's `anthropic-beta` set differs
by dress, and the proxy forwards that header.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from ..proxy.anthropic import BodyPolicy, make_app
from ..proxy.http import RateLimit, serve_tcp
from ..proxy.http import serve as serve_proxy
from .base import (
    PLACEHOLDER_KEY,
    SHARED_PORT_MARKER,
    SHARED_TOKEN_MARKER,
    Backend,
    BackendActivation,
    PreflightError,
    shared_proxy_token,
)

log = logging.getLogger(__name__)

PORT = 8713

# Re-exported for the callers and tests that import it from here: the constant
# moved to `base` when the second client-carrying backend grew, and a moved
# constant that silently vanished from its old home is an import error waiting
# for the next consumer.
__all__ = ["PLACEHOLDER_KEY", "AnthropicBackend"]


class CredentialRefreshError(Exception):
    """Host Claude could not leave behind a sufficiently fresh credential."""


# The default credential location, which is also where the host's Claude Code
# keeps it. A constructor argument so a test never needs the real one. The
# function is the source of truth: guards that enumerate every backend's
# credential (`egress.known_credential_paths`) call it so the answer tracks
# the environment rather than whatever HOME was at import.
def default_credentials() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


DEFAULT_CREDENTIALS = default_credentials()

# The standard Anthropic API endpoint. Callers can override it for a compatible
# proxy without changing the route exposed inside the box.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Refuse a token that expires within this window. A `claude -p` turn is seconds
# and an interactive session is not, so this is not a lifetime guarantee -- it is
# the difference between failing at preflight, where there is a fix command and a
# terminal to read it on, and failing three turns in with an opaque 401.
_EXPIRY_MARGIN_S = 300
_REFRESH_TIMEOUT_S = 30

_FIX_EXPIRED = "claude auth login   (on the HOST -- Claude owns this file)"
_FIX_MISSING = "claude auth login   (no readable credential at that path)"
_FIX_NO_KEY = "pass a non-empty key, or drop --api-key to use the plan login"


# The box-facing name of the relay token, per credential kind. Public because
# the preset snapshots and the CLI's state seeding both reason about them.
OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"  # noqa: S105 -- a variable name
API_KEY_ENV = "ANTHROPIC_API_KEY"
SUBSCRIPTION_ENV = "CLAUDE_CODE_SUBSCRIPTION_TYPE"


def _read_subscription(path: Path) -> str | None:
    """The plan name for the box's status line, or None.

    Display-only and best-effort BY DESIGN. Every failure -- absent file,
    unreadable, wrong shape, key not present -- omits the variable instead of
    raising, because `client_env` is a pure description that `explain` and the
    snapshot tests call on hosts with no credential at all. A dry run that
    tracebacked because nobody is logged in would be a worse bug than a missing
    label.

    Measured: setting it changes no request byte -- headers, `anthropic-beta`
    and body keys were identical with and without -- so nothing but the label
    depends on whether this returns a value.
    """
    try:
        value = _read_oauth(path).get("subscriptionType")
    except (OSError, ValueError, KeyError):
        return None
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class _PlanCredential:
    """The host's Claude Code subscription login."""

    path: Path
    claude_command: tuple[str, ...]
    _refresh_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.path,)

    async def upstream(self) -> dict[str, str]:
        """Return a bearer that has enough life for the next model request."""
        token, _ = await self._ensure_fresh()
        return {"authorization": f"Bearer {token}"}

    async def _ensure_fresh(self) -> tuple[str, int]:
        """Ask host Claude to refresh, while leaving it sole writer of the file."""
        credential = _read_plan_credential(self.path)
        if _remaining_seconds(credential) >= _EXPIRY_MARGIN_S:
            return credential

        # The first read avoids serialising the normal request path. Always
        # reread after taking the lock: another request may have refreshed while
        # this one waited.
        async with self._refresh_lock:
            credential = _read_plan_credential(self.path)
            if _remaining_seconds(credential) >= _EXPIRY_MARGIN_S:
                return credential

            try:
                await _refresh_claude_login(self.claude_command, self.path.parent)
            except (OSError, TimeoutError) as e:
                raise CredentialRefreshError(
                    f"could not run host Claude's refresh path: {e}"
                ) from e
            credential = _read_plan_credential(self.path)
            remaining = _remaining_seconds(credential)
            if remaining < _EXPIRY_MARGIN_S:
                expiry = (
                    f"expired {int(-remaining)}s ago"
                    if remaining < 0
                    else f"expires in {int(remaining)}s"
                )
                raise CredentialRefreshError(
                    f"host Claude ran but the access token in {self.path}"
                    f" {expiry}. Fix: {_FIX_EXPIRED}"
                )
            return credential

    def dress(self, token: str) -> dict[str, str]:
        env = {OAUTH_TOKEN_ENV: token}
        plan = _read_subscription(self.path)
        if plan is not None:
            env[SUBSCRIPTION_ENV] = plan
        return env

    async def check(self, backend: str) -> None:
        try:
            await self._ensure_fresh()
        except OSError as e:
            raise PreflightError(
                backend, f"cannot read {self.path}: {e}", _FIX_MISSING
            ) from e
        except (KeyError, ValueError) as e:
            raise PreflightError(
                backend, f"credential file is not usable: {e}", _FIX_MISSING
            ) from e
        except CredentialRefreshError as e:
            raise PreflightError(
                backend,
                f"host Claude could not refresh its login: {e}",
                _FIX_EXPIRED,
            ) from e


@dataclass(frozen=True)
class _ApiKeyCredential:
    """A static Anthropic API key, held on the host and never bound.

    Nothing on disk to name: the key is handed to the backend by its caller, so
    there is no path for `known_credential_paths` to protect -- the same shape
    as the Vertex and RBE backends, which mint in memory.

    UNMEASURED, and deliberately so: the box-facing half of this kind is the
    dress aisan shipped until this credential split, so the client's behaviour
    under it is the best-measured thing here. What has never been exercised is
    the host-side half -- a real key answering upstream -- because no key was
    available to test with. The proxy half is covered against a fake upstream
    like every other path; the contract it rests on is the one documented header.
    """

    key: str

    files: tuple[Path, ...] = ()

    async def upstream(self) -> dict[str, str]:
        return {"x-api-key": self.key}

    def dress(self, token: str) -> dict[str, str]:
        return {API_KEY_ENV: token}

    async def check(self, backend: str) -> None:
        if not self.key:
            raise PreflightError(
                backend, "the API key given to this backend is empty", _FIX_NO_KEY
            )


# A credential kind owns BOTH halves of one decision: what the proxy attaches
# upstream, and how the box is dressed to reach the relay. They sit on one object
# because they have to agree, and nothing composes them independently.
#
# The pairing is parity, not a security boundary -- a mismatch is wrong, not
# unsafe, since the credential never enters the box either way. What it costs is
# protocol accuracy. Measured (claude-cli 2.1.246): dressed for a subscription
# the client sends `authorization: Bearer` and adds `oauth-2025-04-20` and
# `extended-cache-ttl-2025-04-11` to `anthropic-beta`; dressed for an API key it
# sends `x-api-key` and omits both. The proxy forwards `anthropic-beta`, so a box
# dressed one way while the proxy attaches the other announces a protocol its
# credential does not match.
_Credential = _PlanCredential | _ApiKeyCredential


class AnthropicBackend(Backend):
    """Model calls through a host-side proxy on loopback `PORT`."""

    name = "anthropic"
    port = PORT
    supports_shared_net = True

    def __init__(
        self,
        *,
        credentials: Path | None = None,
        api_key: str | None = None,
        upstream: str = DEFAULT_UPSTREAM,
        claude_command: tuple[str, ...] = ("claude",),
        rpm: int = 120,
    ) -> None:
        """One credential kind, the subscription unless a key is given.

        The default is the plan because that is what the host's Claude Code
        login IS -- reading `~/.claude/.credentials.json` and then dressing the
        box as an API-key client made every boxed session report itself as API
        billing while the requests were billed to the subscription.

        `api_key` and `credentials` are mutually exclusive: a backend holding
        both would have to pick, and the pick would be exactly the kind of
        implicit precedence that makes a credential mix-up hard to see.
        """
        if api_key is not None and credentials is not None:
            raise ValueError("provide at most one of credentials or api_key")
        self._credential: _Credential = (
            _ApiKeyCredential(api_key)
            if api_key is not None
            else _PlanCredential(
                credentials or default_credentials(), claude_command=claude_command
            )
        )
        # For the Box's credential-exposure refusal: the files this backend
        # reads are the ones the box's mounts must leave unreadable inside it.
        # Empty for a kind that holds its credential in memory.
        self.credentials = self._credential.files
        self._upstream = upstream
        self._rpm = rpm

    def client_env(self) -> dict[str, str]:
        """What the box's Claude Code reads to find the relay, and nothing real.

        Both halves matter and for different reasons: the base URL is the route,
        and the token is what stops the client looking for a credential the box
        does not have. Setting only the first leaves an unauthenticated client
        that reads `~/.claude/.credentials.json` -- which is absent in the box,
        so it would try to log in interactively instead of failing.

        Never raises. The credential kind supplies the variable NAMES and, for a
        subscription, a display-only plan label it reads best-effort; a host with
        no credential still gets a complete description here, because `explain`
        and the snapshots call this with nothing logged in.
        """
        return self._client_env(self.port, PLACEHOLDER_KEY)

    def _client_env(self, port: int | str, token: str) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            **self._credential.dress(token),
        }

    def shared_client_env_description(self) -> dict[str, str]:
        return self._client_env(SHARED_PORT_MARKER, SHARED_TOKEN_MARKER)

    async def _upstream_auth(self) -> dict[str, str]:
        """The headers that authenticate one request. Read per request, never cached.

        Not cached deliberately: the host's Claude Code rewrites its credential
        file when it refreshes, and a value captured at preflight would go stale
        in a session that outlives one token -- while a re-read picks the new one
        up for free. The file is small and local; the read is not worth
        optimising against a correctness property.
        """
        return await self._credential.upstream()

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

        The first two are the credential kind's to answer -- what "unusable"
        means depends on what it is -- so the kind checks itself and the dead
        upstream is checked here, where the route lives.

        Claude remains the only process that mints or writes the credential.
        This check can ask a host Claude subprocess to refresh it through the
        supported request path, with inference trapped on loopback.
        """
        await self._credential.check(self.name)
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
            authorization=self._upstream_auth,
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

    @asynccontextmanager
    async def serve_shared(self, runtime_dir: Path) -> AsyncIterator[BackendActivation]:
        client_token = shared_proxy_token()
        # The relaxed body policy, and this is the only place it is built. The
        # Box calls `serve_shared` exactly when the spec does not unshare the
        # network (`box.py`), so "the box has the host's connectivity" is not
        # something a caller asserts here -- it is what choosing this transport
        # already means. `serve` above keeps the strict policy.
        app = make_app(
            authorization=self._upstream_auth,
            upstream=self._upstream,
            body=BodyPolicy.for_shared_network(),
            rate=RateLimit(per_minute=self._rpm),
            client_token=client_token,
        )
        runner, port = await serve_tcp(app)
        log.info("anthropic proxy: listening on 127.0.0.1:%d", port)
        try:
            yield BackendActivation(port, self._client_env(port, client_token))
        finally:
            await runner.cleanup()


def _read_plan_credential(path: Path) -> tuple[str, int]:
    """Read a mutually consistent access token and expiry from one snapshot."""
    data = _read_oauth(path)
    token = data.get("accessToken")
    if not isinstance(token, str) or not token:
        raise KeyError(f"{path} has no claudeAiOauth.accessToken")
    expires_ms = data.get("expiresAt")
    if not isinstance(expires_ms, int):
        raise KeyError(f"{path} has no integer claudeAiOauth.expiresAt")
    return token, expires_ms


def _remaining_seconds(credential: tuple[str, int]) -> float:
    return credential[1] / 1000.0 - time.time()


async def _refresh_claude_login(command: tuple[str, ...], config_dir: Path) -> None:
    """Drive Claude's supported OAuth refresh while trapping its model call."""

    async def sink(request: web.Request) -> web.Response:
        # Never read or log the body or headers. An HTTP answer is enough to end
        # Claude's attempt, and this server has no forwarding client at all.
        if request.path == "/api/hello":
            return web.json_response({"ok": True})
        return web.json_response(
            {"error": {"type": "api_error", "message": "local refresh sink"}},
            status=502,
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", sink)
    runner, port = await serve_tcp(app)
    process: asyncio.subprocess.Process | None = None
    try:
        env = os.environ.copy()
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_FOUNDRY",
            "CLAUDE_CODE_USE_VERTEX",
        ):
            env.pop(name, None)
        no_proxy = env.get("NO_PROXY", env.get("no_proxy", ""))
        bypass = ",".join(
            value for value in (no_proxy, "127.0.0.1", "localhost") if value
        )
        env.update(
            {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                "CLAUDE_CODE_MAX_RETRIES": "0",
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "DISABLE_AUTOUPDATER": "1",
                "NO_PROXY": bypass,
                "no_proxy": bypass,
            }
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            "haiku",
            "-p",
            "Reply with exactly hello.",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        try:
            # The sink makes a nonzero status expected. The credential reread,
            # not this status, decides whether refresh succeeded.
            await asyncio.wait_for(process.wait(), _REFRESH_TIMEOUT_S)
        except TimeoutError as e:
            raise TimeoutError("host Claude token refresh timed out") from e
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        await runner.cleanup()


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
