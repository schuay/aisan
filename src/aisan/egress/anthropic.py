# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Connect a boxed Anthropic client without exposing the host credential.

``aisan.proxy.anthropic`` filters and forwards requests. This module locates and
refreshes credentials, selects the upstream, and configures the boxed client.

The backend only reads credentials. Host Claude owns the OAuth file and replaces
it during refresh. When the token nears expiry, aisan runs host Claude with an
ephemeral loopback inference sink. This uses Claude's lock-aware refresh path
without sending a model request. Rereading the file determines success because a
timed-out process may have refreshed it and a successful process may not have.

Claude may authenticate through ``apiKeyHelper`` or an ``ant`` profile before
checking the OAuth file. Those sources can't be cleared from the child process,
so this case produces an explicit refresh failure.

The box receives a per-box relay token through ``CLAUDE_CODE_OAUTH_TOKEN`` for a
subscription or ``ANTHROPIC_API_KEY`` for a static key. With either variable set,
tests showed that Claude Code didn't access an OAuth file, keychain, or path
outside the transport allowlist. The proxy drops the relay token and adds the
real credential.

The variable must match the credential type because Claude Code sends different
``anthropic-beta`` values for subscription and API-key authentication.
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

from ..hostproc import neutral_child
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

# Preserve the existing import path for callers and tests.
__all__ = ["PLACEHOLDER_KEY", "AnthropicBackend"]


class CredentialRefreshError(Exception):
    """Host Claude could not leave behind a sufficiently fresh credential."""


# Compute the default path when needed so credential guards honor environment
# changes. Tests can pass a different path to the backend constructor.
def claude_config_dir() -> Path | None:
    """Return ``CLAUDE_CONFIG_DIR``, or ``None`` for the default layout.

    Claude Code 2.1.246 uses ``CLAUDE_CONFIG_DIR || ~/.claude`` for the directory
    and ``CLAUDE_CONFIG_DIR/.claude.json || ~/.claude.json`` for the config file.
    This helper shares only the override with ``session_mcp.claude_config_file``.
    """
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(configured) if configured else None


def default_credentials() -> Path:
    """Return the host Claude credential path, honoring its config redirect.

    Compute it on each call so ``known_credential_paths`` sees environment
    changes made after import.
    """
    return (claude_config_dir() or Path.home() / ".claude") / ".credentials.json"


# Callers can replace the standard endpoint with a compatible upstream.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Reject tokens close to expiry during preflight, while the operator can still
# act on a clear error. Interactive sessions may outlast this margin.
_EXPIRY_MARGIN_S = 300
_REFRESH_TIMEOUT_S = 30
# Retain enough stderr to identify a failure in the host log.
_STDERR_TAIL = 800

_FIX_EXPIRED = "claude auth login   (on the HOST -- Claude owns this file)"
_FIX_MISSING = "claude auth login   (no readable credential at that path)"
_FIX_NO_KEY = "pass a non-empty key, or drop --api-key to use the plan login"


# Public relay-token variable names used by presets and state seeding.
OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"  # noqa: S105 -- a variable name
API_KEY_ENV = "ANTHROPIC_API_KEY"
SUBSCRIPTION_ENV = "CLAUDE_CODE_SUBSCRIPTION_TYPE"


def _read_subscription(path: Path) -> str | None:
    """Return the subscription name used in the box's status line.

    Missing or invalid credentials omit this display-only value. Tests showed
    that the setting doesn't change request headers, beta flags, or body fields.
    """
    try:
        value = _read_oauth(path).get("subscriptionType")
    except (OSError, ValueError, KeyError):
        return None
    return value if isinstance(value, str) and value else None


class _Flight:
    """Share one in-progress credential refresh among waiting requests.

    Sharing the task deduplicates failed and successful attempts. A later wave
    may start another attempt after the current task finishes.

    The task outlives a cancelled request handler so credential updates can
    finish. It returns exceptions to avoid leaving failures unobserved when all
    waiters disconnect.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.task: asyncio.Task[Exception | None] | None = None


@dataclass(frozen=True)
class _PlanCredential:
    """The host's Claude Code subscription login."""

    path: Path
    claude_command: tuple[str, ...]
    _flight: _Flight = field(
        default_factory=_Flight, init=False, repr=False, compare=False
    )

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.path,)

    async def upstream(self) -> dict[str, str]:
        """Return a valid bearer, refreshing it when close to expiry.

        During a session, accept any unexpired token if refresh fails. The larger
        preflight margin exists to report failures before bubblewrap takes over
        the terminal.
        """
        token, _ = await self._ensure_fresh(floor=0.0)
        return {"authorization": f"Bearer {token}"}

    async def _ensure_fresh(self, *, floor: float) -> tuple[str, int]:
        """Refresh a near-expired token and determine success from the file.

        ``floor`` is the minimum remaining lifetime the caller accepts. Preflight
        requires the full margin; an active session accepts any valid token.
        Always reread the file because a timed-out Claude process may already
        have completed the refresh.
        """
        credential = _read_plan_credential(self.path)
        if _remaining_seconds(credential) >= _EXPIRY_MARGIN_S:
            return credential

        failure = await self._refresh()

        credential = _read_plan_credential(self.path)
        remaining = _remaining_seconds(credential)
        if remaining >= floor:
            return credential
        raise CredentialRefreshError(self._refusal(remaining, failure))

    async def _refresh(self) -> Exception | None:
        """Join the current refresh attempt or start a new one."""
        async with self._flight.lock:
            task = self._flight.task
            if task is None or task.done():
                task = asyncio.create_task(self._attempt())
                self._flight.task = task
        return await task

    async def _attempt(self) -> Exception | None:
        try:
            await _refresh_claude_login(self.claude_command, self.path.parent)
        except (OSError, TimeoutError) as e:
            return e
        return None

    def _refusal(self, remaining: float, failure: Exception | None) -> str:
        """Explain why the credential is unusable without exposing the token.

        If Claude exits successfully without updating the file, it may have used
        ``apiKeyHelper`` or an ``ant`` profile first. Clearing the child
        environment can't disable either source.
        """
        expiry = (
            f"expired {int(-remaining)}s ago"
            if remaining < 0
            else f"expires in {int(remaining)}s"
        )
        if failure is not None:
            return (
                f"could not run host Claude's refresh path: {failure}."
                f" The access token in {self.path} {expiry}. Fix: {_FIX_EXPIRED}"
            )
        return (
            f"host Claude ran but did not refresh {self.path}, whose access token"
            f" {expiry}. It may have authenticated from a source aisan cannot"
            " clear from its environment -- an apiKeyHelper setting, or an `ant`"
            f" profile. Fix: {_FIX_EXPIRED}"
        )

    def dress(self, token: str) -> dict[str, str]:
        env = {OAUTH_TOKEN_ENV: token}
        plan = _read_subscription(self.path)
        if plan is not None:
            env[SUBSCRIPTION_ENV] = plan
        return env

    async def check(self, backend: str) -> None:
        try:
            await self._ensure_fresh(floor=_EXPIRY_MARGIN_S)
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
    """Hold a static Anthropic API key on the host.

    No credential path needs protection because the caller supplies the key in
    memory. The box-facing behavior is covered by client tests, and the proxy is
    tested against a fake upstream. Sending a real API key upstream hasn't been
    tested because none was available.
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


# Keep upstream authentication and box configuration on the same credential
# object so their types can't diverge.
#
# Claude Code 2.1.246 sends ``authorization: Bearer`` and two extra beta flags
# for subscriptions, but sends ``x-api-key`` without those flags for API keys.
# The proxy forwards the beta flags, so its credential type must match the box.
_Credential = _PlanCredential | _ApiKeyCredential


class AnthropicBackend(Backend):
    """Route model calls through a host proxy on loopback ``PORT``."""

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
        """Use the subscription credential unless an API key is supplied.

        The box must be configured for the selected credential type because the
        client sends different beta flags for subscriptions and API keys. Reject
        both arguments together instead of giving one implicit precedence.
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
        # Box mount checks must keep these host credential files hidden.
        self.credentials = self._credential.files
        self._upstream = upstream
        self._rpm = rpm

    def client_env(self) -> dict[str, str]:
        """Return the relay location and placeholder credential for the box.

        The base URL routes requests to the relay. The placeholder stops Claude
        Code from searching for a real credential and trying an interactive
        login. This method stays usable by ``explain`` when the host isn't logged
        in.
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
        """Read authentication headers for one upstream request.

        Host Claude may replace the credential during a long session, so reread
        it instead of caching the preflight value.
        """
        return await self._credential.upstream()

    async def preflight(self) -> None:
        """Check the upstream and credential before starting the box.

        Check connectivity first because refreshing an expiring OAuth token also
        needs the network. The credential implementation then reports missing,
        invalid, or stale credentials with the appropriate login command.
        """
        await self._check_upstream()
        await self._credential.check(self.name)

    async def _check_upstream(self) -> None:
        """Check whether the configured upstream accepts a connection.

        Any HTTP response proves connectivity, including a 4xx or redirect. Don't
        follow redirects or spend model quota during this check.
        """
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(
                    f"{self._upstream.rstrip('/')}/api/hello",
                    allow_redirects=False,
                ),
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
        # ``serve_proxy`` removes stale sockets before binding.
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
        # ``Box`` calls this method only when the spec shares the host network.
        # Isolated serving above always uses the strict body policy.
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


def _merge_proxy_bypass(*values: str) -> list[str]:
    """Merge comma-separated no-proxy lists while preserving order."""
    merged: list[str] = []
    for value in values:
        for entry in value.split(","):
            host = entry.strip()
            if host and host not in merged:
                merged.append(host)
    return merged


async def _refresh_claude_login(command: tuple[str, ...], config_dir: Path) -> None:
    """Run Claude's OAuth refresh while trapping its model request."""

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
        # Claude Code 2.1.246 checks these authentication sources before the
        # claude.ai credential. Clear them so success requires refreshing the
        # target file. ``apiKeyHelper`` and ``ant`` profiles aren't environment
        # variables and can't be disabled here; ``_refusal`` explains that case.
        #
        # ANTHROPIC_UNIX_SOCKET is a transport override that could bypass the
        # loopback sink.
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_UNIX_SOCKET",
            "CCR_OAUTH_TOKEN_FILE",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_FOUNDRY",
            "CLAUDE_CODE_USE_VERTEX",
        ):
            env.pop(name, None)
        # Preserve exclusions from both spellings. An empty NO_PROXY mustn't
        # discard entries from a populated no_proxy.
        bypass = ",".join(
            _merge_proxy_bypass(
                env.get("NO_PROXY", ""),
                env.get("no_proxy", ""),
                "127.0.0.1",
                "localhost",
            )
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
        # Keep the cwd until the child exits so ``getcwd`` remains valid.
        with neutral_child(env) as child:
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
                stderr=asyncio.subprocess.PIPE,
                env=child.env,
                cwd=child.cwd,
            )
            try:
                # The sink makes a nonzero status expected. Rereading the
                # credential determines whether refresh succeeded.
                _, stderr = await asyncio.wait_for(
                    process.communicate(), _REFRESH_TIMEOUT_S
                )
            except TimeoutError as e:
                raise TimeoutError("host Claude token refresh timed out") from e
            if stderr:
                # Keep child stderr in the host log. Don't return untrusted child
                # output to the box in a refusal message.
                log.warning(
                    "anthropic: host Claude refresh wrote to stderr: %s",
                    stderr.decode("utf-8", "replace").strip()[-_STDERR_TAIL:],
                )
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
