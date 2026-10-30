# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Claude Code subscription login as a proxy credential.

Host Claude owns the OAuth file and replaces it during refresh; this module
only reads it. When the token nears expiry, aisan runs host Claude with an
ephemeral loopback inference sink. This uses Claude's lock-aware refresh path
without sending a model request. Rereading the file determines success because
a timed-out process may have refreshed it and a successful process may not
have.

Claude may authenticate through ``apiKeyHelper`` or an ``ant`` profile before
checking the OAuth file. Those sources can't be cleared from the child process,
so this case produces an explicit refresh failure.

The box receives a per-box relay token through ``CLAUDE_CODE_OAUTH_TOKEN``.
With it set, tests showed that Claude Code didn't access an OAuth file,
keychain, or path outside the transport allowlist.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from ..hostproc import neutral_child
from ..proxy.http import serve_tcp
from .base import PreflightError

log = logging.getLogger(__name__)


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


# Reject tokens close to expiry during preflight, while the operator can still
# act on a clear error. Interactive sessions may outlast this margin.
_EXPIRY_MARGIN_S = 300
_REFRESH_TIMEOUT_S = 30
# Retain enough stderr to identify a failure in the host log.
_STDERR_TAIL = 800

_FIX_EXPIRED = "claude auth login   (on the HOST -- Claude owns this file)"
_FIX_MISSING = "claude auth login   (no readable credential at that path)"


# The relay-token variable and the display-only plan label, used by presets
# and state seeding.
OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"  # noqa: S105 -- a variable name
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
class PlanCredential:
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
