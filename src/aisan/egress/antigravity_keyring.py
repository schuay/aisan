# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Antigravity CLI consumer login as a proxy credential.

Host agy keeps its Google login in the Secret Service keyring and is the only
writer of that item; this module reads it. agy 1.5.11 refreshes the access
token only when it runs within about ten seconds of the expiry or after it,
so aisan cannot refresh early. Once a token enters that window, aisan runs
host agy against a loopback sink in place of Cloud Code and reads the item
again. The sink answers 404, so no account or model call leaves the host.

The keyring is read by the system interpreter in a child process: libsecret's
bindings live in the distribution's Python, not in aisan's environment. The
reader never asks the service to unlock, because an unlock prompt during
preflight would have nowhere to appear.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiohttp import web

from ..hostproc import neutral_child
from ..proxy.http import serve_tcp
from .antigravity import CLOUD_CODE_URL_ENV
from .base import PreflightError
from .hostlogin import CredentialRefreshError, SingleFlight, ensure_fresh

log = logging.getLogger(__name__)

__all__ = [
    "KEYRING_ATTRIBUTES",
    "KeyringCredential",
    "KeyringError",
    "Login",
    "parse_login",
    "read_keyring",
]

# The item as ``secret-tool lookup service gemini username antigravity``.
KEYRING_ATTRIBUTES = (("service", "gemini"), ("username", "antigravity"))

_FIX_LOGIN = "run `agy` on the HOST and sign in with your Google account"
_FIX_LIBSECRET = (
    "install the Secret Service bindings for the system Python"
    " (python-gobject and libsecret)"
)
_FIX_LOCKED = "unlock the login keyring on the HOST, then retry"

# agy refreshed a token with 10.8 s left and left one with 18 min alone. Below
# this lifetime a request waits for host agy; above it the token is used as is.
_CLIENT_WINDOW_S = 10.0
_REFRESH_TIMEOUT_S = 30
_READ_TIMEOUT_S = 10
_STDERR_TAIL = 800

# Environment that selects another provider or login for agy, so that a run
# would leave the keyring item untouched.
_OTHER_LOGIN_ENV = (
    "AGY_ADC_AUTH",
    "AGY_LLM_GATEWAY_API_KEY",
    "AGY_LLM_GATEWAY_URL",
    "GEMINI_API_KEY",
    "GOOGLE_GEMINI_BASE_URL",
)

# Listing models makes agy load its login and call Cloud Code, without
# starting a conversation.
_REFRESH_ARGS = ("models",)

_SYSTEM_PYTHON = "/usr/bin/python3"
_EXIT_MISSING = 3
_EXIT_LOCKED = 4
_EXIT_NO_LIBSECRET = 5

# Search without ``SearchFlags.UNLOCK`` and report a locked item instead.
_READER = f"""
import sys
try:
    import gi
    gi.require_version("Secret", "1")
    from gi.repository import Secret
except (ImportError, ValueError):
    sys.exit({_EXIT_NO_LIBSECRET})
attributes = dict(pair.split("=", 1) for pair in sys.argv[1:])
service = Secret.Service.get_sync(Secret.ServiceFlags.NONE, None)
items = service.search_sync(
    None, attributes, Secret.SearchFlags.LOAD_SECRETS, None
)
if not items:
    sys.exit({_EXIT_MISSING})
secret = None if items[0].get_locked() else items[0].get_secret()
if secret is None:
    sys.exit({_EXIT_LOCKED})
sys.stdout.write(secret.get_text())
"""


class KeyringError(Exception):
    """The login could not be read; ``fix`` is what the operator should do."""

    def __init__(self, reason: str, fix: str) -> None:
        super().__init__(reason)
        self.fix = fix


def read_keyring(
    attributes: tuple[tuple[str, str], ...] = KEYRING_ATTRIBUTES,
    *,
    python: str = _SYSTEM_PYTHON,
) -> str:
    """Return the secret of the keyring item with these attributes."""
    try:
        result = subprocess.run(
            [
                python,
                "-I",
                "-c",
                _READER,
                *(f"{name}={value}" for name, value in attributes),
            ],
            capture_output=True,
            text=True,
            timeout=_READ_TIMEOUT_S,
            check=False,
        )
    except OSError as e:
        raise KeyringError(f"cannot run {python}: {e}", _FIX_LIBSECRET) from e
    except subprocess.TimeoutExpired as e:
        raise KeyringError("the keyring did not answer", _FIX_LOCKED) from e
    if result.returncode == _EXIT_MISSING:
        raise KeyringError("no Antigravity login in the host keyring", _FIX_LOGIN)
    if result.returncode == _EXIT_LOCKED:
        raise KeyringError("the host keyring is locked", _FIX_LOCKED)
    if result.returncode == _EXIT_NO_LIBSECRET:
        raise KeyringError(f"{python} has no libsecret bindings", _FIX_LIBSECRET)
    if result.returncode != 0:
        # The service's own message, never the secret: a failed read has none.
        detail = result.stderr.strip()[-_STDERR_TAIL:]
        raise KeyringError(f"reading the host keyring failed: {detail}", _FIX_LOCKED)
    return result.stdout


@dataclass(frozen=True)
class Login:
    access_token: str
    expires_at: float

    def remaining(self) -> float:
        return self.expires_at - time.time()


def parse_login(secret: str) -> Login:
    """Read the access token and expiry out of agy's keyring item."""
    try:
        data = json.loads(secret)
    except json.JSONDecodeError as e:
        raise KeyringError(
            "the Antigravity keyring item is not JSON", _FIX_LOGIN
        ) from e
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, dict):
        raise KeyringError("the Antigravity keyring item has no token", _FIX_LOGIN)
    # The box presents a consumer login, and Cloud Code would see another kind
    # of account behind it.
    method = data.get("auth_method")
    if method != "consumer":
        raise KeyringError(
            f"the host Antigravity login is {method!r}, not a consumer login",
            _FIX_LOGIN,
        )
    access_token = token.get("access_token")
    expiry = token.get("expiry")
    if not isinstance(access_token, str) or not access_token:
        raise KeyringError(
            "the Antigravity keyring item has no access token", _FIX_LOGIN
        )
    try:
        # Go writes nine fractional digits; ``fromisoformat`` truncates to six.
        expires_at = datetime.fromisoformat(expiry).timestamp()
    except (TypeError, ValueError) as e:
        raise KeyringError(
            "the Antigravity keyring item has no readable expiry", _FIX_LOGIN
        ) from e
    return Login(access_token, expires_at)


class KeyringCredential:
    """The host's Antigravity consumer login."""

    #: The login is in the keyring; no host file needs a mount guard.
    files: tuple[Path, ...] = ()

    def __init__(
        self,
        *,
        agy_command: tuple[str, ...] = ("agy",),
        read: Callable[[], str] = read_keyring,
    ) -> None:
        self._agy_command = agy_command
        self._read_secret = read
        self._flight = SingleFlight()
        self._cached: Login | None = None

    def _read(self) -> Login:
        """Return the login, reading the keyring only when it may have changed.

        A token outside the client's refresh window is valid whatever the
        keyring holds, and agy does not replace it before then. A different
        login made on the host takes effect when the cached token nears expiry.
        """
        cached = self._cached
        if cached is not None and cached.remaining() >= _CLIENT_WINDOW_S:
            return cached
        self._cached = login = parse_login(self._read_secret())
        return login

    async def _ensure_fresh(self) -> Login:
        return await ensure_fresh(
            self._read,
            Login.remaining,
            lambda: self._flight.join(self._attempt),
            margin=_CLIENT_WINDOW_S,
            # agy may leave a token alone until it expires, and the token
            # works until then.
            floor=0.0,
            refusal=_refusal,
        )

    async def _attempt(self) -> Exception | None:
        try:
            await _refresh_agy_login(self._agy_command)
        except (OSError, TimeoutError) as e:
            return e
        return None

    async def token(self) -> str:
        return (await self._ensure_fresh()).access_token

    async def check(self, backend: str) -> None:
        try:
            await self._ensure_fresh()
        except KeyringError as e:
            raise PreflightError(backend, str(e), e.fix) from e
        except CredentialRefreshError as e:
            raise PreflightError(backend, str(e), _FIX_LOGIN) from e


def _refusal(remaining: float, failure: Exception | None) -> str:
    """Explain an expired login without its token."""
    age = f"expired {int(-remaining)}s ago"
    if failure is not None:
        return (
            f"could not run host agy to refresh its login: {failure}."
            f" The access token in the host keyring {age}"
        )
    return (
        "host agy ran but did not refresh its login; the access token in the"
        f" host keyring {age}"
    )


async def _refresh_agy_login(command: tuple[str, ...]) -> None:
    """Run host agy so that it refreshes its keyring login."""

    async def sink(request: web.Request) -> web.Response:
        # The request carries the real access token. Never read or log its
        # headers or body; this server has no forwarding client at all.
        return web.json_response(
            {"error": {"code": 404, "message": "local refresh sink"}}, status=404
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", sink)
    runner, port = await serve_tcp(app)
    process: asyncio.subprocess.Process | None = None
    try:
        env = os.environ.copy()
        for name in _OTHER_LOGIN_ENV:
            env.pop(name, None)
        env[CLOUD_CODE_URL_ENV] = f"http://127.0.0.1:{port}"
        # agy reads workspace configuration from its cwd.
        with neutral_child(env) as child:
            process = await asyncio.create_subprocess_exec(
                *command,
                *_REFRESH_ARGS,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=child.env,
                cwd=child.cwd,
            )
            try:
                # The sink makes a nonzero status expected. Rereading the
                # keyring determines whether the refresh succeeded.
                _, stderr = await asyncio.wait_for(
                    process.communicate(), _REFRESH_TIMEOUT_S
                )
            except TimeoutError as e:
                raise TimeoutError("host agy token refresh timed out") from e
            log.info(
                "antigravity: host agy refresh run exited %d: %s",
                process.returncode,
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
