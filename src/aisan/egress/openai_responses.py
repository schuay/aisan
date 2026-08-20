# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Codex subscription traffic with the ChatGPT login kept on the host.

The boxed Codex uses a custom Responses provider on the in-box relay and sees
only a placeholder key. The host proxy reads the access token and account id
from the host Codex login for each request and reconstructs the subscription
headers. The refresh token and the credential file are never mounted.

Codex remains the sole writer of its login. Before a session, this backend
checks the access-token expiry. If less than one day remains, it asks the host
``codex app-server`` to perform its supported proactive refresh, then rereads
``auth.json``. aisan never sends a refresh token or writes the credential file.

The backend's command-line config overrides disable non-model egress and point
the custom provider at a bare loopback origin. CLI overrides outrank Codex's
writable per-repository user config, so trust and preferences persist without
letting project config rewrite the route. The real ChatGPT Codex path remains
host-side, so the box can reach ``/responses`` and no other subscription
endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ..proxy.http import RateLimit
from ..proxy.http import serve as serve_proxy
from ..proxy.openai_responses import make_app
from .base import PLACEHOLDER_KEY, Backend, PreflightError

log = logging.getLogger(__name__)

PORT = 8715
PROVIDER = "aisan"
CLIENT_KEY_ENV = "AISAN_CODEX_API_KEY"
DEFAULT_UPSTREAM = "https://chatgpt.com/backend-api/codex"
DEFAULT_CREDENTIALS = (
    Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
)

_REFRESH_MARGIN_S = 24 * 60 * 60
_REFRESH_TIMEOUT_S = 30
_FIX_LOGIN = "run `codex login` on the HOST with your ChatGPT subscription"


@dataclass(frozen=True)
class _ChatGPTCredential:
    access_token: str
    account_id: str
    expires_at: int


class CodexBackend(Backend):
    """ChatGPT Codex Responses through a host-side subscription proxy."""

    name = "openai-responses"
    port = PORT

    def __init__(
        self,
        *,
        model: str = "",
        upstream: str = DEFAULT_UPSTREAM,
        credentials: Path | None = None,
        codex_command: tuple[str, ...] = ("codex",),
        rpm: int = 120,
        port: int = PORT,
    ) -> None:
        self._model = model
        self._upstream = upstream
        self._credentials = credentials or DEFAULT_CREDENTIALS
        self._codex_command = codex_command
        self.credentials = (self._credentials,)
        self._rpm = rpm
        self.port = port

    def client_env(self) -> dict[str, str]:
        return {CLIENT_KEY_ENV: PLACEHOLDER_KEY}

    def config_overrides(self) -> tuple[str, ...]:
        """Highest-precedence Codex settings for this confined transport."""
        values: list[tuple[str, str | bool]] = [
            ("model_provider", PROVIDER),
            ("web_search", "disabled"),
            ("check_for_update_on_startup", False),
            ("analytics.enabled", False),
            ("feedback.enabled", False),
            ("features.apps", False),
            (f"model_providers.{PROVIDER}.name", "aisan host proxy"),
            (f"model_providers.{PROVIDER}.base_url", f"http://127.0.0.1:{self.port}"),
            (f"model_providers.{PROVIDER}.env_key", CLIENT_KEY_ENV),
            (f"model_providers.{PROVIDER}.wire_api", "responses"),
            (f"model_providers.{PROVIDER}.requires_openai_auth", False),
            (f"model_providers.{PROVIDER}.supports_websockets", False),
            (f"model_providers.{PROVIDER}.supports_standalone_web_search", False),
        ]
        if self._model:
            values.append(("model", self._model))
        return tuple(f"{key}={json.dumps(value)}" for key, value in values)

    async def _credential(self) -> tuple[str, str]:
        credential = _read_chatgpt_credential(self._credentials)
        return credential.access_token, credential.account_id

    async def preflight(self) -> None:
        try:
            credential = _read_chatgpt_credential(self._credentials)
        except (OSError, TypeError, ValueError, KeyError) as e:
            raise PreflightError(self.name, str(e), _FIX_LOGIN) from e

        if credential.expires_at - time.time() < _REFRESH_MARGIN_S:
            try:
                await _refresh_chatgpt_login(
                    self._codex_command, self._credentials.parent
                )
                credential = _read_chatgpt_credential(self._credentials)
            except (OSError, TypeError, ValueError, KeyError, TimeoutError) as e:
                raise PreflightError(
                    self.name,
                    f"host Codex could not refresh its login: {e}",
                    _FIX_LOGIN,
                ) from e
            if credential.expires_at - time.time() < _REFRESH_MARGIN_S:
                raise PreflightError(
                    self.name,
                    "host Codex refreshed its login but the access token still"
                    " expires within one day",
                    _FIX_LOGIN,
                )

        await self._check_upstream()

    async def _check_upstream(self) -> None:
        from aiohttp import ClientError, ClientSession, ClientTimeout

        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=10)) as session,
                session.get(self._upstream),
            ):
                pass
        except ClientError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} does not answer: {e}",
                "check the network, or point the backend at a different upstream",
            ) from e
        except TimeoutError as e:
            raise PreflightError(
                self.name,
                f"upstream {self._upstream} timed out",
                "check the network, or point the backend at a different upstream",
            ) from e

    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        sock = self.socket_path(runtime_dir)
        app = make_app(
            credential=self._credential,
            upstream=self._upstream,
            rate=RateLimit(per_minute=self._rpm),
        )
        runner = await serve_proxy(sock, app)
        log.info(
            "openai-responses proxy: listening on %s for 127.0.0.1:%d",
            sock,
            self.port,
        )
        try:
            yield
        finally:
            await runner.cleanup()
            sock.unlink(missing_ok=True)


def _read_chatgpt_credential(path: Path) -> _ChatGPTCredential:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON") from e
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    if data.get("auth_mode") != "chatgpt":
        raise ValueError(f"{path} is not a ChatGPT subscription login")
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise KeyError(f"{path} has no ChatGPT token set")
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not isinstance(access_token, str) or not access_token:
        raise KeyError(f"{path} has no ChatGPT access token")
    if not isinstance(account_id, str) or not account_id:
        raise KeyError(f"{path} has no ChatGPT account id")
    return _ChatGPTCredential(
        access_token=access_token,
        account_id=account_id,
        expires_at=_jwt_expiration(access_token, path),
    )


def _jwt_expiration(token: str, path: Path) -> int:
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"{path} contains a malformed ChatGPT access token")
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path} contains a malformed ChatGPT access token") from e
    expires_at = claims.get("exp") if isinstance(claims, dict) else None
    if not isinstance(expires_at, int) or isinstance(expires_at, bool):
        raise TypeError(f"{path} access token has no integer expiry")
    return expires_at


async def _refresh_chatgpt_login(command: tuple[str, ...], codex_home: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        "app-server",
        "--stdio",
        "-c",
        'model_provider="openai"',
        "-c",
        "analytics.enabled=false",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={**os.environ, "CODEX_HOME": str(codex_home)},
    )
    assert process.stdin is not None
    assert process.stdout is not None
    messages = (
        {
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "aisan",
                    "title": "aisan",
                    "version": "1",
                }
            },
        },
        {"method": "initialized", "params": {}},
        {
            "method": "account/read",
            "id": 1,
            "params": {"refreshToken": True},
        },
    )
    try:
        for message in messages:
            process.stdin.write(json.dumps(message).encode() + b"\n")
        await process.stdin.drain()

        async def read_result() -> dict:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and message.get("id") == 1:
                    return message
            raise OSError("host Codex app-server exited before answering")

        response = await asyncio.wait_for(read_result(), _REFRESH_TIMEOUT_S)
        if "error" in response:
            raise OSError("host Codex app-server refused the refresh request")
        result = response.get("result")
        account = result.get("account") if isinstance(result, dict) else None
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            raise OSError("host Codex no longer reports a ChatGPT login")
    finally:
        process.stdin.close()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()


__all__ = ["CodexBackend"]
