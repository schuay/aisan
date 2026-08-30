# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The OpenAI Responses transport used by Codex.

Measured against Codex CLI 0.147.0 with a custom provider and a local recording
stub. Interactive and headless turns, local function-call round trips, and
automatic context compaction all use one route::

    POST {base_url}/responses

The client requests an SSE response. A bare loopback ``base_url`` keeps the
upstream's path prefix host-side, matching the chat-completions transport: the
proxy sees ``/responses`` and an upstream such as ``https://api.openai.com/v1``
receives ``/v1/responses``.

Codex 0.147.0 declares local tools as ``function`` or ``custom`` and groups some
of them in a ``namespace``. GPT-5.6 sends the same declarations through an
``additional_tools`` input item. Compaction sends no tools. The policy permits
those local declarations recursively and refuses every server-side tool type,
including web search and code interpreter.

Forwarding, credential replacement, limits, errors, and streaming are the same
mechanism as the OpenAI-compatible chat transport. This module supplies the
Responses route, body policy, and two host-generated protocol headers to its
existing ``make_app``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import web

from .http import RateLimit, serve
from .openai_compat import BodyPolicy as _BodyPolicy
from .openai_compat import PathAllowlist as _PathAllowlist
from .openai_compat import make_app as _make_app
from .openai_compat import parse_json_object

ALLOWED_PATHS = (("POST", "/responses"),)
CLIENT_TOOL_TYPES = frozenset({"custom", "function"})
CLIENT_TOOL_CONTAINERS = frozenset({"namespace"})
ALLOWED_KEYS = frozenset(
    {
        "client_metadata",
        "include",
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "prompt_cache_key",
        "reasoning",
        "store",
        "stream",
        "text",
        "tool_choice",
        "tools",
    }
)
ALLOWED_INCLUDES = frozenset({"reasoning.encrypted_content"})
ALLOWED_INPUT_TYPES = frozenset(
    {
        "additional_tools",
        "custom_tool_call",
        "custom_tool_call_output",
        "function_call",
        "function_call_output",
        "message",
        "reasoning",
    }
)
ALLOWED_CONTENT_TYPES = frozenset({"input_text", "output_text"})


@dataclass(frozen=True)
class PathAllowlist(_PathAllowlist):
    routes: tuple[tuple[str, str], ...] = ALLOWED_PATHS


@dataclass(frozen=True)
class BodyPolicy(_BodyPolicy):
    client_types: frozenset[str] = CLIENT_TOOL_TYPES
    container_types: frozenset[str] = CLIENT_TOOL_CONTAINERS
    # The unknown-key refusal itself runs in the base policy; this is the set
    # it runs against.
    allowed_keys: frozenset[str] = ALLOWED_KEYS

    def refuse(self, body: bytes) -> str | None:
        reason = super().refuse(body)
        if reason is not None:
            return reason

        payload = parse_json_object(body)
        if payload.get("store", False) is not False:
            return "`store` must be false"
        if payload.get("stream") is not True:
            return "`stream` must be true"

        includes = payload.get("include", [])
        if not isinstance(includes, list) or any(
            not isinstance(item, str) or item not in ALLOWED_INCLUDES
            for item in includes
        ):
            return "`include` contains an unsupported response field"

        inputs = payload.get("input", [])
        if not isinstance(inputs, list):
            return "`input` must be an array"
        for item in inputs:
            reason = self._input_refusal(item)
            if reason is not None:
                return reason

        text = payload.get("text")
        if text is not None and (
            not isinstance(text, dict)
            or set(text) != {"verbosity"}
            or not isinstance(text["verbosity"], str)
            or text["verbosity"] not in {"low", "medium", "high"}
        ):
            return "`text` may only select low, medium, or high verbosity"
        return None

    def _input_refusal(self, item: object) -> str | None:
        if not isinstance(item, dict):
            return "every Responses input item must be a JSON object"
        kind = item.get("type")
        if not isinstance(kind, str) or kind not in ALLOWED_INPUT_TYPES:
            return f"Responses input type {kind!r} is not permitted"
        if kind == "additional_tools":
            if set(item) != {"type", "role", "tools"} or item["role"] != "developer":
                return "`additional_tools` must be the measured developer envelope"
            return self.refuse_tools(item["tools"])
        if kind != "message":
            return None

        content = item.get("content")
        if isinstance(content, str):
            return None
        if not isinstance(content, list):
            return "message `content` must be text or an array"
        for part in content:
            if not isinstance(part, dict):
                return "every message content part must be a JSON object"
            part_kind = part.get("type")
            if not isinstance(part_kind, str) or part_kind not in ALLOWED_CONTENT_TYPES:
                return f"Responses content type {part_kind!r} is not permitted"
        return None


CredentialSource = Callable[[], Awaitable[tuple[str, str]]]


def make_app(
    *,
    credential: CredentialSource,
    upstream: str,
    paths: PathAllowlist | None = None,
    body: BodyPolicy | None = None,
    rate: RateLimit | None = None,
    client_token: str | None = None,
) -> web.Application:
    async def authorization() -> dict[str, str]:
        token, account_id = await credential()
        return {
            "Authorization": f"Bearer {token}",
            "ChatGPT-Account-Id": account_id,
        }

    return _make_app(
        authorization=authorization,
        upstream=upstream,
        paths=paths or PathAllowlist(),
        body=body or BodyPolicy(),
        rate=rate,
        headers=_protocol_headers,
        client_token=client_token,
    )


def _protocol_headers(body: bytes) -> dict[str, str]:
    """Reconstruct measured protocol headers, never forwarding box input."""
    payload = parse_json_object(body)
    headers = {
        "Originator": "codex_cli_rs",
        "X-Codex-Beta-Features": "remote_compaction_v2",
    }
    inputs = payload.get("input", [])
    if isinstance(inputs, list) and any(
        isinstance(item, dict) and item.get("type") == "additional_tools"
        for item in inputs
    ):
        headers["X-OpenAI-Internal-Codex-Responses-Lite"] = "true"
    return headers


__all__ = ["BodyPolicy", "PathAllowlist", "make_app", "serve"]
