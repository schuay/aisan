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

An assistant turn arrives back as ``message`` or, on the multi-agent models,
as ``agent_message`` -- ``author`` and ``recipient`` in place of ``role``, and
a content union of its own. Codex both replays the upstream's and builds its
own: an inter-agent message becomes an ``agent_message`` whose text names the
sender and whose payload rides beside it encrypted.

The gate is therefore per item type, in three buckets a new type must be sorted
into: the tag settles it, it holds parts, or it is the tool envelope. Where the
parts live differs -- a message keeps them under ``content``, a tool result
under ``output`` -- and so does what they may be, which is why a single shared
union would gate the wrong thing. Server-side tools stay refused: a
``web_search_call`` names a page for the upstream to open, and nothing local
needs it. It is the one item a recorded session replays that this policy will
not forward, so a history that already contains one -- resumed from outside a
box -- cannot be continued inside one.

Forwarding, credential replacement, limits, errors, and streaming are the same
mechanism as the OpenAI-compatible chat transport. This module supplies the
Responses route, body policy, and two host-generated protocol headers to its
existing ``make_app``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType

from aiohttp import web

from .http import RateLimit, quoted_names, serve
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
# Item types the tag settles, because what they carry names nothing for the
# upstream to fetch: `function_call` and `custom_tool_call` keep their
# arguments in a string, `reasoning` a summary and an opaque blob, `compaction`
# nothing but an opaque blob. Compaction is measured, and it is this proxy that
# asks for it -- `_protocol_headers` sends the remote-compaction beta header,
# and remote compaction answers with the one item type the turn after every
# compaction then replays.
TAG_SETTLED_INPUT_TYPES = frozenset(
    {"compaction", "custom_tool_call", "function_call", "reasoning"}
)

# The parts a container may hold, pinned key for key. The tag alone does not
# say where a part's bytes come from -- an `input_image` names a url, and a
# text part carrying an `image_url` beside its text would name one past a gate
# that read only the tag. Measured: every real part is exactly one of these
# key sets. `encrypted_content` is from the protocol type, not the wire; no
# `agent_message` traffic has been captured yet.
PART_KEYS = MappingProxyType(
    {
        "input_text": frozenset({"type", "text"}),
        "output_text": frozenset({"type", "text"}),
        "encrypted_content": frozenset({"type", "encrypted_content"}),
        "input_image": frozenset({"type", "image_url", "detail"}),
    }
)

# The part key that names a payload, and the one scheme that keeps the payload
# inline. Measured: every image a tool result carries is a base64 `data:` url,
# so refusing the rest costs nothing and any other scheme is a fetch the box
# chose and the upstream performs. Same posture as anthropic's base64-only
# source, and for the same reason: the fetch, not the bytes, is the capability.
INLINE_URL_KEYS = MappingProxyType({"input_image": "image_url"})
INLINE_URL_SCHEME = "data:"


@dataclass(frozen=True)
class Container:
    """Where an input item keeps its parts, and which parts it may hold."""

    field: str
    parts: frozenset[str]


# A message keeps its parts under `content`, a tool result under `output`, and
# the unions differ too: a `message` carries Codex's `ContentItem`, an
# `agent_message` an `AgentMessageInputContent` (text or the encrypted payload,
# no url in the type at all), a tool result the text and images a local tool
# produced. One shared union would gate every one of them against the wrong
# set.
CONTAINERS = MappingProxyType(
    {
        "message": Container("content", frozenset({"input_text", "output_text"})),
        "agent_message": Container(
            "content", frozenset({"input_text", "encrypted_content"})
        ),
        "function_call_output": Container(
            "output", frozenset({"input_text", "input_image"})
        ),
        "custom_tool_call_output": Container(
            "output", frozenset({"input_text", "input_image"})
        ),
    }
)

TOOL_ENVELOPE_INPUT_TYPE = "additional_tools"

# Derived, so that permitting a type means sorting it into a bucket. Spelled as
# its own union this is where a content-carrying type arrives permitted and
# ungated -- which is what `function_call_output` was, carrying an `input_image`
# whose `image_url` the sandbox never read.
ALLOWED_INPUT_TYPES = (
    TAG_SETTLED_INPUT_TYPES | frozenset(CONTAINERS) | {TOOL_ENVELOPE_INPUT_TYPE}
)


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
        if kind == TOOL_ENVELOPE_INPUT_TYPE:
            if set(item) != {"type", "role", "tools"} or item["role"] != "developer":
                return "`additional_tools` must be the measured developer envelope"
            return self.refuse_tools(item["tools"])
        # Only the envelope declares tools. The item's other keys are left
        # alone deliberately: the upstream returns items carrying ids and
        # metadata this side has not measured, and pinning the key set of an
        # item the box merely replays is how a turn gets refused for a field
        # that was never a capability.
        if "tools" in item:
            return f"{kind} may not declare tools"

        container = CONTAINERS.get(kind)
        if container is None:
            return None
        parts = item.get(container.field)
        # The measured shape of a tool result, and it names nothing.
        if isinstance(parts, str):
            return None
        if not isinstance(parts, list):
            return f"{kind} `{container.field}` must be text or an array"
        for part in parts:
            reason = self._part_refusal(kind, container, part)
            if reason is not None:
                return reason
        return None

    def _part_refusal(
        self, kind: str, container: Container, part: object
    ) -> str | None:
        if not isinstance(part, dict):
            return f"every {kind} content part must be a JSON object"
        part_kind = part.get("type")
        if not isinstance(part_kind, str) or part_kind not in container.parts:
            return f"{kind} content type {part_kind!r} is not permitted"
        if extra := set(part) - PART_KEYS[part_kind]:
            return (
                f"{part_kind} field(s) not permitted by the sandbox proxy:"
                f" {quoted_names(extra)}"
            )
        url_key = INLINE_URL_KEYS.get(part_kind)
        if url_key is None:
            return None
        url = part.get(url_key)
        if (
            not isinstance(url, str)
            or url[: len(INLINE_URL_SCHEME)].lower() != INLINE_URL_SCHEME
        ):
            return (
                f"{part_kind} `{url_key}` must carry the payload inline as a"
                f" {INLINE_URL_SCHEME} url: any other scheme is a fetch for the"
                " upstream"
            )
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
