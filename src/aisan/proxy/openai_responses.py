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

So the gate is per item type, in three buckets a new type has to be sorted
into: the tag settles it, it holds parts, or it is the tool envelope. Where the
parts live differs -- a message keeps them under ``content``, a tool result
under ``output`` -- and so does what they may be.

Server-side tools stay refused. A ``web_search_call`` names a page for the
upstream to open and nothing a local turn does needs it; likewise
``local_shell_call``, ``tool_search_call`` and ``image_generation_call``. Codex
persists and replays all of them, so a history that collected one outside a box
cannot be continued inside one. Of the four, only ``web_search_call`` has been
seen in a recorded session.

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
# Item types the tag settles, because nothing they carry can name a fetch:
# `function_call` keeps its arguments in a string, `custom_tool_call` its
# input, `compaction` nothing but an opaque blob and an id. They hold no parts
# at all, so a tagged array on one of them is refused outright below.
#
# Compaction is the odd one, and not for the reason it looks like. A boxed
# client never produces it: remote compaction wants a provider Codex
# recognises as OpenAI's, and this transport is a loopback under a name of its
# own, so a box compacts locally and sends plain messages. The beta header
# `_protocol_headers` sends advertises the client's features rather than asking
# for any. It is permitted because a history recorded outside a box does carry
# the item -- measured -- and replaying it is how that session resumes inside
# one. Its counterpart `compaction_trigger`, the item that ASKS the upstream to
# compact, stays refused: replaying a summary a recorded session already holds
# is not the same act as a box asking OpenAI to produce one.
TAG_SETTLED_INPUT_TYPES = frozenset({"compaction", "custom_tool_call", "function_call"})

# The parts a container may hold, pinned key for key. The tag alone does not
# say where a part's bytes come from: a text part carrying an `image_url`
# beside its text names one past a gate that reads only the tag, which is
# cfdf7b6's finding one protocol over. Every part observed in a recorded
# session is exactly one of these key sets. `encrypted_content` is from the
# protocol type instead, no `agent_message` having been captured on the wire.
PART_KEYS = MappingProxyType(
    {
        "input_text": frozenset({"type", "text"}),
        "output_text": frozenset({"type", "text"}),
        "encrypted_content": frozenset({"type", "encrypted_content"}),
        "input_image": frozenset({"type", "image_url", "detail"}),
        "summary_text": frozenset({"type", "text"}),
        "reasoning_text": frozenset({"type", "text"}),
        "text": frozenset({"type", "text"}),
    }
)

# The part key that names a payload, and the one scheme that keeps the payload
# inline, as anthropic's source gate does with base64. Codex strips a remote
# image url before the socket sees it and sends the bytes instead, and every
# image observed in a recorded session is a `data:` url -- but the box is not
# its client, and any other scheme is a fetch the box chose and the upstream
# performs.
INLINE_URL_KEYS = MappingProxyType({"input_image": "image_url"})
INLINE_URL_SCHEME = "data:"

# Part keys that carry the payload itself. A key outside this set is either the
# part's inline-url key or an unreviewed way to name a payload, and the test
# that pins the two together reads this rather than guessing from the name: a
# key called `url`, `uri`, `src` or `file_id` names one just as well as one
# ending `_url`.
INERT_PART_KEYS = frozenset({"type", "text", "detail", "encrypted_content"})


@dataclass(frozen=True)
class Container:
    """Where an input item keeps its parts, and which parts it may hold."""

    field: str
    parts: frozenset[str]
    # `reasoning` sends an empty summary and a null content, so a part field
    # is gated when it is there and not required to be.
    optional: bool = False


# `message` carries Codex's `ContentItem`, `agent_message` an
# `AgentMessageInputContent` -- text or the encrypted payload, no url in the
# type at all -- and a tool result the text and images a local tool produced.
# An image rides either way only inline, so the union admits it and the scheme
# check settles it.
CONTAINERS = MappingProxyType(
    {
        "message": (
            Container(
                "content", frozenset({"input_text", "output_text", "input_image"})
            ),
        ),
        "agent_message": (
            Container("content", frozenset({"input_text", "encrypted_content"})),
        ),
        "function_call_output": (
            Container("output", frozenset({"input_text", "input_image"})),
        ),
        "custom_tool_call_output": (
            Container("output", frozenset({"input_text", "input_image"})),
        ),
        "reasoning": (
            Container("summary", frozenset({"summary_text"}), optional=True),
            Container("content", frozenset({"reasoning_text", "text"}), optional=True),
        ),
    }
)

TOOL_ENVELOPE_INPUT_TYPE = "additional_tools"

# Derived, so that permitting a type means sorting it into a bucket. Spelled as
# a union of its own, this is where a content-carrying type arrives permitted
# and ungated: `function_call_output` did, holding an `input_image` whose
# `image_url` nothing on this side read.
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
        # Not `.get(..., False)`: the API's own default for an absent `store`
        # is to retain, so absence is the opposite of what this asks for.
        # Measured across recorded turns, the client always sends it.
        if payload.get("store") is not False:
            return "`store` must be present and false"
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
        # alone: the upstream returns items carrying ids and metadata this side
        # has not measured, and pinning those refuses a turn over a field that
        # was never a capability.
        if "tools" in item:
            return f"{kind} may not declare tools"

        containers = CONTAINERS.get(kind, ())
        # A part array anywhere but the field its own type keeps parts in is a
        # payload this side never reads. Which field an item keeps parts in
        # varies by type, so reading one field per type and ignoring the rest
        # leaves the same hole that `output` was, one key over.
        walked = {container.field for container in containers}
        for key, value in item.items():
            if key in walked or not isinstance(value, list):
                continue
            if any(isinstance(part, dict) and "type" in part for part in value):
                return f"{kind} `{key}` is not a field the sandbox proxy walks"
        for container in containers:
            reason = self._container_refusal(kind, container, item)
            if reason is not None:
                return reason
        return None

    def _container_refusal(
        self, kind: str, container: Container, item: dict
    ) -> str | None:
        parts = item.get(container.field)
        if parts is None:
            if container.optional:
                return None
            return f"{kind} `{container.field}` must be text or an array"
        # Both shapes are measured -- a tool result is a bare string more
        # often than an array -- and a string names nothing.
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
