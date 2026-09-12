# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Apply Codex-specific policy to the OpenAI Responses transport.

Codex CLI 0.147.0 used ``POST {base_url}/responses`` for interactive and
headless turns, local tool calls, and context compaction. A bare loopback URL
keeps the real upstream path prefix on the host.

Codex declares local ``function`` and ``custom`` tools, sometimes grouped in a
``namespace``. GPT-5.6 also sends declarations through ``additional_tools``.
The policy walks these forms recursively and rejects server-side tools such as
web search and code interpreter.

Input items fall into three groups: self-contained tagged values, containers of
content parts, and tool envelopes. Each content part has an exact field set.
Image and audio URLs must use the inline ``data:`` scheme, and payload-like fields
outside inspected positions are refused.

Forwarded fields also receive value checks. For example, ``tool_choice`` accepts
only the observed string form because its object variants can select hosted
capabilities. ``client_metadata`` is limited to Codex's identifier fields.

The OpenAI-compatible proxy handles forwarding, credential replacement, limits,
errors, and streaming. This module supplies the Responses route, body policy,
and host-generated protocol headers.
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
# Item types with no external references. Function and custom calls store their
# input in strings; compaction stores an opaque blob and ID. None contain parts.
#
# Boxed clients compact locally, but a session recorded outside the box may
# contain a ``compaction`` item. Permit replaying that summary. Continue to block
# ``compaction_trigger``, which asks the upstream to perform new work.
TAG_SETTLED_INPUT_TYPES = frozenset({"compaction", "custom_tool_call", "function_call"})

# Exact field sets for content parts. A type alone doesn't prevent an extra URL
# field from naming external data. Recorded sessions matched these sets;
# ``encrypted_content`` comes from the protocol definition because no
# ``agent_message`` was captured.
PART_KEYS = MappingProxyType(
    {
        "input_text": frozenset({"type", "text"}),
        "output_text": frozenset({"type", "text"}),
        "encrypted_content": frozenset({"type", "encrypted_content"}),
        "input_image": frozenset({"type", "image_url", "detail"}),
        "input_audio": frozenset({"type", "audio_url"}),
        "summary_text": frozenset({"type", "text"}),
        "reasoning_text": frozenset({"type", "text"}),
        "text": frozenset({"type", "text"}),
    }
)

# Images observed from Codex used inline ``data:`` URLs. The box can construct
# requests directly, so reject schemes that ask the upstream to fetch data.
INLINE_URL_KEYS = MappingProxyType(
    {"input_image": "image_url", "input_audio": "audio_url"}
)
INLINE_URL_SCHEME = "data:"

# Fields that carry inline content or metadata. Other fields could name external
# payloads regardless of whether their names end in ``_url``.
INERT_PART_KEYS = frozenset({"type", "text", "detail", "encrypted_content"})


@dataclass(frozen=True)
class Container:
    """Describe a content field and the part types it accepts."""

    field: str
    parts: frozenset[str]
    # Reasoning may omit either optional content field.
    optional: bool = False


# Content types returned by local tools.
TOOL_RESULT_PARTS = frozenset(
    {"input_text", "input_image", "input_audio", "encrypted_content"}
)

# Allowed content fields and part types for each protocol item. URL-bearing
# parts receive separate inline checks below. These unions include the complete
# protocol shapes needed by ``codex review`` and MCP tool results.
CONTAINERS = MappingProxyType(
    {
        "message": (
            Container(
                "content",
                frozenset({"input_text", "output_text", "input_image", "input_audio"}),
            ),
        ),
        "agent_message": (
            Container("content", frozenset({"input_text", "encrypted_content"})),
        ),
        "function_call_output": (Container("output", TOOL_RESULT_PARTS),),
        "custom_tool_call_output": (Container("output", TOOL_RESULT_PARTS),),
        "reasoning": (
            Container("summary", frozenset({"summary_text"}), optional=True),
            Container("content", frozenset({"reasoning_text", "text"}), optional=True),
        ),
    }
)

TOOL_ENVELOPE_INPUT_TYPE = "additional_tools"

# Codex sends "auto"; the other string variants select no hosted capability.
TOOL_CHOICES = frozenset({"auto", "none", "required"})

# Field names that may tell the upstream to fetch a payload. Inside known parts,
# scheme checks allow inline data. Elsewhere, the name itself causes refusal.
PAYLOAD_KEYS = frozenset(
    {"image_url", "file_url", "audio_url", "url", "uri", "file_id"}
)


# Derive this set so every allowed type also receives its corresponding checks.
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
    # Structured fields observed in functions, custom tools, and namespaces.
    structured_tool_keys: frozenset[str] = frozenset({"parameters", "format", "tools"})
    # The base policy checks unknown top-level fields against this set.
    allowed_keys: frozenset[str] = ALLOWED_KEYS

    def refuse(self, body: bytes) -> str | None:
        reason = super().refuse(body)
        if reason is not None:
            return reason

        payload = parse_json_object(body)
        # The API retains data when ``store`` is absent, so require explicit false.
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

        # Object forms of ``tool_choice`` can select hosted capabilities that
        # don't appear in ``tools``. Recorded turns used only string values.
        choice = payload.get("tool_choice")
        if choice is not None and (
            not isinstance(choice, str) or choice not in TOOL_CHOICES
        ):
            return "`tool_choice` may only be one of " + quoted_names(TOOL_CHOICES)

        # Metadata field names change across Codex releases and subagent calls.
        # Restrict values to strings, then scan them for payload references.
        metadata = payload.get("client_metadata")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(isinstance(value, str) for value in metadata.values())
        ):
            return "`client_metadata` must be an object of string fields"

        # Reject payload references in fields whose structure isn't inspected.
        for key, value in payload.items():
            if key in {"input", "tools"}:
                continue
            if named := _names_a_payload(value):
                return f"`{key}` names a payload this policy cannot gate: {named!r}"

        return self._text_refusal(payload.get("text"))

    def _text_refusal(self, text: object) -> str | None:
        if text is None:
            return None
        if (
            not isinstance(text, dict)
            or not text
            or set(text) - {"verbosity", "format"}
        ):
            return "`text` may only contain verbosity and an output format"
        if "verbosity" in text and (
            not isinstance(text["verbosity"], str)
            or text["verbosity"] not in {"low", "medium", "high"}
        ):
            return "`text.verbosity` must be low, medium, or high"
        # Codex 0.153.4 uses strict output schemas for task recaps and
        # ``--output-schema``. The payload scan above still covers the schema.
        if "format" in text:
            output_format = text["format"]
            if (
                not isinstance(output_format, dict)
                or set(output_format) != {"type", "name", "strict", "schema"}
                or output_format["type"] != "json_schema"
                or not isinstance(output_format["name"], str)
                or output_format["strict"] is not True
                or not isinstance(output_format["schema"], dict)
            ):
                return "`text.format` must be a named strict JSON schema"
        return None

    def _input_refusal(self, item: object) -> str | None:
        if not isinstance(item, dict):
            return "every Responses input item must be a JSON object"
        kind = item.get("type")
        if not isinstance(kind, str) or kind not in ALLOWED_INPUT_TYPES:
            return f"Responses input type {kind!r} is not permitted"
        if kind == TOOL_ENVELOPE_INPUT_TYPE:
            if (
                not {"type", "role", "tools"} <= set(item)
                or item["role"] != "developer"
            ):
                return "`additional_tools` must be the measured developer envelope"
            # Identifier names change across releases, so constrain their values
            # to strings instead of pinning the names.
            for key, value in item.items():
                if key != "tools" and not isinstance(value, str):
                    return f"`additional_tools` `{key}` must be a string"
            return self.refuse_tools(item["tools"])
        # Only the additional_tools envelope may declare tools. Other item fields
        # include upstream IDs and metadata that vary across releases.
        if "tools" in item:
            return f"{kind} may not declare tools"

        containers = CONTAINERS.get(kind, ())
        # Reject part arrays outside the content fields defined for this type.
        walked = {container.field for container in containers}
        for key, value in item.items():
            if key in walked:
                continue
            if isinstance(value, list) and any(
                isinstance(part, dict) and "type" in part for part in value
            ):
                return f"{kind} `{key}` is not a field the sandbox proxy walks"
            if named := _names_a_payload(value):
                return (
                    f"{kind} `{key}` names a payload this policy cannot gate: {named!r}"
                )
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
        # Tool results may be either plain strings or arrays of parts.
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
        # Observed part values are strings. Reject nested values under allowed keys.
        for key, value in part.items():
            if not isinstance(value, str):
                return f"{part_kind} `{key}` must be a string"
        url_key = INLINE_URL_KEYS.get(part_kind)
        if url_key is None:
            return None
        url = part.get(url_key, "")
        if url[: len(INLINE_URL_SCHEME)].lower() != INLINE_URL_SCHEME:
            return (
                f"{part_kind} `{url_key}` must carry the payload inline as a"
                f" {INLINE_URL_SCHEME} url: any other scheme is a fetch for the"
                " upstream"
            )
        return None


def _names_a_payload(value: object) -> str | None:
    """Find the first field containing an external payload reference.

    Only string values count. A JSON Schema may describe a property named
    ``url`` with an object, which doesn't ask the upstream to fetch anything.
    Inline data URLs remain allowed.

    Walk iteratively because the box controls nesting depth.
    """
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                if (
                    key in PAYLOAD_KEYS
                    and isinstance(nested, str)
                    and nested[: len(INLINE_URL_SCHEME)].lower() != INLINE_URL_SCHEME
                ):
                    return key
                pending.append(nested)
        elif isinstance(current, list):
            pending.extend(current)
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
