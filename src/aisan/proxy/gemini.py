# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Body policy for Gemini ``GenerateContent`` requests."""

from __future__ import annotations

from dataclasses import dataclass

from .http import AmbiguousBody, load_json_unambiguous

__all__ = ["CLIENT_TOOL_FIELDS", "FETCH_PART_FIELDS", "BodyPolicy"]


# Both proto3 JSON spellings for content that makes Google fetch a URI. Inline
# base64, text, function, and thought parts don't fetch external data.
FETCH_PART_FIELDS = frozenset({"fileData", "file_data"})


# Function declarations execute inside the box. Accept both proto3 JSON
# spellings because the upstream accepts both; the REST client emits camelCase.
#
# Other fields make the upstream act for the box. Current discovery documents
# include search, URL context, code execution, retrieval, maps, computer use,
# and external API calls. Unknown future fields are refused.
CLIENT_TOOL_FIELDS = frozenset({"functionDeclarations", "function_declarations"})


@dataclass(frozen=True)
class BodyPolicy:
    """Limit capabilities declared by a Gemini request body.

    Gemini tools and content parts can ask Google to fetch, search, or execute
    outside the box. Allow only function declarations, which execute locally,
    and inline content. Apply the same checks to cached-content requests because
    they accept tool declarations too.

    Mount policy, rather than this body check, keeps credentials out of the box.
    """

    client_fields: frozenset[str] = CLIENT_TOOL_FIELDS

    def refuse(self, body: bytes) -> str | None:
        """Return a refusal reason, or ``None`` if the body is allowed.

        Cache deletion may use an empty body. Other requests must contain one
        unambiguous JSON object emitted by the REST SDK.
        """
        if not body:
            return None
        try:
            payload = load_json_unambiguous(body)
        except AmbiguousBody as e:
            return f"the sandbox proxy cannot read this body unambiguously: {e}"
        except ValueError as e:
            return f"request body is not JSON the sandbox proxy can classify: {e}"
        if not isinstance(payload, dict):
            return "request body must be a JSON object"

        reason = self._contents_refusal(payload.get("contents"))
        if reason is not None:
            return reason

        tools = payload.get("tools")
        if tools is None:
            return None
        if not isinstance(tools, list):
            return "`tools` must be an array"
        for tool in tools:
            if not isinstance(tool, dict):
                return "every tool must be a JSON object"
            for field in tool:
                if field not in self.client_fields:
                    return (
                        f"tool field {field!r} executes on the upstream, not in"
                        " the sandbox, and is not permitted"
                    )
        return None

    def _contents_refusal(self, contents: object) -> str | None:
        """Reject content parts that ask Google to fetch a URI."""
        if not isinstance(contents, list):
            return None
        for content in contents:
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict):
                    for field in part:
                        if field in FETCH_PART_FIELDS:
                            return (
                                f"content part field {field!r} makes the upstream"
                                " fetch a URL the box chose, and is not permitted"
                            )
        return None
