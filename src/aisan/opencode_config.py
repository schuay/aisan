# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Read the selected host OpenCode configuration without merging sources."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def config_file() -> Path:
    configured = os.environ.get("OPENCODE_CONFIG")
    if configured:
        return Path(configured)
    root = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
    base = root / "opencode"
    json_path = base / "opencode.json"
    return json_path if json_path.exists() else base / "opencode.jsonc"


def read_config(path: Path, *, strict: bool = False) -> dict[str, object]:
    if not path.exists():
        if strict and os.environ.get("OPENCODE_CONFIG"):
            raise ValueError(f"host OpenCode config {path} does not exist")
        return {}
    try:
        text = _strip_jsonc(path.read_text(), strict=strict)
        data = (
            json.loads(
                text,
                object_pairs_hook=_unique,
                parse_constant=_constant,
                parse_float=_finite,
            )
            if strict
            else json.loads(text)
        )
    except (ValueError, UnicodeError):
        raise ValueError(f"host OpenCode config {path} is not valid JSONC") from None
    if not isinstance(data, dict):
        raise TypeError(f"host OpenCode config {path} must contain an object")
    return data


def select_model(
    document: dict[str, object],
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    if provider and model:
        return provider, model
    preferred = document.get("model")
    if preferred is not None:
        if not isinstance(preferred, str) or "/" not in preferred:
            raise ValueError("host OpenCode model must be provider/model")
        host_provider, host_model = preferred.split("/", 1)
        if not host_provider or not host_model:
            raise ValueError("host OpenCode model must be provider/model")
        if provider is None or provider == host_provider:
            return provider or host_provider, model or host_model

    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    path = root / "opencode" / "model.json"
    state = read_config(path, strict=True) if path.exists() else {}
    recent = state.get("recent", [])
    if not isinstance(recent, list):
        raise TypeError(f"{path}: recent must be a list")
    for entry in recent:
        if not isinstance(entry, dict):
            raise TypeError(f"{path}: recent entries must be objects")
        host_provider, host_model = entry.get("providerID"), entry.get("modelID")
        if not isinstance(host_provider, str) or not host_provider:
            raise ValueError(f"{path}: recent providerID must be a nonempty string")
        if not isinstance(host_model, str) or not host_model:
            raise ValueError(f"{path}: recent modelID must be a nonempty string")
        if provider is None or provider == host_provider:
            return provider or host_provider, model or host_model
    raise ValueError(
        "no preferred OpenCode model; set model in host config or pass --provider and --model"
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite number")
    return number


def _constant(value):
    raise ValueError("nonfinite number")


def _strip_jsonc(text: str, *, strict: bool = False) -> str:
    """Remove JSONC comments and trailing commas without touching strings."""
    out: list[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            index += 1
            continue
        if text.startswith("//", index):
            index = text.find("\n", index)
            if index < 0:
                break
            out.append("\n")
            index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                if strict:
                    raise ValueError("unterminated comment")
                break
            if strict:
                out.append(" ")
            out.extend("\n" for c in text[index : end + 2] if c == "\n")
            index = end + 2
            continue
        out.append(char)
        index += 1
    return _strip_trailing_commas("".join(out))


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            out.append(char)
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "]}":
                continue
        out.append(char)
    return "".join(out)
