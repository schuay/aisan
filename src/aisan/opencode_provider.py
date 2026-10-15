# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Resolve one host provider into immutable session routing and model data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .upstream import literal, upstream_origin, validate_upstream

SDK = "@ai-sdk/openai-compatible"


@dataclass(frozen=True)
class CustomModel:
    name: str
    upstream_id: str
    context: int
    output: int
    input: int | None
    tool_call: bool

    def config(self, key: str) -> dict[str, object]:
        limits = {"context": self.context, "output": self.output}
        if self.input is not None:
            limits["input"] = self.input
        result = {
            "name": self.name,
            "limit": limits,
            "tool_call": self.tool_call,
            "modalities": {"input": ["text"], "output": ["text"]},
        }
        if self.upstream_id != key:
            result["id"] = self.upstream_id
        return result


@dataclass(frozen=True)
class SelectedProvider:
    provider: str
    model: str
    custom: CustomModel | None
    upstream: str
    provenance: str
    source: Path
    ignored: tuple[str, ...]

    def description(self, auth: str) -> str:
        lines = [
            f"OpenCode config: {self.source}",
            f"OpenCode model: {self.provider}/{self.model}",
            f"Upstream: {upstream_origin(self.upstream)} ({self.provenance})",
            f"Upstream auth: {auth}",
        ]
        if self.custom:
            lines.append(
                f"Model budgets: context={self.custom.context}, output={self.custom.output}, input={self.custom.input}"
            )
        if self.ignored:
            lines.append(
                "Unsupported host fields (ignored): " + ", ".join(self.ignored)
            )
        return "\n".join(lines)


def _object(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{field}: expected an object")
    return value


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field}: expected a positive integer")
    return value


def _model(value: object, key: str, field: str) -> tuple[CustomModel, list[str]]:
    data = _object(value, field)
    for name in ("provider", "npm"):
        if data.get(name):
            raise ValueError(
                f"{field}.{name}: model transport overrides are unsupported"
            )
    limits = _object(data.get("limit"), f"{field}.limit")
    context = _positive(limits.get("context"), f"{field}.limit.context")
    output = _positive(limits.get("output"), f"{field}.limit.output")
    if output >= context:
        raise ValueError(f"{field}.limit.output: must be less than context")
    input_limit = None
    if "input" in limits:
        input_limit = _positive(limits["input"], f"{field}.limit.input")
        if input_limit > context:
            raise ValueError(f"{field}.limit.input: must not exceed context")
    tool = data.get("tool_call", True)
    if type(tool) is not bool:
        raise ValueError(f"{field}.tool_call: expected a boolean")
    model = CustomModel(
        literal(data.get("name", key), f"{field}.name"),
        literal(data.get("id", key), f"{field}.id"),
        context,
        output,
        input_limit,
        tool,
    )
    ignored = [
        f"{field}.{k}" for k in data if k not in {"name", "id", "limit", "tool_call"}
    ]
    ignored += [
        f"{field}.limit.{k}" for k in limits if k not in {"context", "output", "input"}
    ]
    return model, ignored


def resolve_provider(
    provider: str,
    model: str,
    *,
    source: Path,
    document: dict[str, object],
    catalog: Path,
    upstream: str | None = None,
) -> SelectedProvider:
    literal(provider, "provider ID")
    literal(model, "model ID")
    prefix = f"host OpenCode config {source}"
    native = _object(document.get("providers", {}), f"{prefix}: providers")
    if provider in native:
        raise ValueError(
            f"{prefix}: providers.{provider}: native V2 provider import is unsupported; use legacy provider format"
        )
    providers = _object(document.get("provider", {}), f"{prefix}: provider")
    entry = (
        _object(providers[provider], f"{prefix}: provider.{provider}")
        if provider in providers
        else {}
    )
    field = f"{prefix}: provider.{provider}"
    options = _object(entry.get("options", {}), f"{field}.options")
    models = _object(entry.get("models", {}), f"{field}.models")
    if "npm" in entry and entry["npm"] != SDK:
        raise ValueError(f"{field}.npm: only {SDK} is supported")
    custom = None
    ignored = [f"{field}.{k}" for k in entry if k not in {"npm", "options", "models"}]
    ignored += [f"{field}.options.{k}" for k in options if k != "baseURL"]
    catalog_entry = None
    if "npm" not in entry or (upstream is None and "baseURL" not in options):
        try:
            data = json.loads(catalog.read_text())
        except (OSError, ValueError):
            raise ValueError(
                f"cannot read provider catalog {catalog}; run opencode once on the HOST"
            ) from None
        catalog_entry = data.get(provider) if isinstance(data, dict) else None
    if "npm" in entry:
        if model not in models:
            raise ValueError(
                f"{field}.models: missing selected model {model}; define its limits in the host config"
            )
        custom, model_ignored = _model(models[model], model, f"{field}.models.{model}")
        ignored.extend(model_ignored)
    elif not isinstance(catalog_entry, dict):
        raise ValueError(
            f"{field}: unknown catalog provider; define npm={SDK} and selected model {model}"
        )
    elif models:
        raise ValueError(f"{field}.models: custom model import requires npm={SDK}")
    if "baseURL" in options:
        host_api = validate_upstream(options["baseURL"], f"{field}.options.baseURL")
    else:
        host_api = None
    if upstream is not None:
        api = validate_upstream(upstream)
        provenance = "explicit upstream / AISAN_OPENCODE_UPSTREAM"
    elif host_api is not None:
        api, provenance = host_api, f"{field}.options.baseURL"
    else:
        api = validate_upstream(
            catalog_entry.get("api") if isinstance(catalog_entry, dict) else None,
            "catalog api",
        )
        provenance = f"catalog {catalog}"
    return SelectedProvider(
        provider,
        model,
        custom,
        api,
        provenance.removeprefix(f"{prefix}: "),
        source,
        tuple(sorted(name.removeprefix(f"{prefix}: ") for name in ignored)),
    )
