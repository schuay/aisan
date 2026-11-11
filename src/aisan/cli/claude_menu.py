# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Seed the box's /model picker from the models the host credential can use.

Without a claude.ai login, the CLI builds its picker from compiled-in rows plus
``additionalModelOptionsCache``. The host's copy of that cache lags, so seed it
from ``GET /v1/models`` instead: the newest active model of each line.

Brittle: this relies on Claude Code internals, measured on 2.1.322. Rerun
``pytest -m live`` when pinning a new CLI version. If the shape changes, remove
this module and its uses in ``aisan.cli.claude``.
"""

from __future__ import annotations

import logging

from ..egress.anthropic import AnthropicBackend

log = logging.getLogger(__name__)

# Internal CLI config key; rows are {value, label, description}.
# A seeded Fable row replaces the compiled-in one; Opus, Sonnet, Haiku rows merge.
MENU_KEY = "additionalModelOptionsCache"


def picker_rows(models: list[dict]) -> list[dict]:
    """Return one row per model line: its newest active model.

    Entries without ``line``, ``created_at`` or ``lifecycle == "active"`` are
    skipped, so an API shape change shrinks the menu instead of widening it.
    """
    newest: dict[str, dict] = {}
    for m in models:
        line, created = m.get("line"), m.get("created_at")
        if not (
            isinstance(m.get("id"), str)
            and isinstance(line, str)
            and isinstance(created, str)
            and m.get("lifecycle") == "active"
        ):
            continue
        # RFC 3339 timestamps in one zone sort lexically.
        if line not in newest or created > newest[line]["created_at"]:
            newest[line] = m
    return [
        {
            "value": m["id"],
            "label": m.get("display_name") or m["id"],
            "description": m["id"],
        }
        for m in sorted(newest.values(), key=lambda m: m["id"])
    ]


async def fetch_menu(backend: AnthropicBackend) -> list[dict]:
    """Return picker rows for the backend's credential, or none on any failure."""
    try:
        return picker_rows(await backend.list_models())
    except Exception as e:  # The menu is optional; never block a launch on it.
        log.warning("claude: model menu unavailable: %s", e)
        return []
