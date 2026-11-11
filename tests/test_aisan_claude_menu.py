# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import dataclasses
import fcntl
import json
import os
import pty
import re
import select
import struct
import termios
import time
from pathlib import Path

import pytest
from aiohttp import web

from aisan.box import Box
from aisan.cli.claude import claude_command, seed_settings, seed_state
from aisan.cli.claude_menu import MENU_KEY, fetch_menu, picker_rows
from aisan.egress.anthropic import AnthropicBackend
from aisan.presets.claude_code import claude_code, claude_code_binary

# The shape of GET /v1/models as served on 2026-11-11, trimmed.
MODELS = [
    {
        "id": "claude-mythos-5-2",
        "display_name": "Claude Mythos 5.2",
        "line": "mythos",
        "created_at": "2026-11-09T18:30:00Z",
        "lifecycle": "active",
    },
    {
        "id": "claude-fable-5-2",
        "display_name": "Claude Fable 5.2",
        "line": "fable",
        "created_at": "2026-11-09T18:30:00Z",
        "lifecycle": "active",
    },
    {
        "id": "claude-opus-5-5",
        "display_name": "Claude Opus 5.5",
        "line": "opus",
        "created_at": "2026-09-21T16:24:00Z",
        "lifecycle": "active",
    },
    {
        "id": "claude-fable-5-1",
        "display_name": "Claude Fable 5.1",
        "line": "fable",
        "created_at": "2026-08-28T00:00:00Z",
        "lifecycle": "active",
    },
    {
        "id": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5",
        "line": "sonnet",
        "created_at": "2025-09-29T00:00:00Z",
        "lifecycle": "deprecated",
    },
]


def _row(model_id: str, label: str) -> dict:
    return {"value": model_id, "label": label, "description": model_id}


def test_picker_rows_keep_the_newest_active_model_of_each_line():
    assert picker_rows(MODELS) == [
        _row("claude-fable-5-2", "Claude Fable 5.2"),
        _row("claude-mythos-5-2", "Claude Mythos 5.2"),
        _row("claude-opus-5-5", "Claude Opus 5.5"),
    ]


@pytest.mark.parametrize("missing", ["id", "line", "created_at", "lifecycle"])
def test_picker_rows_skip_an_entry_missing_a_field(missing):
    entry = {k: v for k, v in MODELS[0].items() if k != missing}
    assert picker_rows([entry]) == []


def test_seed_state_replaces_the_menu_and_clears_it_without_one(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    host = tmp_path / "absent.json"
    rows = picker_rows(MODELS)

    seed_state(state, host, menu=rows)
    assert json.loads((state / ".claude.json").read_text())[MENU_KEY] == rows

    seed_state(state, host)
    assert MENU_KEY not in json.loads((state / ".claude.json").read_text())


async def _upstream(handler) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}", runner


async def test_fetch_menu_asks_the_upstream_with_the_relay_credential():
    seen = []

    async def handler(request: web.Request) -> web.Response:
        seen.append((request.path, request.headers.get("x-api-key")))
        return web.json_response({"data": MODELS, "has_more": False})

    up, runner = await _upstream(handler)
    try:
        rows = await fetch_menu(AnthropicBackend(api_key="k", upstream=up))
    finally:
        await runner.cleanup()
    assert seen == [("/v1/models", "k")]
    assert rows == picker_rows(MODELS)


@pytest.mark.parametrize(
    "response",
    [
        web.json_response({"error": "nope"}, status=401),
        web.json_response(["not", "an", "object"]),
        web.Response(text="not json"),
    ],
)
async def test_fetch_menu_fails_closed(response):
    async def handler(request: web.Request) -> web.Response:
        return response

    up, runner = await _upstream(handler)
    try:
        assert await fetch_menu(AnthropicBackend(api_key="k", upstream=up)) == []
    finally:
        await runner.cleanup()


def _picker(argv: list[str], env: dict[str, str], cwd: Path) -> str:
    """Open /model in the TUI and return the picker's text, ANSI stripped."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.execvpe(argv[0], argv, env)  # noqa: S606
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))
    buf = b""

    def text(start: int = 0) -> str:
        plain = re.sub(
            rb"\x1b\[[0-9;?<>]*[A-Za-z~]|\x1b\][^\x07]*\x07|\x1b.", b" ", buf[start:]
        )
        return re.sub(" +", " ", plain.decode("utf-8", "replace"))

    def send(keys: bytes, until: tuple[str, ...], seconds: float = 30) -> str:
        """Send ``keys``, then read until new output shows one of ``until``."""
        nonlocal buf
        start = len(buf)
        os.write(fd, keys)
        end = time.monotonic() + seconds
        while time.monotonic() < end and not any(u in text(start) for u in until):
            if select.select([fd], [], [], 0.2)[0]:
                try:
                    buf += os.read(fd, 65536)
                except OSError:
                    break
        return text(start)

    try:
        send(b"", ("shift+tab",), 60)
        send(b"/model", ("Set the AI model",))
        picker = send(b"\r", ("Esc to cancel",))
        return picker[picker.rfind("Select model") :]
    finally:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        os.close(fd)


@pytest.mark.live
@pytest.mark.skipif(claude_code_binary() is None, reason="claude is not installed")
async def test_real_claude_offers_the_seeded_menu(tmp_path):
    async def handler(request: web.Request) -> web.Response:
        if request.path == "/v1/models":
            return web.json_response({"data": MODELS, "has_more": False})
        return web.json_response({})

    up, runner = await _upstream(handler)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    try:
        backend = AnthropicBackend(api_key="k", upstream=up)
        seed_state(state, tmp_path / "absent.json", menu=await fetch_menu(backend))
        seed_settings(state)
        config = json.loads((state / ".claude.json").read_text())
        config["projects"] = {str(worktree): {"hasTrustDialogAccepted": True}}
        (state / ".claude.json").write_text(json.dumps(config))
        base = claude_code(worktree, state=state, egress=(backend,))
        spec = dataclasses.replace(
            base, limits=dataclasses.replace(base.limits, use_cgroup=False)
        )
        command = claude_command(
            mcp_config=None, settings=None, payload=["--model", "opus"]
        )
        async with Box(spec, box_id="claude-menu") as box:
            picker = await asyncio.to_thread(
                _picker,
                box.command(command),
                {**os.environ, **box.env},
                worktree,
            )
    finally:
        await runner.cleanup()

    # Labels are redrawn incrementally; the ids in the description column are not.
    # A seeded Fable row replaces the compiled-in one.
    assert "claude-fable-5-2" in picker
    assert "Fable 5.1" not in picker
    # Seeded rows are offered even though the relay is not a first-party URL.
    assert "claude-mythos-5-2" in picker
