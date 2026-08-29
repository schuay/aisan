# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from aisan.cli.claude import seed_state
from aisan.session import mirror_user_memory
from aisan.session_mcp import SessionMCP
from aisan.statedir import prepare_state_dir, read_sealed_text, write_sealed


def _planted_link(state: Path, name: str, victim: Path) -> Path:
    """A symlink the agent leaves at a fixed seed name, aimed at a host file."""
    victim.write_text("SECRET-HOST-CONTENT\n")
    link = state / name
    link.symlink_to(victim)
    return link


def test_prepare_state_dir_is_private(tmp_path):
    state = prepare_state_dir(tmp_path / "cache" / "aisan-claude" / "repo")
    assert stat.S_IMODE(state.lstat().st_mode) == 0o700


def test_prepare_state_dir_tightens_a_stale_world_readable_dir(tmp_path):
    # A dir left 0o755 by an older version must be tightened in place, not
    # trusted and not refused (the path is stable per repo, so refusing would
    # wedge every future session).
    state = tmp_path / "state"
    state.mkdir(mode=0o755)
    assert stat.S_IMODE(state.lstat().st_mode) == 0o755
    prepare_state_dir(state)
    assert stat.S_IMODE(state.lstat().st_mode) == 0o700


def test_prepare_state_dir_refuses_a_symlinked_path(tmp_path):
    (tmp_path / "real").mkdir()
    link = tmp_path / "state"
    link.symlink_to(tmp_path / "real")
    with pytest.raises(RuntimeError, match="not a directory"):
        prepare_state_dir(link)


def test_write_sealed_does_not_follow_a_planted_symlink(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "victim"
    link = _planted_link(state, "seed", victim)
    write_sealed(link, "fresh\n")
    # The host file the link aimed at is untouched...
    assert victim.read_text() == "SECRET-HOST-CONTENT\n"
    # ...and the seed path is now a real 0o600 file with our content.
    assert not link.is_symlink()
    assert link.read_text() == "fresh\n"
    assert stat.S_IMODE(link.lstat().st_mode) == 0o600


def test_write_sealed_truncates_an_existing_regular_file(tmp_path):
    path = tmp_path / "seed"
    path.write_text("old and longer content\n")
    write_sealed(path, "new\n")
    assert path.read_text() == "new\n"


def test_read_sealed_text_treats_a_symlink_as_absent(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "victim"
    link = _planted_link(state, "seed", victim)
    assert read_sealed_text(link) is None  # not the victim's content
    assert read_sealed_text(state / "missing") is None
    real = state / "real"
    real.write_text("data\n")
    assert read_sealed_text(real) == "data\n"


def test_mirror_user_memory_cannot_overwrite_a_symlink_target(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "gitconfig"
    link = _planted_link(state, "CLAUDE.md", victim)
    source = tmp_path / "host_memory"
    source.write_text("real user memory\n")
    mirror_user_memory(source, link)
    assert victim.read_text() == "SECRET-HOST-CONTENT\n"
    assert not link.is_symlink()
    assert link.read_text() == "real user memory\n"


def test_seed_state_cannot_overwrite_a_symlink_target(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "gitconfig"
    _planted_link(state, ".claude.json", victim)
    seed_state(state, tmp_path / "repo")
    # The host file is untouched, and a real config was rebuilt from scratch.
    assert victim.read_text() == "SECRET-HOST-CONTENT\n"
    written = state / ".claude.json"
    assert not written.is_symlink()
    config = json.loads(written.read_text())
    assert config["hasCompletedOnboarding"] is True


def test_session_mcp_write_cannot_overwrite_a_symlink_target(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "gitconfig"
    link = _planted_link(state, "aisan-host-mcp.json", victim)
    mcp = SessionMCP(document={"mcpServers": {}}, commands=(), kind="json")
    mcp.write(link)
    assert victim.read_text() == "SECRET-HOST-CONTENT\n"
    assert not link.is_symlink()
    assert stat.S_IMODE(link.lstat().st_mode) == 0o600
