# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

import pytest

from aisan import hostproc
from aisan import private as private_mod


def test_host_child_ignores_ambient_tmpdir_and_normalizes_paths(tmp_path, monkeypatch):
    private = tmp_path / "private"
    hostile_tmp = tmp_path / "box-writable"
    hostile_tmp.mkdir()
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    monkeypatch.setenv("TMPDIR", str(hostile_tmp))

    original = {
        "KEEP": "yes",
        "PWD": "/agent/repository",
        "TMPDIR": str(hostile_tmp),
        "TMP": "/agent/tmp",
        "TEMP": "/agent/temp",
        "OLDPWD": "/agent/old",
        "INIT_CWD": "/agent/init",
    }
    with hostproc.neutral_child(original) as child:
        assert child.cwd.parent == private_mod.host_child_root()
        assert child.cwd.is_dir()
        assert not child.cwd.is_relative_to(hostile_tmp)
        assert child.env["KEEP"] == "yes"
        for name in ("PWD", "TMPDIR", "TMP", "TEMP"):
            assert child.env[name] == str(child.cwd)
        assert "OLDPWD" not in child.env
        assert "INIT_CWD" not in child.env
        assert list(child.cwd.iterdir()) == []

    assert original["PWD"] == "/agent/repository"
    assert not child.cwd.exists()


def test_private_root_rejects_unsafe_existing_paths(tmp_path, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", private)
    private.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a directory"):
        private_mod.prepare_private_dir(private_mod.host_child_root())

    private.unlink()
    private.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="too permissive"):
        private_mod.prepare_private_dir(private_mod.host_child_root())


def test_a_root_that_holds_no_socket_is_not_held_to_the_socket_budget(
    tmp_path, monkeypatch
):
    root = tmp_path / ("r" * 90)
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", root)
    assert private_mod.prepare_private_dir(private_mod.host_child_root()).is_dir()


def test_default_host_child_environment_is_a_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(private_mod, "_PRIVATE_ROOT", tmp_path / "private")
    monkeypatch.setenv("OLDPWD", "/agent/old")
    monkeypatch.setenv("INIT_CWD", "/agent/init")
    monkeypatch.setenv("UNCHANGED", "kept")

    with hostproc.neutral_child() as child:
        assert child.env["UNCHANGED"] == "kept"
        assert "OLDPWD" not in child.env
        assert "INIT_CWD" not in child.env
        assert os.environ["OLDPWD"] == "/agent/old"
