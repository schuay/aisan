# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""User bind specs: two keys, validated, appended, guarded.

A user file is the first bind source in the system that is not audited code
plus a snapshot, so the tests here are mostly refusals: a typo'd key, a path
in both lists, and -- the one that matters -- a bind that would re-add a
backend credential the profile exists to keep out. The happy paths are the
two optionality defaults (ro optional, rw mandatory) and the two expansion
rules (`~`, and relative-to-the-file).

The Box-level enforcement (a hand-composed caller who never touches `load`
still cannot build the box) is `test_aisan_box.py`'s; this file covers the
loader and its guard agreeing with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisan.egress.anthropic import AnthropicBackend
from aisan.egress.openai_compat import OpenAICompatBackend
from aisan.sandbox import RO, RW, Bind
from aisan.userbinds import load


def _spec_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "binds.toml"
    path.write_text(text)
    return path


def test_ro_binds_are_optional_and_rw_binds_are_mandatory(tmp_path):
    """The two measured defaults, as the two Bind fields they become.

    A ro reference path that has vanished is a bind to skip; a named WRITABLE
    that silently vanishes is a box quietly missing what the caller asked
    for. No per-path key: the file stays obviously weaker than the Python it
    accompanies."""
    f = _spec_file(tmp_path, 'ro = ["/refs/a"]\nrw = ["/scratch/b"]\n')
    assert load(f) == [
        Bind(Path("/refs/a"), RO, optional=True),
        Bind(Path("/scratch/b"), RW),
    ]


def test_an_empty_file_names_nothing(tmp_path):
    """A no-op file is a valid no-op -- refusing it would be demanding
    content the format has no other way to state."""
    assert load(_spec_file(tmp_path, "")) == []


def test_tilde_expands_and_relative_resolves_against_the_file(tmp_path, monkeypatch):
    """`~` is the user's shorthand; relative is the CHECKED-IN file's -- a
    spec beside a repo names its neighbours without knowing where the
    checkout lives."""
    home = tmp_path / "home"
    (home / "refs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/refs", "neighbour"]\n')
    assert load(f) == [
        Bind(home / "refs", RO, optional=True),
        Bind(tmp_path / "neighbour", RO, optional=True),
    ]


def test_unknown_keys_are_refused_not_ignored(tmp_path):
    """`wr = [...]` is a typo somebody believes is in effect. A silently
    ignored key in a confinement config is the wrong direction to err in."""
    f = _spec_file(tmp_path, 'ro = ["/a"]\nwr = ["/b"]\n')
    with pytest.raises(ValueError, match=r"unknown key.*'wr'"):
        load(f)


@pytest.mark.parametrize(
    ("value", "why"),
    [
        ('ro = "/a"', "not an array"),
        ("ro = [1]", "not strings"),
        ('ro = [""]', "empty string"),
        ('ro = ["  "]', "whitespace only"),
    ],
)
def test_malformed_entries_are_refused(tmp_path, value, why):
    with pytest.raises(ValueError, match="must be an array"):
        load(_spec_file(tmp_path, value))


def test_a_path_in_both_lists_is_refused(tmp_path):
    """There is no order between the two keys to resolve this with, and
    picking one silently would be a precedence rule nobody asked for."""
    f = _spec_file(tmp_path, 'ro = ["/a"]\nrw = ["/a"]\n')
    with pytest.raises(ValueError, match="appears in both ro and rw"):
        load(f)


def test_the_same_path_in_two_spellings_is_one_bind(tmp_path, monkeypatch):
    """Dedupe on the EXPANDED path, so `~/a` and its expansion do not reach
    bwrap twice -- and so the both-lists check above compares paths, not
    spellings."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/a", "%s"]\n' % (home / "a"))
    assert load(f) == [Bind(home / "a", RO, optional=True)]


def test_order_is_the_files_ro_then_rw(tmp_path):
    """One stable order, so the same file always produces the same spec --
    the property that makes a merged profile diffable."""
    f = _spec_file(tmp_path, 'ro = ["/a", "/b"]\nrw = ["/c"]\n')
    assert [b.path for b in load(f)] == [Path("/a"), Path("/b"), Path("/c")]


# --- the credential guard -----------------------------------------------------


def test_a_bind_containing_a_backend_credential_is_refused(tmp_path):
    """The reason `load` takes the egress tuple. The credential-absence claim
    is subtraction, and a user naming `~` (or any ancestor of the credential
    file) re-adds exactly what the profile exists to keep out. Containment,
    not equality: the parent exposes the file just as surely as the file."""
    creds = tmp_path / "creds" / "k.json"
    f = _spec_file(tmp_path, f'ro = ["{tmp_path / "creds"}"]\n')
    with pytest.raises(ValueError) as e:
        load(f, egress=(AnthropicBackend(credentials=creds),))
    assert "anthropic" in str(e.value)
    assert str(creds) in str(e.value)
    assert str(f) in str(e.value)  # names the file, not just the bind


def test_the_guard_covers_every_backend_that_declares_a_file(tmp_path):
    """auth.json carries EVERY provider's key, so exposing it is exposure
    even for a provider the box does not use -- asserted with the other
    file-backed backend, and its name in the refusal."""
    creds = tmp_path / "local" / "share" / "opencode" / "auth.json"
    f = _spec_file(tmp_path, f'rw = ["{tmp_path / "local"}"]\n')
    with pytest.raises(ValueError, match="openai-zai-coding-plan"):
        load(
            f,
            egress=(
                OpenAICompatBackend(
                    provider="zai-coding-plan",
                    upstream="https://x",
                    credentials=creds,
                ),
            ),
        )


def test_a_sibling_of_the_credential_is_not_exposure(tmp_path):
    """The negative that makes the guard mean something: containment is the
    rule, and a path BESIDE the credential's directory contains nothing."""
    creds = tmp_path / "creds" / "k.json"
    f = _spec_file(tmp_path, f'ro = ["{tmp_path / "elsewhere"}"]\n')
    assert load(f, egress=(AnthropicBackend(credentials=creds),))


def test_a_backend_without_a_file_cannot_be_exposed(tmp_path):
    """Minted-in-memory backends declare no paths (the default), so the guard
    is a no-op for them -- which is why it is a tuple on the backend rather
    than a universal 'the credential' that would have to be faked."""
    from aisan.egress.base import Backend

    class _Minted(Backend):
        name = "minted"
        port = 8790

        async def serve(self, runtime_dir):  # pragma: no cover - shape only
            yield

    f = _spec_file(tmp_path, f'ro = ["{Path.home()}"]\n')
    assert load(f, egress=(_Minted(),))  # even ~ is fine: nothing to expose
