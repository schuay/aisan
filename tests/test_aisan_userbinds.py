# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""User bind specs: four keys, validated, appended, guarded.

A user file is the first bind source in the system that is not audited code
plus a snapshot, so the tests here are mostly refusals: a typo'd key, a path
in both lists, and -- the one that matters -- a bind that would re-add a
backend credential the profile exists to keep out. The happy paths are the
optionality defaults (ro optional, rw and overlay mandatory), the two
expansion rules (`~`, and relative-to-the-file), and `path` -- the one key
that grants no mount, and so carries a refusal of its own for an entry no
mount in the file covers.

The Box-level enforcement (a hand-composed caller who never touches `load`
still cannot build the box) is `test_aisan_box.py`'s; this file covers the
loader and its guard agreeing with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisan.egress.anthropic import AnthropicBackend
from aisan.egress.openai_compat import OpenAICompatBackend
from aisan.sandbox import RO, RW, Bind, Overlay
from aisan.userbinds import UserSpec, load


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
    assert load(f).binds == [
        Bind(Path("/refs/a"), RO, optional=True),
        Bind(Path("/scratch/b"), RW),
    ]


def test_an_empty_file_names_nothing(tmp_path):
    """A no-op file is a valid no-op -- refusing it would be demanding
    content the format has no other way to state."""
    assert load(_spec_file(tmp_path, "")) == UserSpec([], ())


def test_tilde_expands_and_relative_resolves_against_the_file(tmp_path, monkeypatch):
    """`~` is the user's shorthand; relative is the CHECKED-IN file's -- a
    spec beside a repo names its neighbours without knowing where the
    checkout lives."""
    home = tmp_path / "home"
    (home / "refs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/refs", "neighbour"]\n')
    assert load(f).binds == [
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


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('ro = ["/a"]\nrw = ["/a"]\n', "both ro and rw"),
        ('overlay = ["/a"]\nro = ["/a"]\n', "both overlay and ro"),
        ('overlay = ["/a"]\nrw = ["/a"]\n', "both overlay and rw"),
    ],
)
def test_a_path_in_two_mount_keys_is_refused(tmp_path, text, match):
    """There is no order between two keys to resolve this with, and picking
    one silently would be a precedence rule nobody asked for. Every pair, not
    just ro/rw: an overlay is the one binding a tool cache tolerates, so a
    file naming it ro as well has stated two incompatible intentions."""
    with pytest.raises(ValueError, match=f"appears in {match}"):
        load(_spec_file(tmp_path, text))


def test_the_same_path_in_two_spellings_is_one_bind(tmp_path, monkeypatch):
    """Dedupe on the EXPANDED path, so `~/a` and its expansion do not reach
    bwrap twice -- and so the both-lists check above compares paths, not
    spellings."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/a", "%s"]\n' % (home / "a"))
    assert load(f).binds == [Bind(home / "a", RO, optional=True)]


def test_order_is_overlays_then_ro_then_rw(tmp_path):
    """One stable order, so the same file always produces the same spec --
    the property that makes a merged profile diffable. Warm caches first,
    then what the box may read, then what it may write: the order v8_job
    writes by hand, since a user file has no way to say where a bind goes."""
    f = _spec_file(tmp_path, 'ro = ["/a", "/b"]\nrw = ["/c"]\noverlay = ["/d"]\n')
    binds = load(f).binds
    assert [b.path for b in binds] == [Path("/d"), Path("/a"), Path("/b"), Path("/c")]
    assert isinstance(binds[0], Overlay)


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
    assert load(f, egress=(AnthropicBackend(credentials=creds),)).binds


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
    assert load(f, egress=(_Minted(),)).binds  # even ~ is fine: nothing to expose


# --- overlay and path ---------------------------------------------------------


def test_an_overlay_is_mandatory_and_keeps_its_own_bind_type(tmp_path):
    """The third mount mode, which nothing else in the format can express: a
    ro bind of a shared tool cache fails on its lock file and an absent one
    hangs (`Overlay`'s docstring measures both). Mandatory for that reason --
    a skipped overlay is a hang, which is the worst way to learn about a bad
    bind."""
    f = _spec_file(tmp_path, 'overlay = ["/cache/store"]\n')
    assert load(f).binds == [Overlay(Path("/cache/store"))]


def test_path_entries_are_returned_separately_from_the_binds(tmp_path):
    """`path` grants no mount, so it does not become one. It travels beside
    the binds because it reaches the spec through the other combinator."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools"]\n')
    spec = load(f)
    assert spec.binds == [Bind(Path("/tools"), RO, optional=True)]
    assert spec.path == (Path("/tools"),)


@pytest.mark.parametrize(
    "mount", ['ro = ["/tools"]', 'rw = ["/tools"]', 'overlay = ["/tools"]']
)
def test_a_path_entry_may_be_covered_by_any_mount_key(tmp_path, mount):
    """Coverage is about the box being able to resolve the directory, which
    all three mount modes provide. A descendant counts: binding a tree grants
    its bin dir too."""
    f = _spec_file(tmp_path, f'{mount}\npath = ["/tools/bin"]\n')
    assert load(f).path == (Path("/tools/bin"),)


def test_a_path_entry_no_mount_covers_is_refused(tmp_path):
    """The rule that keeps `path` from being an env key in disguise: it can
    only complete a grant this same file made, never make one. Without the
    check it would be a way to point the box's PATH at a directory the box
    does not have, which resolves nothing and reads like a working config."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/elsewhere"]\n')
    with pytest.raises(ValueError, match="not covered by any ro, rw or overlay"):
        load(f)


def test_a_sibling_prefix_does_not_cover_a_path_entry(tmp_path):
    """Coverage is containment, not a string prefix: /toolsx is not inside
    /tools, and comparing spellings would say it was."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/toolsx"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f)
