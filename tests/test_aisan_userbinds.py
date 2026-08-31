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
    assert load(f, egress=()).binds == [
        Bind(Path("/refs/a"), RO, optional=True),
        Bind(Path("/scratch/b"), RW),
    ]


def test_an_empty_file_names_nothing(tmp_path):
    """A no-op file is a valid no-op -- refusing it would be demanding
    content the format has no other way to state."""
    assert load(_spec_file(tmp_path, ""), egress=()) == UserSpec([], ())


def test_tilde_expands_and_relative_resolves_against_the_file(tmp_path, monkeypatch):
    """`~` is the user's shorthand; relative is the CHECKED-IN file's -- a
    spec beside a repo names its neighbours without knowing where the
    checkout lives."""
    home = tmp_path / "home"
    (home / "refs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/refs", "neighbour"]\n')
    assert load(f, egress=()).binds == [
        Bind(home / "refs", RO, optional=True),
        Bind(tmp_path / "neighbour", RO, optional=True),
    ]


def test_unknown_keys_are_refused_not_ignored(tmp_path):
    """`wr = [...]` is a typo somebody believes is in effect. A silently
    ignored key in a confinement config is the wrong direction to err in."""
    f = _spec_file(tmp_path, 'ro = ["/a"]\nwr = ["/b"]\n')
    with pytest.raises(ValueError, match=r"unknown key.*'wr'"):
        load(f, egress=())


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
        load(_spec_file(tmp_path, value), egress=())


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
        load(_spec_file(tmp_path, text), egress=())


@pytest.mark.parametrize(
    "text",
    [
        'ro = ["/a/b"]\nrw = ["/a"]\n',  # ro nested under a later, broader rw
        'overlay = ["/a/cache"]\nrw = ["/a"]\n',  # overlay defeated by rw
        'overlay = ["/a/cache"]\nro = ["/a"]\n',  # overlay defeated by ro
        'rw = ["/a"]\nro = ["/a/../a/b"]\n',  # the `..` route around equality
    ],
)
def test_a_mount_nested_under_a_later_broader_one_is_refused(tmp_path, text):
    """The keys emit overlay, then ro, then rw, and later wins -- so a narrow
    ro or overlay under a broad rw is silently upgraded to writable (and an
    overlay's copy-up is defeated, sending writes to the host tree). Exact
    equality has its own refusal; this is the nesting the fixed emission order
    introduced. `..` is normalised away so it cannot dodge the check."""
    with pytest.raises(ValueError, match="remove the nesting"):
        load(_spec_file(tmp_path, text), egress=())


def test_same_mode_nesting_is_allowed(tmp_path):
    """A path under another of the SAME key is not a mode change: the effective
    mode is what the author wrote either way, so it is redundant, not a silent
    upgrade, and refusing it would be stricter than the hazard warrants."""
    f = _spec_file(tmp_path, 'rw = ["/a/b", "/a"]\n')
    assert [b.path for b in load(f, egress=()).binds] == [Path("/a/b"), Path("/a")]


def test_the_includer_may_still_shadow_an_included_mount(tmp_path):
    """The nesting refusal is scoped to one file's own keys: an outer rw over
    an included ro at the same path is the documented later-wins, not a bug."""
    _named_spec(tmp_path / "base.toml", 'ro = ["/a/b"]\n')
    outer = _named_spec(
        tmp_path / "outer.toml", 'include = ["base.toml"]\nrw = ["/a"]\n'
    )
    assert [
        (b.path, getattr(b, "mode", None)) for b in load(outer, egress=()).binds
    ] == [
        (Path("/a/b"), RO),
        (Path("/a"), RW),
    ]


def test_the_same_path_in_two_spellings_is_one_bind(tmp_path, monkeypatch):
    """Dedupe on the EXPANDED path, so `~/a` and its expansion do not reach
    bwrap twice -- and so the both-lists check above compares paths, not
    spellings."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/a", "%s"]\n' % (home / "a"))
    assert load(f, egress=()).binds == [Bind(home / "a", RO, optional=True)]


def test_order_is_overlays_then_ro_then_rw(tmp_path):
    """One stable order, so the same file always produces the same spec --
    the property that makes a merged profile diffable. Warm caches first,
    then what the box may read, then what it may write: the order v8_job
    writes by hand, since a user file has no way to say where a bind goes."""
    f = _spec_file(tmp_path, 'ro = ["/a", "/b"]\nrw = ["/c"]\noverlay = ["/d"]\n')
    binds = load(f, egress=()).binds
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


def test_egress_is_required_so_the_guard_is_never_off_by_default(tmp_path):
    """The guard's only input has no default. A default of `()` would be a
    guard silently switched off for any caller that forgot it, which is the one
    failure it exists to prevent, so omitting egress is a TypeError rather than
    a load whose credential check saw no backends and passed everything."""
    f = _spec_file(tmp_path, 'ro = ["/a"]\n')
    with pytest.raises(TypeError):
        load(f)  # type: ignore[call-arg]


# --- overlay and path ---------------------------------------------------------


def test_an_overlay_is_mandatory_and_keeps_its_own_bind_type(tmp_path):
    """The third mount mode, which nothing else in the format can express: a
    ro bind of a shared tool cache fails on its lock file and an absent one
    hangs (`Overlay`'s docstring measures both). Mandatory for that reason --
    a skipped overlay is a hang, which is the worst way to learn about a bad
    bind."""
    f = _spec_file(tmp_path, 'overlay = ["/cache/store"]\n')
    assert load(f, egress=()).binds == [Overlay(Path("/cache/store"))]


def test_path_entries_are_returned_separately_from_the_binds(tmp_path):
    """`path` grants no mount, so it does not become one. It travels beside
    the binds because it reaches the spec through the other combinator."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools"]\n')
    spec = load(f, egress=())
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
    assert load(f, egress=()).path == (Path("/tools/bin"),)


def test_a_path_entry_no_mount_covers_is_refused(tmp_path):
    """The rule that keeps `path` from being an env key in disguise: it can
    only complete a grant this same file made, never make one. Without the
    check it would be a way to point the box's PATH at a directory the box
    does not have, which resolves nothing and reads like a working config."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/elsewhere"]\n')
    with pytest.raises(ValueError, match="not covered by any ro, rw or overlay"):
        load(f, egress=())


def test_a_sibling_prefix_does_not_cover_a_path_entry(tmp_path):
    """Coverage is containment, not a string prefix: /toolsx is not inside
    /tools, and comparing spellings would say it was."""
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/toolsx"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())


def _named_spec(path: Path, body: str) -> Path:
    """A spec file at a path the caller chose.

    `_spec_file` above always writes binds.toml; an include tree needs several
    files that can name each other.
    """
    path.write_text(body)
    return path


def test_include_expands_in_place_and_the_includer_shadows(tmp_path):
    # The point of the key: composition order is a fact about the files, not
    # about the operator's shell history.
    (tmp_path / "base").mkdir()
    _named_spec(tmp_path / "base" / "tools.toml", 'ro = ["~/tools"]\n')
    outer = _named_spec(
        tmp_path / "outer.toml",
        'include = ["base/tools.toml"]\nrw = ["~/tools"]\n',
    )

    spec = load(outer, egress=())

    # Included first, so the rw the outer file names lands later and wins.
    assert [(b.path, getattr(b, "mode", None)) for b in spec.binds] == [
        (Path.home() / "tools", RO),
        (Path.home() / "tools", RW),
    ]


def test_an_include_resolves_against_the_including_file(tmp_path):
    # Same rule the mount keys follow, which is what lets a directory of specs
    # name each other by basename.
    (tmp_path / "d").mkdir()
    _named_spec(tmp_path / "d" / "inner.toml", 'ro = ["~/inner"]\n')
    outer = _named_spec(tmp_path / "d" / "outer.toml", 'include = ["inner.toml"]\n')

    assert [b.path for b in load(outer, egress=()).binds] == [Path.home() / "inner"]


def test_a_diamond_mounts_once(tmp_path):
    _named_spec(tmp_path / "leaf.toml", 'ro = ["~/leaf"]\n')
    _named_spec(tmp_path / "mid.toml", 'include = ["leaf.toml"]\n')
    top = _named_spec(tmp_path / "top.toml", 'include = ["leaf.toml", "mid.toml"]\n')

    assert [b.path for b in load(top, egress=()).binds] == [Path.home() / "leaf"]


def test_an_include_cycle_is_refused_by_name(tmp_path):
    a = _named_spec(tmp_path / "a.toml", 'include = ["b.toml"]\n')
    _named_spec(tmp_path / "b.toml", 'include = ["a.toml"]\n')

    with pytest.raises(ValueError, match="include cycle"):
        load(a, egress=())


def test_a_file_including_itself_is_a_cycle_not_a_repeat(tmp_path):
    a = _named_spec(tmp_path / "self.toml", 'include = ["self.toml"]\n')

    with pytest.raises(ValueError, match="include cycle"):
        load(a, egress=())


def test_a_missing_include_names_the_file_that_asked_for_it(tmp_path):
    outer = _named_spec(tmp_path / "outer.toml", 'include = ["gone.toml"]\n')

    with pytest.raises(ValueError, match=r"outer\.toml: include .*gone\.toml"):
        load(outer, egress=())


def test_a_path_entry_may_rest_on_an_included_mount(tmp_path):
    # The coverage check counts what the file chose to include: both lines are
    # still in front of the reader, one naming the file and one the directory.
    _named_spec(tmp_path / "tools.toml", 'ro = ["~/tools"]\n')
    outer = _named_spec(
        tmp_path / "outer.toml",
        'include = ["tools.toml"]\npath = ["~/tools/bin"]\n',
    )

    spec = load(outer, egress=())

    assert spec.path == (Path.home() / "tools" / "bin",)


def test_the_credential_guard_reaches_inside_an_include(tmp_path):
    # The guard is the reason `load` takes the egress tuple, and an include is
    # a second way for an unreviewed file to name the credential. The refusal
    # names the file that wrote the line, not the one that included it.
    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    _named_spec(tmp_path / "inner.toml", f'ro = ["{creds.parent}"]\n')
    outer = _named_spec(tmp_path / "outer.toml", 'include = ["inner.toml"]\n')

    with pytest.raises(ValueError, match=r"inner\.toml: .* would expose"):
        load(outer, egress=(AnthropicBackend(credentials=creds),))


def test_the_credential_guard_looks_inside_the_store_too(tmp_path):
    # Overlap, not one-way containment. A credential store is a directory whose
    # point is the token inside it, so a file naming that token names the
    # credential -- and the containment test that only asked "does this bind
    # hold the store" answered no and let it through.
    store = tmp_path / "chrome_infra"
    store.mkdir()
    token = store / "luci_context"
    token.write_text("{}")
    spec = _named_spec(tmp_path / "user.toml", f'ro = ["{token}"]\n')

    with pytest.raises(ValueError, match="would expose"):
        load(spec, egress=(AnthropicBackend(credentials=store),))


def test_a_malformed_include_names_the_inner_file(tmp_path):
    _named_spec(tmp_path / "inner.toml", 'wr = ["~/typo"]\n')
    outer = _named_spec(tmp_path / "outer.toml", 'include = ["inner.toml"]\n')

    with pytest.raises(ValueError, match=r"inner\.toml: unknown key"):
        load(outer, egress=())


def test_a_path_entry_may_not_smuggle_a_second_dir_with_a_colon(tmp_path):
    # L1: os.pathsep in one entry becomes two PATH entries once joined, and the
    # second (/tmp) rides past the coverage check that inspected one string.
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools/bin:/tmp"]\n')
    with pytest.raises(ValueError, match="contains"):
        load(f, egress=())


def test_a_path_entry_cannot_dotdot_escape_its_covering_mount(tmp_path):
    # L1: `/tools/bin/../../../etc` is is_relative_to("/tools") by components but
    # resolves to /etc, outside anything the file mounts. Normalising the
    # coverage check closes that -- the "cannot widen by a byte" invariant holds.
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools/bin/../../../etc"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())


def test_an_unexpandable_home_is_a_named_valueerror_not_a_runtimeerror(tmp_path):
    # L10: ~nosuchuser raises RuntimeError from expanduser, which bypasses the
    # "ValueError naming the file" contract the loader promises its caller.
    f = _spec_file(tmp_path, 'ro = ["~nosuchuser_aisan_xyz/x"]\n')
    with pytest.raises(ValueError, match="cannot be expanded"):
        load(f, egress=())


def test_with_path_prefix_refuses_a_relative_or_colon_bearing_dir(tmp_path):
    # W10: the combinator claimed a PATH prefix names box directories but
    # enforced nothing -- a relative dir resolves against the box cwd and a
    # colon smuggles a second entry. Both are refused now, at the combinator, so
    # a direct caller cannot break the invariant userbinds relies on.
    from aisan.spec import BoxSpec

    base = BoxSpec(
        root=tmp_path,
        binds=(),
        tmpfs=(),
        env=(("PATH", "/usr/bin"),),
        egress=(),
        unshare_net=True,
    )
    with pytest.raises(ValueError, match="not absolute"):
        base.with_path_prefix((Path("relative/dir"),))
    with pytest.raises(ValueError, match="smuggle"):
        base.with_path_prefix((Path("/a:/b"),))
    # A well-formed absolute dir still prepends.
    out = base.with_path_prefix((Path("/opt/tool/bin"),))
    assert dict(out.env)["PATH"].startswith("/opt/tool/bin:")


def test_a_path_prefix_already_present_is_not_repeated(tmp_path):
    """Naming a grant a preset already applied used to render
    `<tree>:<tree>:/usr/bin` -- a second entry that resolves nothing, in the
    artifact whose whole job is to be read closely. First occurrence wins, which
    is what PATH lookup does anyway, so nothing the box resolves changes."""
    from aisan.spec import BoxSpec

    base = BoxSpec(
        root=tmp_path,
        binds=(),
        tmpfs=(),
        env=(("PATH", "/opt/tool/bin:/usr/bin"),),
        egress=(),
        unshare_net=True,
    )

    out = base.with_path_prefix((Path("/opt/tool/bin"),))

    assert dict(out.env)["PATH"] == "/opt/tool/bin:/usr/bin"
    # A dir that is not already there still goes to the front, and the order of
    # everything behind it is untouched.
    later = out.with_path_prefix((Path("/opt/other/bin"),))
    assert dict(later.env)["PATH"] == "/opt/other/bin:/opt/tool/bin:/usr/bin"


def test_diamond_included_mount_covers_a_path_regardless_of_include_order(tmp_path):
    # M15: common mounts /tools; a and b both include it (a diamond, expanded
    # once). b's path=/tools/bin is covered by that mount. Which branch expands
    # common first depends on include order in a file the reader is not looking
    # at -- coverage must not, since the final box has the mount either way.
    (tmp_path / "common.toml").write_text('ro = ["/tools"]\n')
    (tmp_path / "a.toml").write_text('include = ["common.toml"]\n')
    (tmp_path / "b.toml").write_text(
        'include = ["common.toml"]\npath = ["/tools/bin"]\n'
    )
    for order in ('["a.toml", "b.toml"]', '["b.toml", "a.toml"]'):
        root = tmp_path / "root.toml"
        root.write_text(f"include = {order}\n")
        spec = load(root, egress=())  # must not raise in EITHER order
        assert Path("/tools/bin") in spec.path


def test_a_path_entry_no_file_in_the_tree_mounts_is_uncovered(tmp_path):
    # The global check still catches a PATH dir nothing mounts.
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/elsewhere/bin"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())
