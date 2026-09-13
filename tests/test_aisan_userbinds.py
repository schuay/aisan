# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


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
    f = _spec_file(tmp_path, 'ro = ["/refs/a"]\nrw = ["/scratch/b"]\n')
    assert load(f, egress=()).binds == [
        Bind(Path("/refs/a"), RO, optional=True),
        Bind(Path("/scratch/b"), RW),
    ]


def test_an_empty_file_names_nothing(tmp_path):
    assert load(_spec_file(tmp_path, ""), egress=()) == UserSpec([], ())


def test_tilde_expands_and_relative_resolves_against_the_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "refs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/refs", "neighbour"]\n')
    assert load(f, egress=()).binds == [
        Bind(home / "refs", RO, optional=True),
        Bind(tmp_path / "neighbour", RO, optional=True),
    ]


def test_unknown_keys_are_refused_not_ignored(tmp_path):
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
    with pytest.raises(ValueError, match=f"appears in {match}"):
        load(_spec_file(tmp_path, text), egress=())


@pytest.mark.parametrize(
    "text",
    [
        'ro = ["/a/b"]\nrw = ["/a"]\n',
        'overlay = ["/a/cache"]\nrw = ["/a"]\n',
        'overlay = ["/a/cache"]\nro = ["/a"]\n',
        'rw = ["/a"]\nro = ["/a/../a/b"]\n',
    ],
)
def test_a_mount_nested_under_a_later_broader_one_is_refused(tmp_path, text):
    with pytest.raises(ValueError, match="remove the nesting"):
        load(_spec_file(tmp_path, text), egress=())


def test_same_mode_nesting_is_allowed(tmp_path):
    f = _spec_file(tmp_path, 'rw = ["/a/b", "/a"]\n')
    assert [b.path for b in load(f, egress=()).binds] == [Path("/a/b"), Path("/a")]


def test_the_includer_may_still_shadow_an_included_mount(tmp_path):
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
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    f = _spec_file(tmp_path, 'ro = ["~/a", "%s"]\n' % (home / "a"))
    assert load(f, egress=()).binds == [Bind(home / "a", RO, optional=True)]


def test_order_is_overlays_then_ro_then_rw(tmp_path):
    f = _spec_file(tmp_path, 'ro = ["/a", "/b"]\nrw = ["/c"]\noverlay = ["/d"]\n')
    binds = load(f, egress=()).binds
    assert [b.path for b in binds] == [Path("/d"), Path("/a"), Path("/b"), Path("/c")]
    assert isinstance(binds[0], Overlay)


def test_a_bind_containing_a_backend_credential_is_refused(tmp_path):
    creds = tmp_path / "creds" / "k.json"
    f = _spec_file(tmp_path, f'ro = ["{tmp_path / "creds"}"]\n')
    with pytest.raises(ValueError) as e:
        load(f, egress=(AnthropicBackend(credentials=creds),))
    assert "anthropic" in str(e.value)
    assert str(creds) in str(e.value)
    assert str(f) in str(e.value)


def test_the_guard_covers_every_backend_that_declares_a_file(tmp_path):
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
    creds = tmp_path / "creds" / "k.json"
    f = _spec_file(tmp_path, f'ro = ["{tmp_path / "elsewhere"}"]\n')
    assert load(f, egress=(AnthropicBackend(credentials=creds),)).binds


def test_a_backend_without_a_file_cannot_be_exposed(tmp_path):
    from aisan.egress.base import Backend

    class _Minted(Backend):
        name = "minted"
        port = 8790

        async def serve(self, runtime_dir):  # pragma: no cover - shape only
            yield

    f = _spec_file(tmp_path, f'ro = ["{tmp_path / "elsewhere"}"]\n')
    assert load(f, egress=(_Minted(),)).binds


def test_a_user_bind_may_not_name_another_clients_credential(tmp_path, monkeypatch):
    from aisan.egress.anthropic import AnthropicBackend

    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    (codex / "auth.json").write_text("{}")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    f = _spec_file(tmp_path, f'ro = ["{codex}"]\n')
    with pytest.raises(ValueError, match=r"auth\.json"):
        load(f, egress=(AnthropicBackend(credentials=tmp_path / "creds.json"),))


@pytest.mark.parametrize("store", [".ssh", ".gnupg"])
def test_a_user_bind_may_not_name_a_private_key_store(tmp_path, monkeypatch, store):
    home = tmp_path / "home"
    (home / store).mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))

    f = _spec_file(tmp_path, f'ro = ["{home / store}"]\n')
    with pytest.raises(ValueError, match="private key store"):
        load(f, egress=())


def test_egress_is_required_so_the_guard_is_never_off_by_default(tmp_path):
    f = _spec_file(tmp_path, 'ro = ["/a"]\n')
    with pytest.raises(TypeError):
        load(f)  # type: ignore[call-arg]


def test_an_overlay_is_mandatory_and_keeps_its_own_bind_type(tmp_path):
    f = _spec_file(tmp_path, 'overlay = ["/cache/store"]\n')
    assert load(f, egress=()).binds == [Overlay(Path("/cache/store"))]


def test_path_entries_are_returned_separately_from_the_binds(tmp_path):
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools"]\n')
    spec = load(f, egress=())
    assert spec.binds == [Bind(Path("/tools"), RO, optional=True)]
    assert spec.path == (Path("/tools"),)


@pytest.mark.parametrize(
    "mount", ['ro = ["/tools"]', 'rw = ["/tools"]', 'overlay = ["/tools"]']
)
def test_a_path_entry_may_be_covered_by_any_mount_key(tmp_path, mount):
    f = _spec_file(tmp_path, f'{mount}\npath = ["/tools/bin"]\n')
    assert load(f, egress=()).path == (Path("/tools/bin"),)


def test_a_path_entry_no_mount_covers_is_refused(tmp_path):
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/elsewhere"]\n')
    with pytest.raises(ValueError, match="not covered by any ro, rw or overlay"):
        load(f, egress=())


def test_a_sibling_prefix_does_not_cover_a_path_entry(tmp_path):
    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/toolsx"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())


def _named_spec(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_include_expands_in_place_and_the_includer_shadows(tmp_path):

    (tmp_path / "base").mkdir()
    _named_spec(tmp_path / "base" / "tools.toml", 'ro = ["~/tools"]\n')
    outer = _named_spec(
        tmp_path / "outer.toml",
        'include = ["base/tools.toml"]\nrw = ["~/tools"]\n',
    )

    spec = load(outer, egress=())

    assert [(b.path, getattr(b, "mode", None)) for b in spec.binds] == [
        (Path.home() / "tools", RO),
        (Path.home() / "tools", RW),
    ]


def test_an_include_resolves_against_the_including_file(tmp_path):

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

    _named_spec(tmp_path / "tools.toml", 'ro = ["~/tools"]\n')
    outer = _named_spec(
        tmp_path / "outer.toml",
        'include = ["tools.toml"]\npath = ["~/tools/bin"]\n',
    )

    spec = load(outer, egress=())

    assert spec.path == (Path.home() / "tools" / "bin",)


def test_the_credential_guard_reaches_inside_an_include(tmp_path):

    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    _named_spec(tmp_path / "inner.toml", f'ro = ["{creds.parent}"]\n')
    outer = _named_spec(tmp_path / "outer.toml", 'include = ["inner.toml"]\n')

    with pytest.raises(ValueError, match=r"inner\.toml: .* would expose"):
        load(outer, egress=(AnthropicBackend(credentials=creds),))


def test_the_credential_guard_looks_inside_the_store_too(tmp_path):

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

    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools/bin:/tmp"]\n')
    with pytest.raises(ValueError, match="contains"):
        load(f, egress=())


def test_a_path_entry_cannot_dotdot_escape_its_covering_mount(tmp_path):

    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/tools/bin/../../../etc"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())


def test_an_unexpandable_home_is_a_named_valueerror_not_a_runtimeerror(tmp_path):

    f = _spec_file(tmp_path, 'ro = ["~nosuchuser_aisan_xyz/x"]\n')
    with pytest.raises(ValueError, match="cannot be expanded"):
        load(f, egress=())


def test_with_path_prefix_refuses_a_relative_or_colon_bearing_dir(tmp_path):

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

    out = base.with_path_prefix((Path("/opt/tool/bin"),))
    assert dict(out.env)["PATH"].startswith("/opt/tool/bin:")


def test_a_path_prefix_already_present_is_not_repeated(tmp_path):
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

    later = out.with_path_prefix((Path("/opt/other/bin"),))
    assert dict(later.env)["PATH"] == "/opt/other/bin:/opt/tool/bin:/usr/bin"


def test_diamond_included_mount_covers_a_path_regardless_of_include_order(tmp_path):

    (tmp_path / "common.toml").write_text('ro = ["/tools"]\n')
    (tmp_path / "a.toml").write_text('include = ["common.toml"]\n')
    (tmp_path / "b.toml").write_text(
        'include = ["common.toml"]\npath = ["/tools/bin"]\n'
    )
    for order in ('["a.toml", "b.toml"]', '["b.toml", "a.toml"]'):
        root = tmp_path / "root.toml"
        root.write_text(f"include = {order}\n")
        spec = load(root, egress=())
        assert Path("/tools/bin") in spec.path


def test_a_path_entry_no_file_in_the_tree_mounts_is_uncovered(tmp_path):

    f = _spec_file(tmp_path, 'ro = ["/tools"]\npath = ["/elsewhere/bin"]\n')
    with pytest.raises(ValueError, match="not covered"):
        load(f, egress=())


def test_mcp_entries_union_across_the_include_tree_in_file_order(tmp_path):
    inner = tmp_path / "nvim.toml"
    inner.write_text('mcp = ["nvim", "v8-mcp"]\n')
    f = _spec_file(tmp_path, 'include = ["./nvim.toml"]\nmcp = ["bnz", "v8-mcp"]\n')

    # Includes first, like mounts, and each entry once.
    assert load(f, egress=()).mcp == ("nvim", "v8-mcp", "bnz")


def test_an_mcp_entry_may_be_a_name_an_absolute_path_or_the_wildcard(tmp_path):
    f = _spec_file(tmp_path, 'mcp = ["v8-mcp", "~/tools/nvim-mcp/bin/nv", "*"]\n')

    # Entries stay as written; session_mcp resolves them against the host.
    assert load(f, egress=()).mcp == ("v8-mcp", "~/tools/nvim-mcp/bin/nv", "*")


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ('mcp = "v8-mcp"', "array of non-empty strings"),
        ("mcp = [1]", "array of non-empty strings"),
        ('mcp = [""]', "array of non-empty strings"),
        ('mcp = ["./bin/nv"]', "relative path"),
        ('mcp = ["../bin/nv"]', "relative path"),
        ('mcp = ["nv*"]', "the only glob"),
        ('mcp = ["/bin/nv:/bin/other"]', "contains ':'"),
    ],
)
def test_a_malformed_mcp_entry_is_a_named_error(tmp_path, value, match):
    with pytest.raises(ValueError, match=match):
        load(_spec_file(tmp_path, value + "\n"), egress=())


def test_every_shipped_example_spec_loads():
    """The examples are what a reader copies; keep them loadable."""
    examples = sorted((Path(__file__).parents[1] / "examples").glob("*.toml"))
    assert examples
    for path in examples:
        load(path, egress=())
