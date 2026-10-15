# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aisan.egress.openai_compat import OpenAICompatBackend
from aisan.opencode_config import config_file, read_config
from aisan.opencode_provider import SDK, resolve_provider
from aisan.session_mcp import MCPAllowlist, opencode_host_mcp
from aisan.upstream import validate_upstream


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "opencode" / "opencode.json"
    source.parent.mkdir()
    document = {
        "provider": {
            "local": {
                "npm": SDK,
                "options": {
                    "baseURL": "http://localhost:8080/secret/v1/",
                    "apiKey": "HOST_SECRET",
                },
                "models": {
                    "alias": {
                        "id": "upstream-model",
                        "name": "Display",
                        "limit": {"context": 16384, "output": 2048},
                    }
                },
            }
        }
    }
    source.write_text(json.dumps(document))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"known": {"api": "https://catalog.example/v1"}}))
    return source, catalog, document


def resolve(host, **kwargs):
    source, catalog, document = host
    return resolve_provider(
        "local", "alias", source=source, catalog=catalog, document=document, **kwargs
    )


def test_selection_is_frozen_and_client_config_contains_only_supported_fields(host):
    source, _, data = host
    data["provider"]["local"]["models"]["alias"].update(
        {"options": {"secret": "HOST_SECRET"}, "modalities": {"input": ["image"]}}
    )
    selected = resolve(host)
    source.write_text("{}")
    data["provider"]["local"]["options"]["baseURL"] = "http://changed"
    with (
        patch.object(Path, "read_text", side_effect=AssertionError("file read")),
        patch.object(Path, "open", side_effect=AssertionError("file open")),
    ):
        backend = OpenAICompatBackend(
            provider="local", model="alias", resolved=selected, upstream_auth="none"
        )
        env = backend.client_env()
        assert backend._resolve_api() == "http://localhost:8080/secret/v1"
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == config["model"] == "local/alias"
    entry = config["provider"]["local"]
    assert entry["npm"] == SDK
    assert (
        entry["options"]["apiKey"]
        == json.loads(env["OPENCODE_AUTH_CONTENT"])["local"]["key"]
    )
    assert entry["models"] == {
        "alias": {
            "id": "upstream-model",
            "name": "Display",
            "limit": {"context": 16384, "output": 2048},
            "tool_call": True,
            "modalities": {"input": ["text"], "output": ["text"]},
        }
    }
    assert "HOST_SECRET" not in str(env) + selected.description("none")
    assert "/secret" not in selected.description("none")
    assert "options.apiKey" in selected.description("none")
    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.custom.context = 1


def test_config_selection_and_empty_xdg(host, monkeypatch):
    source, _, _ = host
    assert config_file() == source
    source.with_suffix(".jsonc").write_text("{}")
    assert config_file() == source
    source.unlink()
    assert config_file() == source.with_suffix(".jsonc")
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(source.parent))
    assert config_file() == source.parent / ".config/opencode/opencode.jsonc"
    monkeypatch.setenv("OPENCODE_CONFIG", "relative.jsonc")
    assert config_file() == Path("relative.jsonc")
    with pytest.raises(ValueError, match="does not exist"):
        read_config(config_file(), strict=True)
    monkeypatch.delenv("OPENCODE_CONFIG")
    assert read_config(config_file(), strict=True) == {}


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1,"a":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        "{} /* unfinished",
        '{"x":1/* c */2}',
        '{"secret":',
    ],
)
def test_strict_jsonc_rejects_ambiguity_without_contents(host, text):
    source, _, _ = host
    source.write_text(text)
    with pytest.raises(ValueError) as error:
        read_config(source, strict=True)
    assert str(source) in str(error.value)
    assert text not in str(error.value)


def test_jsonc_and_mcp_use_one_document(host):
    source, _, _ = host
    source.write_text(
        '{/* comment */ "mcp": {"demo": {"type":"local", "command":["/usr/bin/true"],},},}'
    )
    snapshot = read_config(source, strict=True)
    source.write_text("malformed")
    mcp = opencode_host_mcp(source, document=snapshot, allow=MCPAllowlist(("demo",)))
    assert mcp.names == ("demo",)
    assert "provider" not in mcp.document


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://user:SECRET@host/v1",
        "http://@host",
        "http://host:99999",
        "http://host:bad",
        "http://host:0",
        "file:///tmp/a",
        "//host/v1",
        "http://host/?SECRET",
        "http://host/#SECRET",
        "http://host/ bad",
        "http://host/\nSECRET",
        "http://host/\\SECRET",
        "{env:SECRET}",
        "http://host/${SECRET}",
        "http://host/{file:SECRET}",
    ],
)
def test_url_rejected_without_value(url):
    with pytest.raises(ValueError) as error:
        validate_upstream(url)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "url", ["https://host/v1/", "http://[::1]:8080/v1", "http://host:1234/a/b"]
)
def test_valid_url_preserves_prefix(url):
    assert validate_upstream(url) == url.rstrip("/")


def test_route_precedence_and_catalog_fallback(host):
    assert (
        resolve(host, upstream="https://explicit/v2").upstream == "https://explicit/v2"
    )
    source, catalog, data = host
    assert resolve(host).upstream == "http://localhost:8080/secret/v1"
    data["provider"] = {"known": {"options": {"baseURL": "https://host/v3"}}}
    assert (
        resolve_provider(
            "known", "m", source=source, catalog=catalog, document=data
        ).upstream
        == "https://host/v3"
    )
    data["provider"] = {}
    assert (
        resolve_provider(
            "known", "m", source=source, catalog=catalog, document=data
        ).upstream
        == "https://catalog.example/v1"
    )
    with pytest.raises(ValueError, match="unknown catalog provider"):
        resolve(host, upstream="https://explicit")


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "{env:SECRET}"),
        ("id", "${SECRET}"),
        ("name", ""),
        ("tool_call", 1),
        ("provider", {"npm": "other"}),
        ("npm", "other"),
        ("limit", []),
        ("limit", {"context": True, "output": 1}),
        ("limit", {"context": 100, "output": 100}),
        ("limit", {"context": 100, "output": 10, "input": 101}),
        ("limit", {"context": 100, "output": 0}),
        ("limit", {"context": 100.5, "output": 10}),
    ],
)
def test_selected_model_validation(host, field, value):
    host[2]["provider"]["local"]["models"]["alias"][field] = value
    with pytest.raises((ValueError, TypeError)) as error:
        resolve(host, upstream="http://override")
    assert str(host[0]) in str(error.value)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "change", ["native", "sdk", "missing", "provider", "models", "options", "baseURL"]
)
def test_selected_provider_refusals_even_with_explicit_route(host, change):
    data = host[2]
    entry = data["provider"]["local"]
    if change == "native":
        data["providers"] = {"local": {}}
    elif change == "sdk":
        entry["npm"] = "unsupported"
    elif change == "missing":
        entry["models"] = {}
    elif change == "provider":
        data["provider"]["local"] = []
    elif change == "baseURL":
        entry["options"]["baseURL"] = ""
    else:
        entry[change] = []
    with pytest.raises((ValueError, TypeError)) as error:
        resolve(host, upstream="http://override")
    assert str(host[0]) in str(error.value)


def test_unrelated_provider_and_model_semantics_are_ignored(host):
    host[2]["providers"] = {"unrelated": 1}
    host[2]["provider"]["unrelated"] = 1
    host[2]["provider"]["local"]["models"]["unrelated"] = 1
    assert resolve(host).custom.context == 16384


@pytest.mark.parametrize("shared", [False, True])
async def test_cli_explain_reads_config_but_never_credentials_or_network(
    host, monkeypatch, capsys, shared
):
    import importlib

    from aisan.egress import openai_compat

    cli = importlib.import_module("aisan.cli.opencode")
    source, catalog, _ = host
    monkeypatch.setenv("OPENCODE_CONFIG", str(source))
    monkeypatch.setattr(cli, "DEFAULT_CATALOG", catalog)
    monkeypatch.setattr(cli, "state_dir", lambda *_: source.parent / "state")
    monkeypatch.setattr(cli, "USER_MEMORY", source.parent / "absent-memory")
    repo = source.parent / "repo"
    repo.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("explain attempted credentials or network")

    monkeypatch.setattr(openai_compat, "_read_api_key", forbidden)
    monkeypatch.setattr(OpenAICompatBackend, "_check_upstream", forbidden)
    monkeypatch.setattr(cli, "opencode_binary", forbidden)
    args = [
        str(repo),
        "--provider",
        "local",
        "--model",
        "alias",
        "--no-upstream-auth",
        "--explain",
    ]
    if shared:
        args.append("--net")
    assert await cli._main(args) == 0
    report = capsys.readouterr().out
    assert "context=16384, output=2048" in report
    assert "Upstream auth: none" in report
    assert "http://localhost:8080" in report
    assert "HOST_SECRET" not in report
    assert "/secret" not in report
    assert "assigned at launch" in report if shared else "8714" in report


async def test_cli_freezes_config_and_adds_standalone(host, monkeypatch):
    import importlib

    cli = importlib.import_module("aisan.cli.opencode")
    source, catalog, _ = host
    monkeypatch.setenv("OPENCODE_CONFIG", str(source))
    monkeypatch.setattr(cli, "DEFAULT_CATALOG", catalog)
    monkeypatch.setattr(cli, "state_dir", lambda *_: source.parent / "state")
    monkeypatch.setattr(cli, "USER_MEMORY", source.parent / "absent")
    repo = source.parent / "repo"
    repo.mkdir()

    async def launch(**kwargs):
        assert kwargs["command"](None) == [
            "opencode",
            "--auto",
            "--standalone",
            "--prompt",
            "hello",
        ]
        source.write_text('{"provider":{"local":{"npm":"bad"}}}')
        backend = kwargs["spec"].egress[0]
        assert backend._resolve_api() == "http://localhost:8080/secret/v1"
        assert (
            "alias"
            in json.loads(backend.client_env()["OPENCODE_CONFIG_CONTENT"])["provider"][
                "local"
            ]["models"]
        )
        return 0

    monkeypatch.setattr(cli, "run_interactive", launch)
    assert (
        await cli._main(
            [
                str(repo),
                "--provider",
                "local",
                "--model",
                "alias",
                "--",
                "--prompt",
                "hello",
            ]
        )
        == 0
    )
