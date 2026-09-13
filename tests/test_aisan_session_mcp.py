# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import tomllib

import pytest

from aisan import Box
from aisan.presets.codex import codex, codex_argv, codex_binary
from aisan.session import LaunchRefused, mcp_launcher_binds, mcp_notice
from aisan.session_mcp import (
    MCPAllowlist,
    SessionMCP,
    claude_host_mcp,
    codex_host_mcp,
    mcp_ro_binds,
    opencode_host_mcp,
)

# Admits every local stdio declaration, as a spec's `mcp = ["*"]` does.
ANY = MCPAllowlist(("*",))


def test_codex_import_keeps_local_servers_and_tool_policy(tmp_path):
    source = tmp_path / "config.toml"
    source.write_text(
        '[mcp_servers.local]\ncommand = "local-mcp"\nargs = ["--one"]\n'
        '[mcp_servers.local.tools.write]\napproval_mode = "approve"\n'
        '[mcp_servers.remote]\nurl = "https://example.test/mcp"\n'
        '[mcp_servers.disabled]\ncommand = "off-mcp"\nenabled = false\n'
    )

    imported = codex_host_mcp(source, allow=ANY)
    output = tmp_path / "state" / "aisan-host-mcp.config.toml"
    imported.write(output)

    assert imported.commands == ("local-mcp",)
    config = tomllib.loads(output.read_text())
    assert set(config["mcp_servers"]) == {"local"}
    assert config["mcp_servers"]["local"]["args"] == ["--one"]
    assert config["mcp_servers"]["local"]["tools"]["write"] == {
        "approval_mode": "approve"
    }
    assert output.stat().st_mode & 0o777 == 0o600


def test_claude_import_keeps_only_enabled_stdio_servers(tmp_path):
    source = tmp_path / ".claude.json"
    source.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "type": "stdio",
                        "command": "local-mcp",
                        "args": ["--one"],
                        "env": {"EXPLICIT": "value"},
                    },
                    "remote": {"type": "http", "url": "https://example.test"},
                    "disabled": {"command": "off-mcp", "enabled": False},
                }
            }
        )
    )

    imported = claude_host_mcp(source, allow=ANY)

    assert imported.commands == ("local-mcp",)
    assert imported.document == {
        "mcpServers": {
            "local": {
                "type": "stdio",
                "command": "local-mcp",
                "args": ["--one"],
                "env": {"EXPLICIT": "value"},
            }
        }
    }


def test_opencode_import_parses_jsonc_and_keeps_only_local_servers(
    tmp_path, monkeypatch
):
    source = tmp_path / "opencode.jsonc"
    source.write_text(
        """
        {
          // A local server starts inside the box.
          "mcp": {
            "local": {
              "type": "local",
              "command": ["local-mcp", "--one"], // trailing after comment
            },
            "remote": {
              "type": "remote",
              "url": "https://example.test/mcp",
            },
          },
        }
        """
    )
    monkeypatch.setenv("OPENCODE_CONFIG", str(source))

    imported = opencode_host_mcp(allow=ANY)

    assert imported.commands == ("local-mcp",)
    assert imported.document == {
        "mcp": {
            "local": {
                "type": "local",
                "command": ["local-mcp", "--one"],
            }
        }
    }


def test_mcp_binds_chase_a_uv_tool_launcher_and_python(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / ".local" / "share" / "uv" / "tools" / "local-mcp"
    interpreter = home / ".local" / "share" / "uv" / "python" / "cpython"
    shim_dir.mkdir(parents=True)
    (tool / "bin").mkdir(parents=True)
    (interpreter / "bin").mkdir(parents=True)
    executable = tool / "bin" / "local-mcp"
    executable.write_text("#!python\n")
    executable.chmod(0o755)
    (interpreter / "bin" / "python3").write_text("")
    (tool / "bin" / "python").symlink_to(interpreter / "bin" / "python3")
    (shim_dir / "local-mcp").symlink_to(executable)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    config = SessionMCP({}, ("local-mcp",), "json")

    assert mcp_ro_binds(config, str(shim_dir)) == (
        shim_dir,
        tool,
        interpreter,
    )


def test_a_bare_local_bin_script_is_refused_not_overmounted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    shim_dir.mkdir(parents=True)
    script = shim_dir / "bare-mcp"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    config = SessionMCP({}, ("bare-mcp",), "json")

    with pytest.raises(ValueError, match="self-contained"):
        mcp_ro_binds(config, str(shim_dir))


def test_a_launcher_symlinked_into_a_checkout_is_refused(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / "projects" / "tool"
    shim_dir.mkdir(parents=True)
    tool.mkdir(parents=True)
    exe = tool / "serve"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    (shim_dir / "checkout-mcp").symlink_to(exe)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    config = SessionMCP({}, ("checkout-mcp",), "json")

    with pytest.raises(ValueError, match="self-contained"):
        mcp_ro_binds(config, str(shim_dir))


def test_a_pyvenv_cfg_proves_a_tool_root(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / "venvs" / "local-mcp"
    shim_dir.mkdir(parents=True)
    (tool / "bin").mkdir(parents=True)
    (tool / "pyvenv.cfg").write_text("home = /usr/bin\n")
    exe = tool / "bin" / "local-mcp"
    exe.write_text("#!python\n")
    exe.chmod(0o755)
    (shim_dir / "local-mcp").symlink_to(exe)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    config = SessionMCP({}, ("local-mcp",), "json")

    assert mcp_ro_binds(config, str(shim_dir)) == (shim_dir, tool)


def test_the_known_credential_list_covers_the_mint_from_disk_backends(monkeypatch):
    from aisan.egress import known_credential_paths
    from aisan.egress.reapi import LUCI_STORE
    from aisan.egress.vertex import default_credentials as adc

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    known = set(known_credential_paths())
    assert LUCI_STORE in known
    assert set(adc()) <= known


def test_the_claude_credential_follows_the_hosts_config_redirect(tmp_path, monkeypatch):
    from pathlib import Path

    from aisan.egress import known_credential_paths
    from aisan.egress.anthropic import default_credentials
    from aisan.session_mcp import claude_config_file

    configured = tmp_path / "elsewhere"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(configured))

    assert default_credentials() == configured / ".credentials.json"
    assert claude_config_file() == configured / ".claude.json"
    assert default_credentials() in set(known_credential_paths())

    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert default_credentials() == Path.home() / ".claude" / ".credentials.json"
    assert claude_config_file() == Path.home() / ".claude.json"


def test_no_launcher_bind_may_contain_a_backend_credential_store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    root = home / ".local" / "share" / "opencode"
    (root / "bin").mkdir(parents=True)
    shim_dir.mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (root / "auth.json").write_text("{}")
    exe = root / "bin" / "evil-mcp"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    (shim_dir / "evil-mcp").symlink_to(exe)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    config = SessionMCP({}, ("evil-mcp",), "json")

    with pytest.raises(ValueError, match=r"auth\.json"):
        mcp_ro_binds(config, str(shim_dir))


def test_an_unbindable_launcher_is_a_launch_refusal_not_a_traceback(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    shim_dir.mkdir(parents=True)
    script = shim_dir / "bare-mcp"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    config = SessionMCP({}, ("bare-mcp",), "json")

    with pytest.raises(LaunchRefused, match="self-contained"):
        mcp_launcher_binds(config)


def test_missing_mcp_command_fails_before_the_box_owns_the_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    config = SessionMCP({}, ("definitely-absent-mcp",), "json")

    with pytest.raises(FileNotFoundError, match="definitely-absent-mcp"):
        mcp_ro_binds(config, "/usr/bin")


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
def test_real_codex_loads_the_generated_mcp_profile(tmp_path):
    source = tmp_path / "host-config.toml"
    source.write_text('[mcp_servers.local]\ncommand = "local-mcp"\n')
    imported = codex_host_mcp(source, allow=ANY)
    state = tmp_path / "state"
    imported.write(state / "aisan-host-mcp.config.toml")

    result = subprocess.run(
        [
            "codex",
            "--profile",
            "aisan-host-mcp",
            "mcp",
            "list",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CODEX_HOME": str(state)},
    )

    assert result.returncode == 0, result.stderr
    servers = json.loads(result.stdout)
    assert [(server["name"], server["transport"]["command"]) for server in servers] == [
        ("local", "local-mcp")
    ]


@pytest.mark.live
@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_real_codex_starts_an_imported_mcp_server_inside_the_box(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / ".local" / "share" / "tools" / "local-mcp"
    shim_dir.mkdir(parents=True)
    (tool / "bin").mkdir(parents=True)

    (tool / "pyvenv.cfg").write_text("home = /usr/bin\n")
    server = tool / "bin" / "local-mcp"
    server.write_text(
        "#!/usr/bin/python3\n"
        "import json, pathlib, sys\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "marker.write_text(json.dumps({\n"
        "    'host_only_visible': (pathlib.Path.home() / 'host-only').exists(),\n"
        "}))\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    method = request.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-06-18',\n"
        "                  'capabilities': {'tools': {}},\n"
        "                  'serverInfo': {'name': 'boxed', 'version': '1'}}\n"
        "    elif method == 'tools/list':\n"
        "        result = {'tools': [{'name': 'inside_box',\n"
        "                  'description': 'started in the box',\n"
        "                  'inputSchema': {'type': 'object'}}]}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'],\n"
        "                      'result': result}), flush=True)\n"
    )
    server.chmod(0o755)
    (shim_dir / "local-mcp").symlink_to(server)
    (home / "host-only").write_text("not bound into the box")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    marker = worktree / "mcp-started.json"
    source = tmp_path / "host-config.toml"
    source.write_text(
        '[mcp_servers.local]\ncommand = "local-mcp"\n'
        f"args = [{json.dumps(str(marker))}]\n"
    )
    imported = codex_host_mcp(source, allow=ANY)
    state = tmp_path / "state"
    imported.write(state / "config.toml")
    base = codex(
        worktree,
        state=state,
        extra_ro=mcp_ro_binds(imported),
        extra_env=(("PATH", f"{shim_dir}:/usr/bin"),),
    )
    spec = dataclasses.replace(
        base, limits=dataclasses.replace(base.limits, use_cgroup=False)
    )
    box = Box(spec, box_id="codex-mcp-inside-box")

    async with box:
        process = await asyncio.create_subprocess_exec(
            *box.command(
                codex_argv(
                    ("app-server", "--stdio"),
                    overrides=("features.apps=false",),
                )
            ),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **box.env},
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            initialize = {
                "method": "initialize",
                "id": 0,
                "params": {"clientInfo": {"name": "aisan-test", "version": "1"}},
            }
            process.stdin.write((json.dumps(initialize) + "\n").encode())
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=60)
            if not line:
                assert process.stderr is not None
                pytest.fail((await process.stderr.read()).decode())
            response = json.loads(line)
            assert response.get("id") == 0, response

            requests = (
                {"method": "initialized", "params": {}},
                {
                    "method": "mcpServerStatus/list",
                    "id": 1,
                    "params": {"detail": "full"},
                },
            )
            for request in requests:
                process.stdin.write((json.dumps(request) + "\n").encode())
            await process.stdin.drain()
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.05)
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.terminate()
                await process.wait()

    assert json.loads(marker.read_text()) == {"host_only_visible": False}


def _claude_config(path, servers):
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def test_a_box_that_names_no_server_starts_none(tmp_path):
    source = _claude_config(
        tmp_path / ".claude.json",
        {"v8-utils": {"command": "v8-mcp"}, "nvim": {"command": "nv"}},
    )

    imported = claude_host_mcp(source)

    assert not imported.enabled
    assert imported.names == ()
    assert imported.document == {"mcpServers": {}}
    # The notice needs the command: it is one of the tokens that selects it.
    assert imported.withheld == (("v8-utils", "v8-mcp"), ("nvim", "nv"))


def test_an_entry_selects_by_config_name_or_by_command_basename(tmp_path):
    source = _claude_config(
        tmp_path / ".claude.json",
        {
            "v8-utils": {"command": "v8-mcp"},
            "bnz": {"command": "bnz-mcp"},
            "nvim": {"command": "nv"},
        },
    )

    # The name the operator reads in the client and the binary that runs in the
    # box usually differ, so both select.
    assert claude_host_mcp(source, allow=MCPAllowlist(("v8-utils",))).names == (
        "v8-utils",
    )
    by_command = claude_host_mcp(source, allow=MCPAllowlist(("bnz-mcp",)))
    assert by_command.names == ("bnz",)
    assert by_command.withheld == (("v8-utils", "v8-mcp"), ("nvim", "nv"))


def test_entries_that_name_no_declaration_are_reported(tmp_path):
    source = _claude_config(tmp_path / ".claude.json", {"nvim": {"command": "nv"}})

    imported = claude_host_mcp(source, allow=MCPAllowlist(("nvim", "gone", "typo")))

    assert imported.names == ("nvim",)
    assert imported.unmatched == ("gone", "typo")


def test_a_command_entry_admits_every_declaration_that_runs_it(tmp_path):
    """A multiplexer command is not a selector; the config name is."""
    source = _claude_config(
        tmp_path / ".claude.json",
        {
            "mine": {"command": "npx", "args": ["-y", "@me/a"]},
            "theirs": {"command": "npx", "args": ["-y", "@them/b"]},
        },
    )

    assert claude_host_mcp(source, allow=MCPAllowlist(("npx",))).names == (
        "mine",
        "theirs",
    )
    assert claude_host_mcp(source, allow=MCPAllowlist(("mine",))).names == ("mine",)


def test_a_remote_or_disabled_declaration_reads_as_unimportable_not_unknown(tmp_path):
    source = _claude_config(
        tmp_path / ".claude.json",
        {
            "linear": {"type": "sse", "url": "https://example.test"},
            "off": {"command": "off-mcp", "enabled": False},
        },
    )

    imported = claude_host_mcp(source, allow=MCPAllowlist(("linear", "off")))

    assert imported.unmatched == ("linear", "off")
    # Naming one of these is not a typo, and the notice must not imply it is.
    notice = mcp_notice(imported)
    assert "no local stdio server" in notice
    assert "remote or disabled declaration is never imported" in notice


def test_the_wildcard_is_never_reported_as_naming_nothing(tmp_path):
    source = _claude_config(tmp_path / ".claude.json", {})

    # A host that declares no server is not a mistake in the spec.
    assert claude_host_mcp(source, allow=ANY).unmatched == ()
    assert mcp_notice(claude_host_mcp(source, allow=ANY)) is None


def test_a_full_path_entry_selects_one_binary_and_not_its_namesake(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / "venvs" / "nvim-mcp"
    shim_dir.mkdir(parents=True)
    (tool / "bin").mkdir(parents=True)
    exe = tool / "bin" / "nv"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    (shim_dir / "nv").symlink_to(exe)
    decoy_dir = tmp_path / "elsewhere"
    decoy_dir.mkdir()
    decoy = decoy_dir / "nv"
    decoy.write_text("#!/bin/sh\n")
    decoy.chmod(0o755)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.setenv("HOME", str(home))  # for `~` in an entry
    source = _claude_config(tmp_path / ".claude.json", {"nvim": {"command": "nv"}})

    # The shim on the search path, its target, and the `~` spelling are one
    # binary; a different binary of the same name is not.
    for entry in (str(shim_dir / "nv"), str(exe), "~/venvs/nvim-mcp/bin/nv"):
        assert claude_host_mcp(source, allow=MCPAllowlist((entry,))).names == ("nvim",)
    assert claude_host_mcp(source, allow=MCPAllowlist((str(decoy),))).names == ()


def test_the_wildcard_admits_every_local_declaration_in_each_client(
    tmp_path, monkeypatch
):
    claude = _claude_config(tmp_path / ".claude.json", {"a": {"command": "a-mcp"}})
    codex = tmp_path / "config.toml"
    codex.write_text('[mcp_servers.a]\ncommand = "a-mcp"\n')
    opencode = tmp_path / "opencode.json"
    opencode.write_text(
        json.dumps({"mcp": {"a": {"type": "local", "command": ["a-mcp"]}}})
    )

    for imported in (
        claude_host_mcp(claude, allow=ANY),
        codex_host_mcp(codex, allow=ANY),
        opencode_host_mcp(opencode, allow=ANY),
    ):
        assert imported.names == ("a",)
        assert imported.withheld == ()
    for denied in (
        claude_host_mcp(claude),
        codex_host_mcp(codex),
        opencode_host_mcp(opencode),
    ):
        assert denied.names == ()
        assert denied.withheld == (("a", "a-mcp"),)


def test_the_notice_reports_withheld_servers_and_unnamed_entries():
    config = SessionMCP(
        document={},
        commands=(),
        kind="json",
        withheld=(("nvim", "/opt/nvim-mcp/bin/nv"),),
        unmatched=("typo",),
    )

    notice = mcp_notice(config)

    assert "1 host MCP server(s) withheld" in notice
    assert "nvim (/opt/nvim-mcp/bin/nv)" in notice
    assert "typo" in notice
    assert "start inside the box" not in notice


def test_the_notice_is_absent_when_the_host_declares_nothing():
    assert mcp_notice(SessionMCP(document={}, commands=(), kind="json")) is None


def test_the_import_notice_names_the_servers_and_the_env_carriers():
    config = SessionMCP(
        document={},
        commands=("a-mcp", "b-mcp"),
        kind="json",
        names=("plain", "tokened"),
        env_names=("tokened",),
    )

    notice = mcp_notice(config)

    assert "plain" in notice
    assert "2 host MCP server(s)" in notice

    assert notice.count("tokened") == 2
    assert "readable" in notice


def test_the_import_notice_omits_the_environment_line_when_none_carries_one():
    config = SessionMCP(document={}, commands=("a-mcp",), kind="json", names=("plain",))

    notice = mcp_notice(config)

    assert "plain" in notice
    assert "environment" not in notice


def test_imported_servers_report_their_env_carriers_per_client_key(tmp_path):
    claude_source = tmp_path / ".claude.json"
    claude_source.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "plain": {"command": "local-mcp"},
                    "tokened": {"command": "local-mcp", "env": {"TOKEN": "x"}},
                    "empty-env": {"command": "local-mcp", "env": {}},
                }
            }
        )
    )
    claude = claude_host_mcp(claude_source, allow=ANY)
    assert claude.names == ("plain", "tokened", "empty-env")
    assert claude.env_names == ("tokened",)

    opencode_source = tmp_path / "opencode.json"
    opencode_source.write_text(
        json.dumps(
            {
                "mcp": {
                    "plain": {"type": "local", "command": ["local-mcp"]},
                    "tokened": {
                        "type": "local",
                        "command": ["local-mcp"],
                        "environment": {"TOKEN": "x"},
                    },
                }
            }
        )
    )
    opencode = opencode_host_mcp(opencode_source, allow=ANY)
    assert opencode.names == ("plain", "tokened")
    assert opencode.env_names == ("tokened",)
