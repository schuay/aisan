# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host MCP declarations imported without moving their processes outside."""

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
from aisan.session import mcp_notice
from aisan.session_mcp import (
    SessionMCP,
    claude_host_mcp,
    codex_host_mcp,
    mcp_ro_binds,
    opencode_host_mcp,
)


def test_codex_import_keeps_local_servers_and_tool_policy(tmp_path):
    source = tmp_path / "config.toml"
    source.write_text(
        '[mcp_servers.local]\ncommand = "local-mcp"\nargs = ["--one"]\n'
        '[mcp_servers.local.tools.write]\napproval_mode = "approve"\n'
        '[mcp_servers.remote]\nurl = "https://example.test/mcp"\n'
        '[mcp_servers.disabled]\ncommand = "off-mcp"\nenabled = false\n'
    )

    imported = codex_host_mcp(source)
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

    imported = claude_host_mcp(source)

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

    imported = opencode_host_mcp()

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


def test_missing_mcp_command_fails_before_the_box_owns_the_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    config = SessionMCP({}, ("definitely-absent-mcp",), "json")

    with pytest.raises(FileNotFoundError, match="definitely-absent-mcp"):
        mcp_ro_binds(config, "/usr/bin")


@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
def test_real_codex_loads_the_generated_mcp_profile(tmp_path):
    source = tmp_path / "host-config.toml"
    source.write_text('[mcp_servers.local]\ncommand = "local-mcp"\n')
    imported = codex_host_mcp(source)
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


@pytest.mark.skipif(codex_binary() is None, reason="codex is not installed")
async def test_real_codex_starts_an_imported_mcp_server_inside_the_box(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    shim_dir = home / ".local" / "bin"
    tool = home / ".local" / "share" / "tools" / "local-mcp"
    shim_dir.mkdir(parents=True)
    (tool / "bin").mkdir(parents=True)
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
    imported = codex_host_mcp(source)
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


def test_the_import_notice_names_the_servers_and_the_env_carriers():
    """The import is by operator choice, but a silent one is not reviewable.

    Each declaration is copied verbatim into the box, so a server carrying an
    API token in its environment puts that token where the agent can read it --
    the one thing that widens the box without announcing itself. The notice
    names the servers, and names again the ones whose environment travels.
    """
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
    # The env carrier is called out a second time; the plain one is not.
    assert notice.count("tokened") == 2
    assert "readable" in notice


def test_the_import_notice_omits_the_environment_line_when_none_carries_one():
    config = SessionMCP(document={}, commands=("a-mcp",), kind="json", names=("plain",))

    notice = mcp_notice(config)

    assert "plain" in notice
    assert "environment" not in notice


def test_imported_servers_report_their_env_carriers_per_client_key(tmp_path):
    """Claude Code and Codex spell it `env`; opencode spells it `environment`.

    A single spelling would silently under-report on one of the three, which is
    the failure mode a notice about credentials must not have.
    """
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
    claude = claude_host_mcp(claude_source)
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
    opencode = opencode_host_mcp(opencode_source)
    assert opencode.names == ("plain", "tokened")
    assert opencode.env_names == ("tokened",)
