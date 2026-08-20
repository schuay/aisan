# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host MCP declarations imported into an isolated interactive session.

Only local process transports cross this boundary. The clients start those
processes inside the box, where they inherit its filesystem, cleared environment
and network namespace. Each retained declaration is copied verbatim, so explicit
arguments and environment values are inside the box by operator choice. Remote
MCP declarations and their separate authentication state stay on the host.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .launch import interpreter_roots


@dataclass(frozen=True)
class SessionMCP:
    """One client's local MCP declarations and the commands they launch."""

    document: dict[str, object]
    commands: tuple[str, ...]
    kind: Literal["json", "toml"]

    @property
    def enabled(self) -> bool:
        return bool(self.commands)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "toml":
            text = _toml_document(self.document)
        else:
            text = json.dumps(self.document, indent=2) + "\n"
        path.write_text(text)
        path.chmod(0o600)


def codex_host_mcp(path: Path | None = None) -> SessionMCP:
    source = path or _codex_home() / "config.toml"
    data = _read_toml(source)
    servers = _table(data.get("mcp_servers"), source, "mcp_servers")
    kept = {
        name: server
        for name, server in servers.items()
        if isinstance(server.get("command"), str)
        and server.get("enabled", True) is not False
    }
    return SessionMCP(
        document={"mcp_servers": kept},
        commands=tuple(str(server["command"]) for server in kept.values()),
        kind="toml",
    )


def claude_host_mcp(path: Path | None = None) -> SessionMCP:
    source = path or _claude_config_file()
    data = _read_json(source)
    servers = _table(data.get("mcpServers"), source, "mcpServers")
    kept = {
        name: server
        for name, server in servers.items()
        if server.get("type", "stdio") == "stdio"
        and isinstance(server.get("command"), str)
        and server.get("enabled", True) is not False
    }
    return SessionMCP(
        document={"mcpServers": kept},
        commands=tuple(str(server["command"]) for server in kept.values()),
        kind="json",
    )


def opencode_host_mcp(path: Path | None = None) -> SessionMCP:
    source = path or _opencode_config_file()
    data = _read_jsonc(source)
    servers = _table(data.get("mcp"), source, "mcp")
    kept: dict[str, dict[str, object]] = {}
    commands: list[str] = []
    for name, server in servers.items():
        command = server.get("command")
        if (
            server.get("type") != "local"
            or server.get("enabled", True) is False
            or not isinstance(command, list)
            or not command
            or not isinstance(command[0], str)
        ):
            continue
        kept[name] = server
        commands.append(command[0])
    return SessionMCP(document={"mcp": kept}, commands=tuple(commands), kind="json")


def mcp_search_path() -> str:
    """The PATH shared by all interactive boxes and launcher resolution."""
    return os.pathsep.join(
        (str(Path.home() / ".local" / "bin"), "/usr/bin", "/usr/local/bin")
    )


def mcp_ro_binds(
    config: SessionMCP, search_path: str | None = None
) -> tuple[Path, ...]:
    """Read-only paths needed to execute the configured servers in the box."""
    if not config.enabled:
        return ()
    path = search_path or mcp_search_path()
    binds = [Path.home() / ".local" / "bin"]
    for command in config.commands:
        binds.extend(_launcher_binds(command, path))
    return tuple(dict.fromkeys(binds))


def _launcher_binds(command: str, search_path: str) -> list[Path]:
    """Resolve a home-installed launcher and the interpreter behind its venv."""
    executable = shutil.which(command, path=search_path)
    if executable is None:
        raise FileNotFoundError(
            f"MCP command {command!r} is not on the box PATH ({search_path})"
        )
    real = Path(executable).resolve()
    home = Path.home().resolve()
    if not real.is_relative_to(home):
        return []

    root = real.parents[1] if len(real.parents) > 1 else real
    if root == home:
        root = real
    binds = [root]
    python = root / "bin" / "python"
    if python.is_symlink():
        binds.extend(path for path in interpreter_roots(python) if path not in binds)
    return binds


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def _claude_config_file() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    return (
        Path(configured) / ".claude.json"
        if configured
        else Path.home() / ".claude.json"
    )


def _opencode_config_file() -> Path:
    configured = os.environ.get("OPENCODE_CONFIG")
    if configured:
        return Path(configured)
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    base = root / "opencode"
    json_path = base / "opencode.json"
    return json_path if json_path.exists() else base / "opencode.jsonc"


def _read_toml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"host MCP config {path} is not valid TOML") from error
    if not isinstance(data, dict):
        raise TypeError(f"host MCP config {path} must contain a table")
    return data


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"host MCP config {path} is not valid JSON") from error
    if not isinstance(data, dict):
        raise TypeError(f"host MCP config {path} must contain an object")
    return data


def _read_jsonc(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(_strip_jsonc(path.read_text()))
    except json.JSONDecodeError as error:
        raise ValueError(f"host MCP config {path} is not valid JSONC") from error
    if not isinstance(data, dict):
        raise TypeError(f"host MCP config {path} must contain an object")
    return data


def _table(value: object, path: Path, key: str) -> dict[str, dict[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(server, dict)
        for name, server in value.items()
    ):
        raise ValueError(f"host MCP config {path}: {key} must be a table of tables")
    return value


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas without touching strings."""
    out: list[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            index += 1
            continue
        if text.startswith("//", index):
            index = text.find("\n", index)
            if index < 0:
                break
            out.append("\n")
            index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                return text
            out.extend("\n" for char in text[index : end + 2] if char == "\n")
            index = end + 2
            continue
        out.append(char)
        index += 1
    return _strip_trailing_commas("".join(out))


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            out.append(char)
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "]}":
                continue
        out.append(char)
    return "".join(out)


def _toml_document(document: dict[str, object]) -> str:
    lines: list[str] = []

    def emit(table: dict[str, object], path: tuple[str, ...]) -> None:
        values = [
            (key, value) for key, value in table.items() if not isinstance(value, dict)
        ]
        children = [
            (key, value) for key, value in table.items() if isinstance(value, dict)
        ]
        if path:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(json.dumps(part) for part in path) + "]")
        lines.extend(
            f"{json.dumps(key)} = {_toml_value(value)}" for key, value in values
        )
        for key, child in children:
            emit(child, (*path, key))

    emit(document, ())
    return "\n".join(lines) + "\n"


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{json.dumps(key)} = {_toml_value(item)}" for key, item in value.items()
        )
        return "{ " + entries + " }"
    raise TypeError(f"unsupported MCP config value {value!r}")
