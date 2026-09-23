# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Import host MCP declarations into an isolated interactive session.

Only local process transports are imported, and only those a bind spec names
through `MCPAllowlist`. A host client config is one list shared by every box,
so importing it whole gives an unattended box whatever was added for an
attended one. Clients start the selected servers inside the box with its
filesystem, environment, and network namespace. Arguments and environment
values from each imported declaration are readable inside the box. Remote MCP
declarations remain on the host.

Selection happens before `mcp_ro_binds` resolves anything, so a declaration
this box does not want cannot refuse the launch by being unresolvable.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .egress import known_credential_paths
from .egress.anthropic import claude_config_dir
from .egress.base import credential_overlap
from .launch import interpreter_roots
from .statedir import write_sealed
from .userbinds import WILDCARD


@dataclass(frozen=True)
class MCPAllowlist:
    """The host MCP servers a box may start, as written in its bind specs.

    An entry matches a declaration by config name, by the basename of its
    command, or by the command's full path. The name is what the operator
    reads in the client; the command is what actually runs in the box, and the
    two often differ.

    A command entry admits every declaration that runs that binary. For a
    multiplexer such as `npx` or `uvx` that is every server it launches,
    including ones the host config gains later, because the arguments that
    distinguish them are not matched. Only the config name selects exactly one
    declaration.

    A full path decides which binary the match and the launcher binds resolve
    to. It does not decide what the client execs: a user spec's `path` entries
    come earlier on the box PATH, so a directory mounted there can still
    shadow a launcher of the same name.

    The empty allowlist admits nothing.
    """

    entries: tuple[str, ...] = ()


def _matches(entry: str, name: str, command: str, search_path: str) -> bool:
    """Report whether one spec entry names this declaration."""
    if entry in (WILDCARD, name):
        return True
    if "/" in entry or entry.startswith("~"):
        return _same_binary(entry, command, search_path)
    return Path(command).name == entry


def _same_binary(entry: str, command: str, search_path: str) -> bool:
    """Compare a full-path entry with the binary a declaration would run.

    A launcher is usually a symlink into a tool root, so compare both the
    spelling found on the search path and its target, and accept either
    spelling from the spec. An unresolvable command matches nothing; only a
    selected server's command has to resolve.
    """
    found = shutil.which(command, path=search_path)
    if found is None:
        return False
    target = Path(entry).expanduser()
    declared = Path(found)
    return {target, target.resolve()} & {declared, declared.resolve()} != set()


def _select(
    servers: dict[str, dict[str, object]],
    commands: Mapping[str, str],
    allow: MCPAllowlist,
    search_path: str | None = None,
) -> tuple[dict[str, dict[str, object]], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Split host declarations into the selected ones and the rest.

    Also report entries that named no local stdio declaration, which covers a
    typo as well as an entry naming a remote or disabled server. A per-tool
    spec included on a host without that tool is normal, so this is a note
    rather than a refusal. The wildcard is not a name and is never reported,
    since a host that declares nothing is not a mistake in the spec.
    """
    path = search_path or mcp_search_path()
    kept: dict[str, dict[str, object]] = {}
    withheld: list[tuple[str, str]] = []
    used: set[str] = set()
    for name, server in servers.items():
        command = commands[name]
        hits = [e for e in allow.entries if _matches(e, name, command, path)]
        used.update(hits)
        if hits:
            kept[name] = server
        else:
            withheld.append((name, command))
    unmatched = tuple(e for e in allow.entries if e not in used and e != WILDCARD)
    return kept, tuple(withheld), unmatched


#: The default: a box that names no server starts none.
DENY_ALL = MCPAllowlist()


@dataclass(frozen=True)
class SessionMCP:
    """A client's local MCP declarations and launcher commands."""

    document: dict[str, object]
    commands: tuple[str, ...]
    kind: Literal["json", "toml"]
    #: Imported server names shown in the launch notice.
    names: tuple[str, ...] = ()
    #: Servers with environment values that may contain host credentials.
    env_names: tuple[str, ...] = ()
    #: Declared but unselected servers, as (name, command), for the notice.
    withheld: tuple[tuple[str, str], ...] = ()
    #: Spec entries that named no declaration on this host.
    unmatched: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.commands)

    def write(self, path: Path) -> None:
        if self.kind == "toml":
            write_toml_document(path, self.document)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Prevent a symlink planted in the writable state directory from
        # redirecting this host-side write and chmod.
        write_sealed(path, json.dumps(self.document, indent=2) + "\n")


def write_toml_document(path: Path, document: dict[str, object]) -> None:
    """Write a TOML document into the box-writable state directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prevent a symlink planted in the writable state directory from
    # redirecting this host-side write and chmod.
    write_sealed(path, _toml_document(document))


def codex_host_mcp(
    path: Path | None = None, *, allow: MCPAllowlist = DENY_ALL
) -> SessionMCP:
    source = path or codex_config_file()
    data = _read_toml(source)
    servers = _table(data.get("mcp_servers"), source, "mcp_servers")
    local = {
        name: server
        for name, server in servers.items()
        if isinstance(server.get("command"), str)
        and server.get("enabled", True) is not False
    }
    commands = {name: str(server["command"]) for name, server in local.items()}
    kept, withheld, unmatched = _select(local, commands, allow)
    return SessionMCP(
        document={"mcp_servers": kept},
        commands=tuple(commands[name] for name in kept),
        kind="toml",
        names=tuple(kept),
        env_names=_env_names(kept, "env"),
        withheld=withheld,
        unmatched=unmatched,
    )


def claude_host_mcp(
    path: Path | None = None, *, allow: MCPAllowlist = DENY_ALL
) -> SessionMCP:
    source = path or claude_config_file()
    data = _read_json(source)
    servers = _table(data.get("mcpServers"), source, "mcpServers")
    local = {
        name: server
        for name, server in servers.items()
        if server.get("type", "stdio") == "stdio"
        and isinstance(server.get("command"), str)
        and server.get("enabled", True) is not False
    }
    commands = {name: str(server["command"]) for name, server in local.items()}
    kept, withheld, unmatched = _select(local, commands, allow)
    return SessionMCP(
        document={"mcpServers": kept},
        commands=tuple(commands[name] for name in kept),
        kind="json",
        names=tuple(kept),
        env_names=_env_names(kept, "env"),
        withheld=withheld,
        unmatched=unmatched,
    )


def opencode_host_mcp(
    path: Path | None = None, *, allow: MCPAllowlist = DENY_ALL
) -> SessionMCP:
    source = path or _opencode_config_file()
    data = _read_jsonc(source)
    servers = _table(data.get("mcp"), source, "mcp")
    local: dict[str, dict[str, object]] = {}
    commands: dict[str, str] = {}
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
        local[name] = server
        commands[name] = command[0]
    kept, withheld, unmatched = _select(local, commands, allow)
    return SessionMCP(
        document={"mcp": kept},
        commands=tuple(commands[name] for name in kept),
        kind="json",
        names=tuple(kept),
        # OpenCode calls this field ``environment``; the other clients use ``env``.
        env_names=_env_names(kept, "environment"),
        withheld=withheld,
        unmatched=unmatched,
    )


def _env_names(
    servers: dict[str, dict[str, object]], key: Literal["env", "environment"]
) -> tuple[str, ...]:
    """Return servers that declare a non-empty environment."""
    return tuple(
        name
        for name, server in servers.items()
        if isinstance(server.get(key), dict) and server[key]
    )


def mcp_search_path(local_bin: bool = True) -> str:
    """Build the PATH used to resolve MCP launchers.

    Include ``~/.local/bin`` only when the box mounts home-installed launchers.
    """
    dirs = ["/usr/bin", "/usr/local/bin"]
    if local_bin:
        dirs.insert(0, str(Path.home() / ".local" / "bin"))
    return os.pathsep.join(dirs)


def mcp_ro_binds(
    config: SessionMCP, search_path: str | None = None
) -> tuple[Path, ...]:
    """Return read-only paths needed by the configured MCP servers."""
    if not config.enabled:
        return ()
    path = search_path or mcp_search_path()
    binds = [Path.home() / ".local" / "bin"]
    for command in config.commands:
        binds.extend(_launcher_binds(command, path))
    out = tuple(dict.fromkeys(binds))
    # Check every known credential store because MCP launchers are independent
    # of the model backend configured for this box.
    hit = credential_overlap(out, known_credential_paths())
    if hit is not None:
        raise ValueError(
            f"MCP launcher bind {hit[0]} would expose the credential store"
            f" {hit[1]}, and is refused"
        )
    return out


# These shared trees contain state for many tools, including credentials such as
# OpenCode's auth.json. Refuse launchers that would require mounting one whole.
def _shared_roots(home: Path) -> frozenset[Path]:
    return frozenset(
        {
            home,
            home / ".local",
            home / ".local" / "share",
            home / ".local" / "state",
            home / ".config",
            home / ".cache",
        }
    )


def _launcher_binds(command: str, search_path: str) -> list[Path]:
    """Resolve a home-installed launcher and its virtualenv interpreter.

    Accept ``<root>/bin/<exe>`` only when ``root`` contains ``pyvenv.cfg`` or
    ``bin/python``. Inferring the root from path depth could mount shared tool
    state and credentials under ``~/.local``.
    """
    executable = shutil.which(command, path=search_path)
    if executable is None:
        raise FileNotFoundError(
            f"MCP command {command!r} is not on the box PATH ({search_path})"
        )
    real = Path(executable).resolve()
    home = Path.home().resolve()
    if not real.is_relative_to(home):
        return []

    if len(real.parents) < 2 or real.parents[1] == home:
        # A script in ~/bin needs only the resolved file.
        return [real]
    root = real.parents[1]
    if (
        root in _shared_roots(home)
        or real.parent.name != "bin"
        or not ((root / "pyvenv.cfg").is_file() or (root / "bin" / "python").exists())
    ):
        raise ValueError(
            f"MCP command {command!r} resolves to {real}, which is not inside a"
            " self-contained tool root; install it into its own venv (e.g."
            " `uv tool install`) so only that tree is bound into the box"
        )
    binds = [root]
    python = root / "bin" / "python"
    if python.is_symlink():
        binds.extend(path for path in interpreter_roots(python) if path not in binds)
    return binds


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def codex_config_file() -> Path:
    """Return the host ``config.toml`` path, honoring ``CODEX_HOME``."""
    return _codex_home() / "config.toml"


def claude_config_file() -> Path:
    """Return the host ``.claude.json`` path, honoring ``CLAUDE_CONFIG_DIR``."""
    return (claude_config_dir() or Path.home()) / ".claude.json"


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
                # An unterminated block comment consumes the rest of the input.
                break
            out.extend("\n" for c in text[index : end + 2] if c == "\n")
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
