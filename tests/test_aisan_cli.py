# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisan.cli
from aisan.cli import main as cli
from aisan.egress.anthropic import DEFAULT_UPSTREAM


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_dispatcher_passes_every_remaining_token_untouched(monkeypatch, command):
    received = []

    def handler(argv):
        received.append(argv)
        return 17

    monkeypatch.setitem(cli.COMMANDS, command, handler)
    remainder = ["repo", "--binds", "x.toml", "--", "--model", "m", "--help"]
    assert cli.main([command, *remainder]) == 17
    assert received == [remainder]


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_each_subcommand_owns_its_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: aisan {command}")


def test_top_level_help_lists_public_commands(capsys):
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "usage: aisan <command>" in output
    assert all(command in output for command in cli.COMMANDS)


@pytest.mark.parametrize(
    "argv, message",
    [([], "usage: aisan <command>"), (["other"], "unknown command: other")],
)
def test_missing_and_unknown_commands_fail(argv, message, capsys):
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_main_accepts_explicit_argv(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    received = []

    async def run(argv):
        received.append(argv)
        return 23

    monkeypatch.setattr(module, "_main", run)
    argv = ["repo", "--", "--model", "m"]
    assert module.main(argv) == 23
    assert received == [argv]


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_parsers_preserve_explain_and_payload(module_name):
    module = importlib.import_module(module_name)
    args, payload = module.parse_args(
        ["repo", "--explain", "--", "--model", "m", "--help"]
    )
    assert args.repo == "repo"
    assert args.explain is True
    assert payload == ["--model", "m", "--help"]


@pytest.mark.parametrize(
    "module_name",
    ["aisan.cli.claude", "aisan.cli.codex", "aisan.cli.opencode"],
)
def test_session_parsers_default_offline_and_accept_net(module_name):
    module = importlib.import_module(module_name)
    default, _ = module.parse_args([])
    enabled, _ = module.parse_args(["--net"])
    assert default.net is False
    assert enabled.net is True


def test_claude_defaults_to_the_anthropic_api(monkeypatch):
    monkeypatch.delenv("AISAN_UPSTREAM", raising=False)
    module = importlib.import_module("aisan.cli.claude")
    args, payload = module.parse_args([])
    assert args.upstream == DEFAULT_UPSTREAM
    assert payload == []


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [("aisan.cli.claude", "CLAUDE.md"), ("aisan.cli.codex", "AGENTS.md")],
)
def test_user_memory_lands_where_the_cli_reads_it(module_name, filename, tmp_path):
    """The state dir is the box's config dir (CLAUDE_CONFIG_DIR, CODEX_HOME),
    which is where each client reads user memory -- a mirror anywhere else, the
    host path included, is a file the box never opens."""
    module = importlib.import_module(module_name)
    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "source.md"
    source.write_text("host memory\n")

    module.seed_user_memory(state, source)
    assert (state / filename).read_text() == "host memory\n"

    # The host file is the source of truth: an in-box edit lasts one session.
    (state / filename).write_text("agent memory\n")
    source.write_text("host memory, edited\n")
    module.seed_user_memory(state, source)
    assert (state / filename).read_text() == "host memory, edited\n"


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [("aisan.cli.claude", "CLAUDE.md"), ("aisan.cli.codex", "AGENTS.md")],
)
def test_a_host_without_user_memory_leaves_the_state_dir_alone(
    module_name, filename, tmp_path
):
    """Nothing here can tell a stale mirror from memory the agent wrote, so a
    missing source seeds nothing rather than deleting the operator's file."""
    module = importlib.import_module(module_name)
    state = tmp_path / "state"
    state.mkdir()

    module.seed_user_memory(state, tmp_path / "absent.md")
    assert not (state / filename).exists()

    (state / filename).write_text("agent memory\n")
    module.seed_user_memory(state, tmp_path / "absent.md")
    assert (state / filename).read_text() == "agent memory\n"


@pytest.mark.parametrize(
    "malformed",
    [
        "not json at all",
        '{"projects": "a string"}',
        '{"customApiKeyResponses": [1, 2]}',
        "[1, 2, 3]",
        '{"projects": {"/repo": "not an object"}}',
    ],
)
def test_seed_state_rebuilds_malformed_state_instead_of_crashing(tmp_path, malformed):
    """The .claude.json lives in the box-writable state dir, so its contents are
    attacker-reachable between sessions. A non-JSON, non-object or wrong-typed
    nested value used to crash seed_state and wedge every later `aisan claude`
    on this repo until the operator deleted the cache. It must be repaired."""
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    (state / ".claude.json").write_text(malformed)

    seed_state(state, Path("/repo"))

    config = json.loads((state / ".claude.json").read_text())
    assert config["hasCompletedOnboarding"] is True
    assert config["bypassPermissionsModeAccepted"] is True
    assert isinstance(config["customApiKeyResponses"]["approved"], list)
    assert config["projects"]["/repo"]["hasTrustDialogAccepted"] is True


def test_seed_state_preserves_unrelated_valid_state(tmp_path):
    """A tolerant rebuild only repairs what is malformed: valid, unrelated keys
    the CLI wrote survive the reseed."""
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    (state / ".claude.json").write_text(
        json.dumps({"userID": "keep-me", "projects": {"/other": {"seen": True}}})
    )

    seed_state(state, Path("/repo"))

    config = json.loads((state / ".claude.json").read_text())
    assert config["userID"] == "keep-me"
    assert config["projects"]["/other"] == {"seen": True}
    assert config["projects"]["/repo"]["hasTrustDialogAccepted"] is True


def test_seed_state_lends_the_box_the_hosts_model_menu(tmp_path):
    """The /model picker's row for a model outside the CLI's compiled-in catalog
    comes from `additionalModelOptionsCache`, which the CLI fills from a fetch
    the box cannot make: it goes to the compiled-in api.anthropic.com rather than
    through ANTHROPIC_BASE_URL, so no proxy allowlist reaches it. Without the
    host's copy the picker is short those rows while `--model <alias>` still
    works, which is exactly the reported asymmetry."""
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    host = tmp_path / ".claude.json"
    options = [
        {"value": "claude-fable-5", "label": "Fable", "description": "Fable 5"},
        {
            "value": "claude-mythos-5",
            "label": "Mythos",
            "description": "",
            "disabled": True,
        },
    ]
    host.write_text(json.dumps({"additionalModelOptionsCache": options}))

    seed_state(state, Path("/repo"), host)

    config = json.loads((state / ".claude.json").read_text())
    # Verbatim, the disabled entry included: an entitlement the host does not
    # have must not become one the box appears to.
    assert config["additionalModelOptionsCache"] == options


@pytest.mark.parametrize(
    "host_state",
    ["absent", "not json at all", "[1, 2, 3]", '{"additionalModelOptionsCache": "no"}'],
)
def test_seed_state_keeps_the_boxs_model_menu_when_the_host_lends_none(
    tmp_path, host_state
):
    """A host that has never run Claude Code, or whose config is unreadable or
    wrong-shaped, leaves the cached menu alone rather than clearing it: a stale
    menu is a worse failure than none only if it is wrong, and an emptied one is
    wrong every time."""
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    kept = [{"value": "claude-fable-5", "label": "Fable", "description": "Fable 5"}]
    (state / ".claude.json").write_text(
        json.dumps({"additionalModelOptionsCache": kept})
    )
    host = tmp_path / ".claude.json"
    if host_state != "absent":
        host.write_text(host_state)

    seed_state(state, Path("/repo"), host)

    config = json.loads((state / ".claude.json").read_text())
    assert config["additionalModelOptionsCache"] == kept


def test_the_seeded_sources_are_the_hosts_own_user_memory():
    """The constants the launchers seed FROM, named once so a rename of either
    host file does not silently turn seeding into a no-op."""
    claude = importlib.import_module("aisan.cli.claude")
    codex = importlib.import_module("aisan.cli.codex")
    assert Path.home() / ".claude" / "CLAUDE.md" == claude.USER_MEMORY
    assert Path.home() / ".claude.json" == claude.HOST_CONFIG
    assert Path.home() / ".codex" / "AGENTS.md" == codex.USER_MEMORY


def test_opencode_mounts_user_memory_where_the_box_reads_it(tmp_path):
    """opencode's config root is the HOME tmpfs, so its global AGENTS.md is a
    bind, and the destination is the box's path rather than the host's -- the
    two differ exactly when the host sets XDG_CONFIG_HOME."""
    opencode = importlib.import_module("aisan.cli.opencode")
    source = tmp_path / "AGENTS.md"
    source.write_text("host memory\n")

    (bind,) = opencode.user_memory_bind(source)
    assert bind.src == source
    assert bind.dst == Path.home() / ".config" / "opencode" / "AGENTS.md"

    assert opencode.user_memory_bind(tmp_path / "absent.md") == []


def plugin_entry_point(name, target="aisan.cli:handler_under_test", dist="aisan-corp"):
    """An `aisan.commands` entry point as the environment would declare it.

    A real EntryPoint, loaded through real import machinery, so the dispatch
    under test is the one an installed plugin gets. EntryPoint is immutable and
    `_for` is how importlib.metadata itself attaches the providing distribution.
    """
    ep = importlib.metadata.EntryPoint(name=name, value=target, group=cli.PLUGIN_GROUP)
    return ep if dist is None else ep._for(SimpleNamespace(name=dist))


@pytest.fixture
def plugin_handler(monkeypatch):
    """A loadable target for `plugin_entry_point`, recording the argv it got."""
    received = []

    def handler(argv):
        received.append(argv)
        return 19

    monkeypatch.setattr(aisan.cli, "handler_under_test", handler, raising=False)
    return received


def test_a_plugin_command_dispatches_with_its_tokens_untouched(
    monkeypatch, plugin_handler
):
    monkeypatch.setattr(
        cli, "_installed_entry_points", lambda: [plugin_entry_point("jetski")]
    )
    remainder = ["repo", "--binds", "x.toml", "--", "--model", "m", "--help"]

    assert cli.main(["jetski", *remainder]) == 19
    assert plugin_handler == [remainder]


def test_builtin_dispatch_never_looks_at_installed_plugins(monkeypatch):
    """The core keeps the dispatch it had before plugins existed: `aisan claude`
    neither scans distribution metadata nor imports a plugin.

    Counted, not raised: discovery swallows exceptions by design, so a seam that
    failed loudly would be reported as unreadable metadata and prove nothing.
    """
    scans = []

    def scan():
        scans.append(True)
        return []

    monkeypatch.setattr(cli, "_installed_entry_points", scan)
    monkeypatch.setitem(cli.COMMANDS, "claude", lambda argv: 17)
    assert cli.main(["claude", "repo"]) == 17
    assert scans == []


def test_a_plugin_cannot_take_over_a_builtin_command(monkeypatch, capsys):
    """Installing a plugin installs its whole dependency closure, and any
    distribution in it can register here. `aisan claude` stays this package's."""
    monkeypatch.setattr(
        cli,
        "_installed_entry_points",
        lambda: [plugin_entry_point("claude", dist="impostor")],
    )
    monkeypatch.setattr(cli, "handler_under_test", None, raising=False)

    plugins, refused = cli.plugin_commands()
    assert plugins == {}
    assert refused == ["impostor cannot override built-in command 'claude'"]

    # And the refusal is reported rather than presenting as a missing command.
    assert cli.main(["--help"]) == 0
    assert "impostor cannot override built-in command" in capsys.readouterr().err


@pytest.mark.parametrize("order", [("a-corp", "b-corp"), ("b-corp", "a-corp")])
def test_one_name_claimed_twice_resolves_independently_of_scan_order(
    monkeypatch, order
):
    monkeypatch.setattr(
        cli,
        "_installed_entry_points",
        lambda: [plugin_entry_point("jetski", dist=dist) for dist in order],
    )

    plugins, refused = cli.plugin_commands()
    assert cli._origin(plugins["jetski"]) == "a-corp"
    assert refused == ["b-corp lost command 'jetski' to a-corp"]


def test_a_plugin_that_cannot_be_imported_leaves_the_builtins_working(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        cli,
        "_installed_entry_points",
        lambda: [plugin_entry_point("jetski", target="aisan_corp_absent.cli:main")],
    )

    assert cli.main(["jetski"]) == 2
    error = capsys.readouterr().err
    assert "plugin command 'jetski' from aisan-corp failed to load" in error
    assert "aisan_corp_absent" in error

    monkeypatch.setitem(cli.COMMANDS, "claude", lambda argv: 17)
    assert cli.main(["claude", "repo"]) == 17


def test_unreadable_plugin_metadata_degrades_to_the_builtins(monkeypatch, capsys):
    def broken():
        raise ValueError("bad metadata")

    monkeypatch.setattr(cli, "_installed_entry_points", broken)

    assert cli.main(["--help"]) == 0
    assert "cannot read plugin commands: bad metadata" in capsys.readouterr().err


def test_help_lists_plugin_commands_with_the_package_providing_them(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "_installed_entry_points", lambda: [plugin_entry_point("jetski")]
    )

    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "plugin commands:\n  jetski    (aisan-corp)\n" in output
    assert all(command in output for command in cli.COMMANDS)


def test_help_without_plugins_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_installed_entry_points", list)

    assert cli.main(["--help"]) == 0
    assert "plugin commands" not in capsys.readouterr().out
