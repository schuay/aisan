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
    module = importlib.import_module(module_name)
    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "source.md"
    source.write_text("host memory\n")

    module.seed_user_memory(state, source)
    assert (state / filename).read_text() == "host memory\n"

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
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    (state / ".claude.json").write_text(malformed)

    seed_state(state)

    config = json.loads((state / ".claude.json").read_text())
    assert config["hasCompletedOnboarding"] is True
    assert isinstance(config["customApiKeyResponses"]["approved"], list)


def test_seed_state_preserves_unrelated_valid_state(tmp_path):
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    (state / ".claude.json").write_text(
        json.dumps({"userID": "keep-me", "projects": {"/other": {"seen": True}}})
    )

    seed_state(state)

    config = json.loads((state / ".claude.json").read_text())
    assert config["userID"] == "keep-me"
    assert config["projects"]["/other"] == {"seen": True}


def test_seed_state_lends_the_box_the_hosts_model_menu(tmp_path):
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

    seed_state(state, host)

    config = json.loads((state / ".claude.json").read_text())

    assert config["additionalModelOptionsCache"] == options


def test_seed_state_lends_the_box_the_hosts_feature_flags(tmp_path):
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    host = tmp_path / ".claude.json"
    features = {"tengu_saffron_lattice": {"enabled": False}}
    data = {"tengu_copper_fox": {"experimentId": "x", "variationId": 0}}
    host.write_text(
        json.dumps(
            {
                "cachedGrowthBookFeatures": features,
                "cachedGrowthBookFeaturesAt": 1788580762611,
                "cachedExperimentFeatures": ["tengu_copper_fox"],
                "cachedExperimentData": data,
            }
        )
    )
    (state / ".claude.json").write_text(
        json.dumps(
            {
                "cachedGrowthBookFeatures": {
                    "tengu_saffron_lattice": {
                        "enabled": True,
                        "planLimitsEndDate": "2026-07-20T07:00:00Z",
                    }
                },
                "cachedGrowthBookFeaturesAt": 1788079074316,
            }
        )
    )

    seed_state(state, host)

    config = json.loads((state / ".claude.json").read_text())
    assert config["cachedGrowthBookFeatures"] == features
    assert config["cachedGrowthBookFeaturesAt"] == 1788580762611
    assert config["cachedExperimentFeatures"] == ["tengu_copper_fox"]

    assert config["cachedExperimentData"] == data


def test_seed_state_mirrors_each_cache_on_its_own(tmp_path):
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    kept = [{"value": "claude-fable-5", "label": "Fable", "description": ""}]
    (state / ".claude.json").write_text(
        json.dumps({"additionalModelOptionsCache": kept})
    )
    host = tmp_path / ".claude.json"
    features = {"tengu_saffron_lattice": {"enabled": False}}
    host.write_text(
        json.dumps(
            {
                "additionalModelOptionsCache": "not a menu",
                "cachedGrowthBookFeatures": features,
                "cachedGrowthBookFeaturesAt": True,
            }
        )
    )

    seed_state(state, host)

    config = json.loads((state / ".claude.json").read_text())
    assert config["additionalModelOptionsCache"] == kept
    assert config["cachedGrowthBookFeatures"] == features
    assert "cachedGrowthBookFeaturesAt" not in config


@pytest.mark.parametrize(
    "host_state",
    ["absent", "not json at all", "[1, 2, 3]", '{"additionalModelOptionsCache": "no"}'],
)
def test_seed_state_keeps_the_boxs_model_menu_when_the_host_lends_none(
    tmp_path, host_state
):
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

    seed_state(state, host)

    config = json.loads((state / ".claude.json").read_text())
    assert config["additionalModelOptionsCache"] == kept


def test_the_model_menu_is_read_from_the_hosts_real_config_dir(tmp_path, monkeypatch):
    import json

    from aisan.cli.claude import host_config, seed_state

    configured = tmp_path / "elsewhere"
    configured.mkdir()
    options = [{"value": "claude-fable-5", "label": "Fable", "description": "Fable 5"}]
    (configured / ".claude.json").write_text(
        json.dumps({"additionalModelOptionsCache": options})
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(configured))

    assert host_config() == configured / ".claude.json"

    state = tmp_path / "state"
    state.mkdir()
    seed_state(state)

    config = json.loads((state / ".claude.json").read_text())
    assert config["additionalModelOptionsCache"] == options


def test_the_seeded_sources_are_the_hosts_own_user_memory():
    claude = importlib.import_module("aisan.cli.claude")
    codex = importlib.import_module("aisan.cli.codex")
    assert Path.home() / ".claude" / "CLAUDE.md" == claude.USER_MEMORY
    assert Path.home() / ".codex" / "AGENTS.md" == codex.USER_MEMORY


def test_claude_takes_its_skills_from_the_hosts_own_directory(tmp_path):
    claude = importlib.import_module("aisan.cli.claude")
    assert Path.home() / ".claude" / "skills" == claude.USER_SKILLS

    skills = tmp_path / "skills"
    skills.mkdir()
    assert claude.user_skills(skills) == skills
    assert claude.user_skills(tmp_path / "absent") is None
    # A file at the skills path would make the mandatory bind-over fail.
    plain = tmp_path / "plain"
    plain.write_text("")
    assert claude.user_skills(plain) is None


def test_opencode_mounts_user_memory_where_the_box_reads_it(tmp_path):
    opencode = importlib.import_module("aisan.cli.opencode")
    source = tmp_path / "AGENTS.md"
    source.write_text("host memory\n")

    (bind,) = opencode.user_memory_bind(source)
    assert bind.src == source
    assert bind.dst == Path.home() / ".config" / "opencode" / "AGENTS.md"

    assert opencode.user_memory_bind(tmp_path / "absent.md") == []


def plugin_entry_point(name, target="aisan.cli:handler_under_test", dist="aisan-corp"):
    ep = importlib.metadata.EntryPoint(name=name, value=target, group=cli.PLUGIN_GROUP)
    return ep if dist is None else ep._for(SimpleNamespace(name=dist))


@pytest.fixture
def plugin_handler(monkeypatch):
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
    scans = []

    def scan():
        scans.append(True)
        return []

    monkeypatch.setattr(cli, "_installed_entry_points", scan)
    monkeypatch.setitem(cli.COMMANDS, "claude", lambda argv: 17)
    assert cli.main(["claude", "repo"]) == 17
    assert scans == []


def test_a_plugin_cannot_take_over_a_builtin_command(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_entry_points",
        lambda: [plugin_entry_point("claude", dist="impostor")],
    )
    monkeypatch.setattr(cli, "handler_under_test", None, raising=False)

    plugins, refused = cli.plugin_commands()
    assert plugins == {}
    assert refused == ["impostor cannot override built-in command 'claude'"]

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


def test_seed_state_never_answers_the_folder_trust_prompt(tmp_path):
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()

    seed_state(state)

    config = json.loads((state / ".claude.json").read_text())
    assert "hasTrustDialogAccepted" not in json.dumps(config)


def test_seed_state_keeps_the_trust_answer_given_in_the_box(tmp_path):
    import json

    from aisan.cli.claude import seed_state

    state = tmp_path / "state"
    state.mkdir()
    (state / ".claude.json").write_text(
        json.dumps({"projects": {"/repo": {"hasTrustDialogAccepted": True}}})
    )

    seed_state(state)

    config = json.loads((state / ".claude.json").read_text())
    assert config["projects"]["/repo"]["hasTrustDialogAccepted"] is True


def test_seed_settings_accepts_the_disclaimer_where_the_cli_reads_it(tmp_path):
    import json

    from aisan.cli.claude import seed_settings, seed_state

    state = tmp_path / "state"
    state.mkdir()

    seed_state(state)
    seed_settings(state)

    settings = json.loads((state / "settings.json").read_text())
    assert settings["skipDangerousModePermissionPrompt"] is True
    config = json.loads((state / ".claude.json").read_text())
    assert "bypassPermissionsModeAccepted" not in config


def test_seed_settings_keeps_the_settings_the_cli_wrote(tmp_path):
    import json

    from aisan.cli.claude import seed_settings

    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.json").write_text(
        json.dumps({"statusLine": {"type": "command"}})
    )

    seed_settings(state)

    settings = json.loads((state / "settings.json").read_text())
    assert settings["statusLine"] == {"type": "command"}
    assert settings["skipDangerousModePermissionPrompt"] is True


@pytest.mark.parametrize("malformed", ["not json", "[1, 2, 3]", '"a string"'])
def test_seed_settings_rebuilds_a_mangled_file_instead_of_crashing(tmp_path, malformed):
    import json

    from aisan.cli.claude import seed_settings

    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.json").write_text(malformed)

    seed_settings(state)

    settings = json.loads((state / "settings.json").read_text())
    assert settings == {"skipDangerousModePermissionPrompt": True}


def test_seed_settings_reads_nothing_through_a_planted_symlink(tmp_path):
    import json

    from aisan.cli.claude import seed_settings

    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "host-settings.json"
    victim.write_text(json.dumps({"plantedFromHost": True}))
    (state / "settings.json").symlink_to(victim)

    seed_settings(state)

    settings = json.loads((state / "settings.json").read_text())
    assert settings == {"skipDangerousModePermissionPrompt": True}
    assert json.loads(victim.read_text()) == {"plantedFromHost": True}
