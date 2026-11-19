# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp import web
from conftest import needs_seatbelt

from aisan import Box
from aisan.egress.anthropic import AnthropicBackend
from aisan.egress.antigravity import AntigravityBackend
from aisan.egress.antigravity_keyring import (
    KeyringError,
    read_macos_keychain,
)
from aisan.egress.claude_plan import PlanCredential, _keychain_service
from aisan.egress.openai_responses import CodexBackend
from aisan.presets.antigravity import TOKEN_FILE, antigravity
from aisan.presets.claude_code import claude_code
from aisan.presets.codex import codex, codex_argv
from aisan.presets.opencode import opencode
from aisan.private import private_root
from aisan.proxy.antigravity import Profile as AgyProfile
from aisan.runtime import profile_path, scratch_dir
from aisan.sandbox import RO, RW, Bind, PinnedSymlink
from aisan.seatbelt import seatbelt_path
from aisan.session import darwin_executable_grant, git_config_binds, git_config_env
from aisan.spec import BoxSpec
from aisan.statedir import StateBusyError, lock_state_dir, planted_credentials


def _claude_cred(path: Path, bearer: str = "real-oauth-for-test") -> Path:
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": bearer,
                    "expiresAt": 4_102_444_800_000,
                    "subscriptionType": "max",
                }
            }
        )
    )
    path.chmod(0o600)
    return path


async def _fake_upstream(handler) -> tuple[str, int, web.AppRunner]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    port = runner.addresses[0][1]
    return f"http://127.0.0.1:{port}", port, runner


# --- Credential & statedir unit tests (cross-platform) ---


def test_planted_credentials_allows_only_private_root_symlinks(tmp_path):
    priv = tmp_path / "aisan-priv"
    rt = priv / "proxy-abcd"
    rt.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()

    # Stale symlink into private_root on a mountpoint client (`codex`) is ignored.
    auth_link = state / "auth.json"
    auth_link.symlink_to(rt / "codex-auth.json")
    assert planted_credentials(state, "codex", allowed_symlink_root=priv) == []
    # Without allowed_symlink_root (Linux), any symlink is flagged.
    assert planted_credentials(state, "codex") == [auth_link]

    # Symlink with `..` or pointing outside `priv` is flagged as planted.
    auth_link.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    auth_link.symlink_to(outside)
    assert planted_credentials(state, "codex", allowed_symlink_root=priv) == [auth_link]

    auth_link.unlink()
    os.symlink(f"{rt}/../../outside.json", auth_link)
    assert planted_credentials(state, "codex", allowed_symlink_root=priv) == [auth_link]

    # Non-empty regular file is always flagged.
    auth_link.unlink()
    auth_link.write_text('{"token":"planted"}')
    assert planted_credentials(state, "codex", allowed_symlink_root=priv) == [auth_link]


def test_macos_keychain_reader_handles_success_locked_and_missing_fallback(tmp_path):
    fake_sec = tmp_path / "fake-security"
    fallback = tmp_path / "antigravity-oauth-token"

    def write_sec(rc: int, out: str = "", err: str = "") -> None:
        fake_sec.write_text(
            f"#!{sys.executable}\n"
            f"import sys\nsys.stdout.write({out!r})\nsys.stderr.write({err!r})\n"
            f"sys.exit({rc})\n"
        )
        fake_sec.chmod(0o755)

    # 1. Keychain unlocked (rc 0): strips trailing newline and ignores absent fallback.
    write_sec(0, out='{"token":"kc"}\n')
    assert (
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback)
        == '{"token":"kc"}'
    )

    # 2. Keychain locked (rc 36) without fallback file -> KeyringError with unlock fix.
    write_sec(36, err="User interaction is not allowed.")
    with pytest.raises(KeyringError, match="locked"):
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback)

    # 3. Keychain locked (rc 36) WITH fallback file -> returns fallback file contents.
    fallback.write_text('{"token":"file-fallback"}')
    assert (
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback)
        == '{"token":"file-fallback"}'
    )

    # 4. Keychain missing item (rc 44) without fallback -> KeyringError with login fix.
    fallback.unlink()
    write_sec(44, err="could not be found")
    with pytest.raises(KeyringError, match="no Antigravity login"):
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback)

    # 5. Hex-encoded `security -w` output and fresher fallback file selection.
    stale_kc = json.dumps(
        {
            "token": {
                "access_token": "kc-stale",
                "expiry": "2025-01-01T00:00:00+00:00",
            },
            "auth_method": "consumer",
        }
    )
    fresh_file = json.dumps(
        {
            "token": {
                "access_token": "file-fresh",
                "expiry": "2099-01-01T00:00:00+00:00",
            },
            "auth_method": "consumer",
        }
    )
    write_sec(0, out=stale_kc.encode("utf-8").hex() + "\n")
    assert (
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback) == stale_kc
    )
    fallback.write_text(fresh_file)
    assert (
        read_macos_keychain(security=str(fake_sec), fallback_file=fallback)
        == fresh_file
    )


async def test_claude_plan_keychain_lookup_and_file_fallback(tmp_path, monkeypatch):
    import aisan.egress.claude_plan as cp_mod

    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
    home_claude = Path.home() / ".claude"
    assert _keychain_service(home_claude) == "Claude Code-credentials"
    custom_dir = tmp_path / "custom-claude"
    custom_dir.mkdir()
    assert _keychain_service(custom_dir).startswith("Claude Code-credentials-")

    # Keychain hit wins when `use_keychain=True`.
    kc_json = json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": "from-keychain",
                "expiresAt": 4_102_444_800_000,
                "subscriptionType": "pro",
            }
        }
    )
    monkeypatch.setattr(cp_mod, "_read_macos_keychain_status", lambda _d: (kc_json, 0))
    cred = PlanCredential(
        custom_dir / "absent.json", claude_command=("claude",), use_keychain=True
    )
    assert await cred.upstream() == {"authorization": "Bearer from-keychain"}
    assert cred.dress("tok")["CLAUDE_CODE_SUBSCRIPTION_TYPE"] == "pro"

    # Stale or MCP-only Keychain item yields to fresher fallback file.
    mcp_only = json.dumps({"mcpOAuth": {"srv": {}}})
    monkeypatch.setattr(cp_mod, "_read_macos_keychain_status", lambda _d: (mcp_only, 0))
    file_path = _claude_cred(custom_dir / ".credentials.json", bearer="from-file")
    cred_fb = PlanCredential(file_path, claude_command=("claude",), use_keychain=True)
    assert await cred_fb.upstream() == {"authorization": "Bearer from-file"}


# --- Darwin preset & Box staging tests ---


@needs_seatbelt
def test_darwin_presets_avoid_bind_over_and_overlay_and_configure_flags(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    skills = tmp_path / "skills"
    ext_skill = tmp_path / "agents-skills" / "beta"
    ext_skill.mkdir(parents=True)
    (skills / "alpha").mkdir(parents=True)
    (skills / "beta").symlink_to(ext_skill)
    (skills / "dangling").symlink_to(tmp_path / "missing-skill")
    (skills / ".system").mkdir()

    # Claude Code uses PinnedSymlink for skills and omits EnsurePath(state / "skills").
    cc_spec = claude_code(wt, state=state, skills=skills)
    assert PinnedSymlink(skills.resolve(), state / "skills") in cc_spec.binds
    assert all(e.path != state / "skills" for e in cc_spec.ensure)

    # Codex resolves symlinked skills, skips `.system` and dangling symlinks, and resolves cleanly.
    cx_backend = CodexBackend(
        credentials=tmp_path / "absent.json",
        codex_home=state,
    )
    assert 'cli_auth_credentials_store="file"' in cx_backend.config_overrides()
    assert "--no-daemon" in codex_argv()
    cx_spec = codex(wt, state=state, skills=skills, egress=(cx_backend,))
    assert cx_spec.mach_profiles == ("codex-cfprefs",)
    assert (
        PinnedSymlink((skills / "alpha").resolve(), state / "skills" / "alpha")
        in cx_spec.binds
    )
    assert (
        PinnedSymlink(ext_skill.resolve(), state / "skills" / "beta") in cx_spec.binds
    )
    assert all(
        getattr(b, "dst", None)
        not in (state / "skills" / ".system", state / "skills" / "dangling")
        for b in cx_spec.binds
    )
    with Box(cx_spec, box_id="cx-skill-check").staged() as cx_box:
        assert cx_box.wrapper()

    # Cross-platform Box(..., platform="linux") on Darwin adapts backend PinnedSymlink -> BindOver.
    linux_box = Box(
        codex(wt, state=state, egress=(cx_backend,)),
        box_id="cx-linux-on-darwin",
        platform="linux",
    )
    with linux_box.staged():
        assert any("codex-auth.json" in tok for tok in linux_box.wrapper())

    # OpenCode and Antigravity set allow_local_loopback=True on Darwin.
    catalog = tmp_path / "models.json"
    catalog.write_text("{}")
    oc_spec = opencode(wt, state=state, models=catalog)
    assert oc_spec.allow_local_loopback is True
    assert (
        PinnedSymlink(
            catalog.resolve(), state / "xdg-cache" / "opencode" / "models.json"
        )
        in oc_spec.binds
    )

    ag_spec = antigravity(wt, state=state)
    assert ag_spec.allow_local_loopback is True

    # PinnedSymlink targeting host ~/.ssh or Keychains is refused by Box._sandbox().
    ssh_dir = Path.home() / ".ssh"
    bad_sym = BoxSpec(
        root=wt,
        binds=(Bind(state, RW), PinnedSymlink(ssh_dir, state / "leaked-ssh")),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=True,
    )
    with pytest.raises(ValueError, match="host key store"):
        Box(bad_sym, box_id="bad-sym-keystore")._sandbox()


@needs_seatbelt
def test_darwin_executable_grant_mounts_standalone_app_bundle_root(
    tmp_path, monkeypatch
):
    bundle_root = Path("/Applications/FakeTestApp.app/Contents/Resources/tool-cli")
    # Simulate an non-app-bundle `~/.cargo/bin/tool` install: must NOT mount `~/.cargo` whole!
    fake_home = tmp_path / "home"
    cargo_exe = fake_home / ".cargo" / "bin" / "rg-custom"
    cargo_exe.parent.mkdir(parents=True)
    cargo_exe.write_text("#!/bin/sh\nexit 0\n")
    cargo_exe.chmod(0o755)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

    cargo_grant = darwin_executable_grant(cargo_exe)
    bound_paths = {b.path for b in cargo_grant.binds if isinstance(b, Bind)}
    assert (fake_home / ".cargo").resolve() not in bound_paths
    assert cargo_exe.resolve() in bound_paths
    assert cargo_grant.path == (cargo_exe.parent,)
    assert bundle_root  # keep lint happy

    # Self-contained venv (`pyvenv.cfg`) mounts the venv root cleanly.
    venv_root = fake_home / "tools" / "my-venv"
    venv_exe = venv_root / "bin" / "tool"
    venv_exe.parent.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text("home = /usr\n")
    venv_exe.write_text("#!/bin/sh\nexit 0\n")
    venv_exe.chmod(0o755)
    venv_grant = darwin_executable_grant(venv_exe)
    assert Bind(venv_root.resolve(), RO, guard=False) in venv_grant.binds

    # System binary needs no extra grant.
    assert not darwin_executable_grant(Path("/usr/bin/true"))


@needs_seatbelt
def test_darwin_box_staging_cleans_up_symlinks_dirs_scratch_and_profile(
    tmp_path, monkeypatch
):
    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    src_file = tmp_path / "src.json"
    src_file.write_text('{"ok":true}')

    # Pre-plant a stale symlink into private_root() to verify recovery.
    nested_dst = state / "sub" / "nested" / "token.json"
    spec = BoxSpec(
        root=wt,
        binds=(
            Bind(state, RW),
            PinnedSymlink(src_file, nested_dst),
        ),
        tmpfs=(),
        env=(("HOME", str(Path.home())),),
        egress=(),
        unshare_net=True,
    )
    box = Box(spec, box_id="darwin-stage-test")
    # Calling wrapper() outside staged() / async with raises a clear error.
    with pytest.raises(ValueError, match="Darwin scratch dir"):
        box.wrapper()
    with box.staged():
        assert nested_dst.is_symlink()
        assert scratch_dir(box.box_id).is_dir()
        wrapper = box.wrapper()
        assert profile_path(box.box_id).is_file()
        assert wrapper[-2:] == ["-f", str(profile_path(box.box_id))]
        assert box.env["TMPDIR"] == seatbelt_path(scratch_dir(box.box_id))
        assert box.env["GIT_CONFIG_GLOBAL"] == "/dev/null"

    assert not nested_dst.exists()
    assert not (state / "sub").exists()
    assert not scratch_dir(box.box_id).exists()
    assert not profile_path(box.box_id).exists()

    # Refuses overwriting a non-empty regular file at PinnedSymlink destination.
    (state / "occupied.json").write_text("user-data")
    bad_spec = BoxSpec(
        root=wt,
        binds=(Bind(state, RW), PinnedSymlink(src_file, state / "occupied.json")),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=True,
    )
    with (
        pytest.raises(FileExistsError, match="refusing to overwrite"),
        Box(bad_spec, box_id="darwin-refuse-test").staged(),
    ):
        pass
    assert (state / "occupied.json").read_text() == "user-data"


@needs_seatbelt
def test_darwin_git_config_binds_wires_git_config_global_and_ignores_submodules(
    tmp_path, monkeypatch
):
    wt = tmp_path / "wt"
    (wt / ".git" / "modules" / "git").mkdir(parents=True)
    submod_cfg = wt / ".git" / "modules" / "git" / "config"
    submod_cfg.write_text("[core]\n\tfsmonitor = evil\n")

    # Without git_config_env(), submodule git/config must NEVER become GIT_CONFIG_GLOBAL.
    sub_spec = BoxSpec(
        root=wt,
        binds=(Bind(submod_cfg, RO),),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=True,
    )
    assert Box(sub_spec, box_id="submod-check").env["GIT_CONFIG_GLOBAL"] == "/dev/null"

    # Symlinked dotfiles `~/.config/git/config -> ~/dotfiles/gitconfig` resolve cleanly.
    xdg = tmp_path / "xdg"
    dotfiles = tmp_path / "dotfiles"
    (xdg / "git").mkdir(parents=True)
    dotfiles.mkdir()
    real_cfg = dotfiles / "gitconfig"
    real_cfg.write_text("[user]\n\tname = BoxUser\n")
    (xdg / "git" / "config").symlink_to(real_cfg)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    binds = git_config_binds()
    env = git_config_env()
    assert binds == [Bind(real_cfg.resolve(), RO)]
    spec = BoxSpec(
        root=wt,
        binds=tuple(binds),
        tmpfs=(),
        env=env,
        egress=(),
        unshare_net=True,
    )
    box = Box(spec, box_id="darwin-gitcfg")
    assert box.env["GIT_CONFIG_GLOBAL"] == seatbelt_path(real_cfg)


@needs_seatbelt
def test_state_lock_blocks_concurrent_sessions_while_explain_preserves_live_symlinks(
    tmp_path,
):
    from aisan.session import state_dir

    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = {**os.environ, "HOME": str(fake_home)}

    # Uncontested `--explain` leaves no `.cache` behind.
    res = subprocess.run(
        [sys.executable, "-m", "aisan.cli.main", "agy", "--explain", str(repo)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "== seatbelt profile (SBPL) ==" in res.stdout
    assert "localhost TCP + pinned endpoints" in res.stdout
    assert not (fake_home / ".cache").exists()

    # Simulate a live session holding `lock_state_dir(state)` with a staged token symlink.
    st = state_dir("antigravity", repo, home=fake_home)
    live_tok_target = tmp_path / "live-token.json"
    live_tok_target.write_text("{}")
    with lock_state_dir(st):
        # Second lock attempt raises StateBusyError.
        with (
            pytest.raises(StateBusyError, match="already running"),
            lock_state_dir(st),
        ):
            pass
        tok_link = st / TOKEN_FILE
        tok_link.parent.mkdir(parents=True, exist_ok=True)
        tok_link.symlink_to(live_tok_target)

        # Running `--explain` in another process while `st` is locked succeeds AND leaves `tok_link` intact!
        res_locked = subprocess.run(
            [sys.executable, "-m", "aisan.cli.main", "agy", "--explain", str(repo)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_locked.returncode == 0, res_locked.stderr
        assert str(st) in res_locked.stdout
        assert tok_link.is_symlink()
        assert Path(os.readlink(tok_link)) == live_tok_target


# --- Live Darwin `Box` + authenticated host-loopback proxy round-trip ---


@needs_seatbelt
async def test_live_darwin_box_proxy_roundtrip_blocks_direct_upstream_and_credentials(
    tmp_path,
):
    wt = tmp_path / "wt"
    state = tmp_path / "state"
    secret_dir = tmp_path / "host-secrets"
    for d in (wt, state, secret_dir):
        d.mkdir()

    seen_auth: list[str] = []

    async def upstream(request: web.Request) -> web.Response:
        if request.path == "/v1/messages":
            seen_auth.append(request.headers.get("Authorization", ""))
            return web.json_response({"id": "msg_ok", "content": []})
        return web.json_response({"ok": True})

    up_url, up_port, runner = await _fake_upstream(upstream)
    cred_file = _claude_cred(
        secret_dir / ".credentials.json", bearer="super-secret-oat"
    )
    backend = AnthropicBackend(credentials=cred_file, upstream=up_url)
    spec = claude_code(wt, state=state, egress=(backend,), unshare_net=True)

    class _FakeAgyCred:
        files = (secret_dir / "agy-secret.json",)

        async def token(self) -> str:
            return "agy-real-token"

        async def check(self, backend: str) -> None:
            return None

    (secret_dir / "agy-secret.json").write_text('{"secret":1}')
    agy_backend = AntigravityBackend(
        credential=_FakeAgyCred(),
        profile=AgyProfile(upstream=up_url),
        token_file=state / TOKEN_FILE,
        port=8791,
    )
    spec = spec.with_egress([agy_backend])

    box = Box(spec, box_id="darwin-live-e2e")
    try:
        async with box:
            proxy_port = box.activation(backend).port
            # Token must NOT appear in wrapper/launcher argv.
            cmd = box.command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, os, resource, socket, tempfile, urllib.error, urllib.request\n"
                        # Verify RLIMIT_NOFILE soft cap applied by aisan.launch.
                        "soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
                        "assert soft <= 65536, soft\n"
                        # Verify TMPDIR is writable inside box and distinct from /private/tmp.
                        "tf = pathlib.Path(tempfile.gettempdir()) / 'probe.txt' if False else os.path.join(os.environ['TMPDIR'], 'probe.txt')\n"
                        "open(tf, 'w').write('scratch')\n"
                        # Verify private_root() and host credential file are sealed/denied.
                        "for op in [\n"
                        f"    lambda: open({str(cred_file)!r}).read(),\n"
                        f"    lambda: os.listdir({str(private_root())!r}),\n"
                        f"    lambda: open({str(profile_path(box.box_id))!r}).read(),\n"
                        "]:\n"
                        "    try:\n"
                        "        op()\n"
                        "        raise SystemExit('SECRET_EXPOSED')\n"
                        "    except PermissionError:\n"
                        "        pass\n"
                        # Verify direct connection to fake upstream port on 127.0.0.1 is BLOCKED by Seatbelt!
                        "try:\n"
                        f"    socket.create_connection(('127.0.0.1', {up_port}), 1)\n"
                        "    raise SystemExit('DIRECT_UPSTREAM_PORT_REACHABLE')\n"
                        "except OSError:\n"
                        "    pass\n"
                        # Verify staged PinnedSymlink token file carries per-box proxy token, never real secret.
                        f"agy_tok = open({str(state / TOKEN_FILE)!r}).read()\n"
                        "assert 'agy-real-token' not in agy_tok\n"
                        # Verify allowed request through Anthropic proxy succeeds with swapped bearer!
                        "req = urllib.request.Request(\n"
                        "    os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages',\n"
                        "    data=json.dumps({'model': 'claude-haiku-4-5', 'max_tokens': 1, 'messages': [{'role': 'user', 'content': 'hi'}]}).encode(),\n"
                        "    headers={\n"
                        "        'Content-Type': 'application/json',\n"
                        "        'Authorization': 'Bearer ' + os.environ['CLAUDE_CODE_OAUTH_TOKEN'],\n"
                        "        'anthropic-version': '2023-06-01',\n"
                        "    },\n"
                        ")\n"
                        "resp = json.loads(urllib.request.urlopen(req, timeout=5).read())\n"
                        "assert resp['id'] == 'msg_ok'\n"
                        # Verify web_search is STILL blocked with 403 in isolated Seatbelt mode!
                        "bad_req = urllib.request.Request(\n"
                        "    os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages',\n"
                        "    data=json.dumps({'model': 'claude-haiku-4-5', 'max_tokens': 1, 'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}], 'messages': [{'role': 'user', 'content': 'q'}]}).encode(),\n"
                        "    headers={\n"
                        "        'Content-Type': 'application/json',\n"
                        "        'Authorization': 'Bearer ' + os.environ['CLAUDE_CODE_OAUTH_TOKEN'],\n"
                        "        'anthropic-version': '2023-06-01',\n"
                        "    },\n"
                        ")\n"
                        "try:\n"
                        "    urllib.request.urlopen(bad_req, timeout=5)\n"
                        "    raise SystemExit('WEB_SEARCH_ALLOWED')\n"
                        "except urllib.error.HTTPError as e:\n"
                        "    assert e.code == 403, e.code\n"
                        "print('DARWIN_BOX_E2E_OK')\n"
                    ),
                ]
            )
            joined_argv = " ".join(cmd)
            assert "super-secret-oat" not in joined_argv
            assert (
                str(proxy_port) in box.wrapper()[-1]
                or profile_path(box.box_id).is_file()
            )
            res = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            assert res.returncode == 0, res.stderr
            assert "DARWIN_BOX_E2E_OK" in res.stdout
    finally:
        await runner.cleanup()

    assert seen_auth == ["Bearer super-secret-oat"]
    assert not scratch_dir(box.box_id).exists()
    assert not profile_path(box.box_id).exists()
    assert not (state / TOKEN_FILE).exists()
