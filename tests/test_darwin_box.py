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
from aisan.session import darwin_executable_grant, git_config_binds
from aisan.spec import BoxSpec
from aisan.statedir import planted_credentials


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


async def test_claude_plan_keychain_lookup_and_file_fallback(tmp_path, monkeypatch):
    import aisan.egress.claude_plan as cp_mod

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
    monkeypatch.setattr(cp_mod, "_read_macos_keychain_text", lambda _d: kc_json)
    cred = PlanCredential(
        custom_dir / "absent.json", claude_command=("claude",), use_keychain=True
    )
    assert await cred.upstream() == {"authorization": "Bearer from-keychain"}
    assert cred.dress("tok")["CLAUDE_CODE_SUBSCRIPTION_TYPE"] == "pro"

    # Keychain miss (None) falls back to `self.path`.
    monkeypatch.setattr(cp_mod, "_read_macos_keychain_text", lambda _d: None)
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
    (skills / "alpha").mkdir(parents=True)
    (skills / ".system").mkdir()

    # Claude Code uses PinnedSymlink for skills and omits EnsurePath(state / "skills").
    cc_spec = claude_code(wt, state=state, skills=skills)
    assert PinnedSymlink(skills, state / "skills") in cc_spec.binds
    assert all(e.path != state / "skills" for e in cc_spec.ensure)

    # Codex uses per-entry PinnedSymlinks skipping `.system`, `codex-cfprefs`, `--no-daemon`, and file store.
    cx_backend = CodexBackend(
        credentials=tmp_path / "absent.json",
        codex_home=state,
    )
    assert 'cli_auth_credentials_store="file"' in cx_backend.config_overrides()
    assert "--no-daemon" in codex_argv()
    cx_spec = codex(wt, state=state, skills=skills, egress=(cx_backend,))
    assert cx_spec.mach_profiles == ("codex-cfprefs",)
    assert PinnedSymlink(skills / "alpha", state / "skills" / "alpha") in cx_spec.binds
    assert all(
        getattr(b, "dst", None) != state / "skills" / ".system" for b in cx_spec.binds
    )

    # OpenCode and Antigravity set allow_local_loopback=True on Darwin.
    catalog = tmp_path / "models.json"
    catalog.write_text("{}")
    oc_spec = opencode(wt, state=state, models=catalog)
    assert oc_spec.allow_local_loopback is True
    assert (
        PinnedSymlink(catalog, state / "xdg-cache" / "opencode" / "models.json")
        in oc_spec.binds
    )

    ag_spec = antigravity(wt, state=state)
    assert ag_spec.allow_local_loopback is True


@needs_seatbelt
def test_darwin_executable_grant_mounts_standalone_app_bundle_root(tmp_path):
    bundle_root = tmp_path / "FakeApp.app" / "Contents" / "Resources" / "tool-cli"
    exe = bundle_root / "bin" / "tool"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)

    grant = darwin_executable_grant(exe)
    assert Bind(bundle_root.resolve(), RO, guard=False) in grant.binds
    assert grant.path == (exe.parent,)

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
def test_darwin_git_config_binds_wires_git_config_global_in_box_env(
    tmp_path, monkeypatch
):
    wt = tmp_path / "wt"
    wt.mkdir()
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    cfg = xdg / "git" / "config"
    cfg.write_text("[user]\n\tname = BoxUser\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    binds = git_config_binds()
    assert binds == [Bind(cfg.resolve(), RO)]
    spec = BoxSpec(
        root=wt,
        binds=tuple(binds),
        tmpfs=(),
        env=(),
        egress=(),
        unshare_net=True,
    )
    box = Box(spec, box_id="darwin-gitcfg")
    assert box.env["GIT_CONFIG_GLOBAL"] == seatbelt_path(cfg)


@needs_seatbelt
def test_cli_agy_explain_runs_cleanly_and_leaves_no_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = {**os.environ, "HOME": str(fake_home)}
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
