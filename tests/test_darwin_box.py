# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from aiohttp import web
from conftest import needs_seatbelt

from aisan import Box
from aisan.egress.anthropic import AnthropicBackend
from aisan.egress.antigravity import USER_INFO_URL_ENV, AntigravityBackend
from aisan.egress.antigravity_keyring import (
    KeyringError,
    read_macos_keychain,
)
from aisan.egress.base import PLACEHOLDER_KEY
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
from aisan.seatbelt import SeatbeltSandbox, seatbelt_path
from aisan.session import (
    darwin_executable_grant,
    git_config_binds,
    git_config_env,
    state_dir,
)
from aisan.spec import BoxSpec
from aisan.statedir import (
    StateLease,
    planted_credentials,
    prepare_state_dir,
    share_state_dir,
)


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


@pytest.fixture
def force_darwin(monkeypatch):
    monkeypatch.setattr("aisan.policy_tree.sys.platform", "darwin")
    monkeypatch.setattr("aisan.egress.sys.platform", "darwin")
    monkeypatch.setattr("aisan.userbinds.sys.platform", "darwin")


def test_darwin_presets_avoid_bind_over_and_overlay_and_configure_flags(
    tmp_path, force_darwin
):
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


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin Keychain layout")
def test_security_server_profile_refuses_a_system_keychain_mount(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()

    def spec(profiles):
        return BoxSpec(
            root=wt,
            binds=(Bind(Path("/Library/Keychains"), RO),),
            tmpfs=(),
            env=(),
            egress=(),
            unshare_net=False,
            mach_profiles=profiles,
        )

    Box(spec(()), box_id="syskc-plain", platform="darwin")._sandbox()
    with pytest.raises(ValueError, match="system Keychain"):
        Box(
            spec(("security-server",)), box_id="syskc-secsrv", platform="darwin"
        )._sandbox()


def test_darwin_executable_grant_mounts_standalone_app_bundle_root(
    tmp_path, monkeypatch, force_darwin
):
    # 1) Standalone `~/.cargo/bin/tool` install: must NOT mount `~/.cargo` whole!
    fake_home = tmp_path / "home"
    cargo_exe = fake_home / ".cargo" / "bin" / "rg-custom"
    cargo_exe.parent.mkdir(parents=True)
    cargo_exe.write_text("#!/bin/sh\nexit 0\n")
    cargo_exe.chmod(0o755)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

    cargo_grant = darwin_executable_grant(cargo_exe)
    bound_paths = {b.path for b in cargo_grant.binds if isinstance(b, Bind)}
    assert (fake_home / ".cargo").resolve() not in bound_paths
    assert cargo_exe.parent.resolve() not in bound_paths
    assert cargo_exe.resolve() in bound_paths
    assert cargo_grant.path == (cargo_exe.parent,)

    # 2) Self-contained venv (`pyvenv.cfg`) and `uv tool` symlinked-Python venv
    #    mount the venv + interpreter root without mounting `~/.local/bin` whole.
    venv_root = fake_home / "tools" / "my-venv"
    venv_exe = venv_root / "bin" / "tool"
    venv_exe.parent.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text("home = /usr\n")
    venv_exe.write_text("#!/bin/sh\nexit 0\n")
    venv_exe.chmod(0o755)
    venv_grant = darwin_executable_grant(venv_exe)
    assert Bind(venv_root.resolve(), RO, guard=False) in venv_grant.binds

    uv_py_root = fake_home / ".local" / "share" / "uv" / "python" / "cpython-3.14"
    (uv_py_root / "bin").mkdir(parents=True)
    (uv_py_root / "lib").mkdir(parents=True)
    uv_py_bin = uv_py_root / "bin" / "python3"
    uv_py_bin.write_text("#!/bin/sh\nexit 0\n")
    uv_py_bin.chmod(0o755)
    uv_venv = fake_home / ".local" / "share" / "uv" / "tools" / "mytool"
    (uv_venv / "bin").mkdir(parents=True)
    (uv_venv / "pyvenv.cfg").write_text(f"home = {uv_py_root / 'bin'}\n")
    (uv_venv / "bin" / "python").symlink_to(uv_py_bin)
    uv_entry = uv_venv / "bin" / "mytool"
    uv_entry.write_text("#!/bin/sh\nexit 0\n")
    uv_entry.chmod(0o755)
    local_bin = fake_home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    uv_link = local_bin / "mytool"
    uv_link.symlink_to(uv_entry)

    uv_grant = darwin_executable_grant(uv_link)
    uv_bound = {b.path for b in uv_grant.binds if isinstance(b, Bind)}
    assert local_bin.resolve() not in uv_bound
    assert uv_link in uv_bound
    assert uv_venv.resolve() in uv_bound
    assert uv_py_root.resolve() in uv_bound
    assert (uv_venv / "bin" / "python") not in uv_bound
    wt_check = tmp_path / "wt-check"
    wt_check.mkdir()
    assert SeatbeltSandbox(root=wt_check, binds=uv_grant.binds).resolve()

    # 3) Version-managed `nvm` / `npm -g` scoped `@openai/codex` symlink mounts
    #    both the `@openai/codex` package directory AND `<nvm_ver>/bin` (for `node`),
    #    while NOT mounting sibling `node_modules` packages!
    nvm_ver = fake_home / ".nvm" / "versions" / "node" / "v22.14.0"
    node_bin = nvm_ver / "bin"
    node_bin.mkdir(parents=True)
    (node_bin / "node").write_text("#!/bin/sh\nexit 0\n")
    (node_bin / "node").chmod(0o755)
    pkg_dir = nvm_ver / "lib" / "node_modules" / "@openai" / "codex"
    (pkg_dir / "bin").mkdir(parents=True)
    (pkg_dir / "package.json").write_text('{"name":"@openai/codex"}')
    entry_js = pkg_dir / "bin" / "codex.js"
    entry_js.write_text("#!/usr/bin/env node\n")
    entry_js.chmod(0o755)
    other_pkg = nvm_ver / "lib" / "node_modules" / "unrelated-pkg"
    other_pkg.mkdir(parents=True)
    codex_link = node_bin / "codex"
    codex_link.symlink_to(entry_js)

    nvm_grant = darwin_executable_grant(codex_link)
    nvm_bound = {b.path for b in nvm_grant.binds if isinstance(b, Bind)}
    assert pkg_dir.resolve() in nvm_bound
    assert node_bin.resolve() in nvm_bound
    assert other_pkg.resolve() not in nvm_bound
    assert (nvm_ver / "lib" / "node_modules").resolve() not in nvm_bound
    assert node_bin in nvm_grant.path
    assert SeatbeltSandbox(root=wt_check, binds=nvm_grant.binds).resolve()

    # 4) Custom `.app` bundle mounts the `.app` root.
    app_bundle = fake_home / "Applications" / "ToolApp.app"
    app_exe = app_bundle / "Contents" / "MacOS" / "tool"
    app_exe.parent.mkdir(parents=True)
    app_exe.write_text("#!/bin/sh\nexit 0\n")
    app_exe.chmod(0o755)
    app_grant = darwin_executable_grant(app_exe)
    assert Bind(app_bundle.resolve(), RO, guard=False) in app_grant.binds

    # 5) System binary needs no extra grant.
    assert not darwin_executable_grant(Path("/usr/bin/true"))


def test_darwin_box_staging_cleans_up_symlinks_dirs_scratch_and_profile(
    tmp_path, monkeypatch, force_darwin
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
        sb_scratch = seatbelt_path(scratch_dir(box.box_id))
        assert box.env["TMPDIR"] == sb_scratch
        assert box.env["XDG_CACHE_HOME"] == f"{sb_scratch}/xdg-cache"
        assert box.env["NPM_CONFIG_CACHE"] == f"{sb_scratch}/npm-cache"
        assert (scratch_dir(box.box_id) / "xdg-cache").is_dir()
        assert (scratch_dir(box.box_id) / "npm-cache").is_dir()
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


def test_darwin_git_config_binds_wires_git_config_global_and_ignores_submodules(
    tmp_path, monkeypatch, force_darwin
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
def test_an_uncontested_explain_leaves_no_state_behind(tmp_path):
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


@needs_seatbelt
async def test_concurrent_sessions_share_a_state_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    state = prepare_state_dir(state_dir("antigravity", repo, home=fake_home))
    token_link = state / TOKEN_FILE

    async def upstream(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    up_url, _, runner = await _fake_upstream(upstream)

    class _FakeAgyCred:
        files = (tmp_path / "agy-secret.json",)

        async def token(self) -> str:
            return "agy-real-token"

        async def check(self, backend: str) -> None:
            return None

    def session(n: int, lease: StateLease) -> Box:
        backend = AntigravityBackend(
            credential=_FakeAgyCred(),
            profile=AgyProfile(upstream=up_url),
            token_file=token_link,
        )
        spec = antigravity(repo, state=state, egress=(backend,), unshare_net=True)
        return Box(spec, box_id=f"darwin-shared-{n}", lease=lease)

    # Each box presents the token at the fixed client path to its own proxy.
    probe = (
        "import json, os, urllib.request\n"
        f"tok = json.load(open({str(token_link)!r}))['token']['access_token']\n"
        f"req = urllib.request.Request(os.environ[{USER_INFO_URL_ENV!r}],"
        " headers={'Authorization': 'Bearer ' + tok})\n"
        "urllib.request.urlopen(req, timeout=5).read()\n"
        "print('TOKEN=' + tok)\n"
    )

    async def token_seen_in(box: Box) -> str:
        res = await asyncio.to_thread(
            subprocess.run,
            box.command([sys.executable, "-c", probe]),
            env={**os.environ, **box.env},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        return res.stdout.strip().removeprefix("TOKEN=")

    try:
        with share_state_dir(state) as first:
            links = first.links
            async with session(1, first) as box1:
                token = await token_seen_in(box1)
                assert token != PLACEHOLDER_KEY
                with share_state_dir(state) as second:
                    async with session(2, second) as box2:
                        assert await token_seen_in(box2) == token
                        assert await token_seen_in(box1) == token
                # The first session still reads through the shared link.
                assert Path(os.readlink(token_link)).parent == links
                assert await token_seen_in(box1) == token

                # `--explain` joins and leaves without disturbing it.
                res = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "aisan.cli.main",
                        "agy",
                        "--explain",
                        str(repo),
                    ],
                    env={**os.environ, "HOME": str(fake_home)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert res.returncode == 0, res.stderr
                assert str(state) in res.stdout
                assert await token_seen_in(box1) == token
            assert not os.path.lexists(token_link)
        assert not links.exists()
    finally:
        await runner.cleanup()


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


@needs_seatbelt
async def test_darwin_depot_tools_job_and_reapi_unix_socket_wrapper(
    tmp_path, monkeypatch
):
    import importlib

    from aisan.egress.reapi import ReapiBackend
    from aisan.sandbox import WriteRegex

    dt_mod = importlib.import_module("aisan.presets.depot_tools_job")

    wt = tmp_path / "wt"
    (wt / "build" / "config" / "siso").mkdir(parents=True)
    sisoenv = wt / "build" / "config" / "siso" / ".sisoenv"
    sisoenv.write_text(
        "SISO_PROJECT=rbe-chromium-untrusted\nSISO_REAPI_INSTANCE=default_instance\n"
    )
    fake_siso = wt / "third_party" / "siso" / "cipd" / "siso"
    fake_siso.parent.mkdir(parents=True)
    fake_siso.write_text(
        "#!/bin/sh\n"
        'printf "PROJ=%s INST=%s ADDR=%s ARGS=%s\\n"'
        ' "$SISO_PROJECT" "$SISO_REAPI_INSTANCE" "$SISO_REAPI_ADDRESS" "$*"\n'
    )
    fake_siso.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=wt, check=True)

    cache = tmp_path / "vpython-root.501"
    (cache / "store").mkdir(parents=True)
    monkeypatch.setattr(dt_mod, "_vpython_cache", lambda **_: cache)
    fake_dt = tmp_path / "depot_tools"
    fake_dt.mkdir()
    (fake_dt / "autoninja").write_text("#!/bin/sh\nexit 0\n")
    (fake_dt / "autoninja").chmod(0o755)

    backend = ReapiBackend(
        project="rbe-chromium-untrusted",
        sisoenv=sisoenv,
        mint=lambda: asyncio.sleep(0, result="fake-luci-token"),
    )
    spec = dt_mod.depot_tools_job(wt, egress=(backend,), depot_tools=fake_dt)
    assert any(isinstance(b, WriteRegex) and b.path == cache for b in spec.binds)
    assert dict(spec.env)["DEPOT_TOOLS_BUILD_TELEMETRY"] == "0"

    other_sock = Path(f"/private/tmp/aisan-unauth-{os.getpid()}.sock")
    other_sock.unlink(missing_ok=True)
    unauth_srv = await asyncio.start_unix_server(
        lambda r, w: w.close(), str(other_sock)
    )
    box = Box(spec, box_id="darwin-rbe-test")
    try:
        async with box:
            # Direct Unix-socket-only box needs no `aisan.launch` relay wrapper!
            assert box.launch_prefix() == []
            act = box.activation(backend)
            assert act.client_env["SISO_PROJECT"] == "rbe-chromium-untrusted"
            assert act.client_env["SISO_REAPI_ADDRESS"].startswith("unix://")
            cmd = box.command(
                [
                    "/bin/sh",
                    "-c",
                    (
                        'cd / && PATH="" SISO_REAPI_ADDRESS=override:1234 SISO_REAPI_INSTANCE=default_instance '
                        f'"$SISO_PATH" ninja -C {wt / "out" / "x"} d8 && '
                        f'"{sys.executable}" -c "'
                        "import os, socket; "
                        "sock_path = os.environ['SISO_REAPI_ADDRESS'].removeprefix('unix://'); "
                        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
                        "s.connect(sock_path); s.close(); "
                        "u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
                        "ok = False\n"
                        "try:\n"
                        f"    u.connect({str(other_sock)!r})\n"
                        "except OSError:\n"
                        "    ok = True\n"
                        "assert ok, 'Seatbelt allowed unauthorized unix socket'"
                        '"'
                    ),
                ]
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
            assert (
                "INST=projects/rbe-chromium-untrusted/instances/default_instance"
                in res.stdout
            )
            assert "ADDR=unix://" in res.stdout
            assert "override:1234" not in res.stdout
            assert "rbe.sock" in res.stdout
            assert f"ARGS=ninja -C {wt / 'out' / 'x'} d8" in res.stdout
    finally:
        unauth_srv.close()
        await unauth_srv.wait_closed()
        other_sock.unlink(missing_ok=True)
    assert not box.runtime_dir.exists()


def test_darwin_casefold_and_firmlink_credential_guards(
    tmp_path, monkeypatch, force_darwin
):
    from aisan.userbinds import load

    firmlink_home = Path("/Users/aisan-test-user")
    monkeypatch.setenv("HOME", str(firmlink_home))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: firmlink_home))

    for bad_toml in (
        'ro = ["~/.SSH"]\n',
        'ro = ["~/library/keychains"]\n',
        'ro = ["~/.CODEX"]\n',
        f'ro = ["/System/Volumes/Data{firmlink_home}/.ssh"]\n',
        f'ro = ["/System/Volumes/Data{firmlink_home}/Library/Keychains"]\n',
    ):
        spec_file = tmp_path / "bad.toml"
        spec_file.write_text(bad_toml)
        with pytest.raises(ValueError):
            load(spec_file, egress=())

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

    wt = tmp_path / "wt"
    wt.mkdir()
    upper_ssh = fake_home / ".SSH"
    upper_ssh.mkdir()
    with pytest.raises(ValueError, match="host key store"):
        Box(
            BoxSpec(
                root=wt,
                binds=(Bind(upper_ssh, RO, guard=False),),
                tmpfs=(),
                env=(),
                egress=(),
                unshare_net=True,
            ),
            box_id="casefold-ssh",
        )._sandbox()


@needs_seatbelt
def test_darwin_socket_path_checks_canonical_private_tmp_length(monkeypatch):
    from aisan.egress.reapi import ReapiBackend

    backend = ReapiBackend(project="p", sisoenv=Path("/tmp/x"))
    # `/tmp/<88 chars>/rbe.sock` is 102 bytes raw (<= 103) but `/private/tmp/...`
    # is 110 bytes (> 103), so `socket_path()` must reject it upfront.
    borderline = Path("/tmp") / ("a" * 88)
    assert len(str(borderline / "rbe.sock").encode()) == 102
    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        backend.socket_path(borderline)


def test_darwin_codex_symlinked_agents_skills_and_cross_platform_binds(
    tmp_path, monkeypatch, force_darwin
):
    fake_home = tmp_path / "home"
    dotfiles_skills = tmp_path / "dotfiles" / "skills"
    (dotfiles_skills / "my-skill").mkdir(parents=True)
    (fake_home / ".agents").mkdir(parents=True)
    (fake_home / ".agents" / "skills").symlink_to(dotfiles_skills)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

    wt = tmp_path / "wt"
    wt.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    spec = codex(wt, state=state, skills=fake_home / ".agents" / "skills")
    assert (
        PinnedSymlink(
            (dotfiles_skills / "my-skill").resolve(), state / "skills" / "my-skill"
        )
        in spec.binds
    )
    with Box(spec, box_id="sym-agents-skills").staged() as box:
        assert (state / "skills" / "my-skill").is_symlink()
        assert box.wrapper()

    # Simulated Linux host constructing `Box(..., platform="darwin")` gets PinnedSymlink.
    from aisan.egress.antigravity_keyring import KeyringCredential

    monkeypatch.setattr("aisan.policy_tree.sys.platform", "linux")
    cx_backend = CodexBackend(credentials=tmp_path / "c.json", codex_home=state)
    ag_backend = AntigravityBackend(
        credential=KeyringCredential(), token_file=state / TOKEN_FILE
    )
    rt = tmp_path / "rt"
    assert isinstance(cx_backend.box_binds(rt, platform="darwin")[0], PinnedSymlink)
    assert isinstance(ag_backend.box_binds(rt, platform="darwin")[0], PinnedSymlink)


@needs_seatbelt
def test_darwin_explain_includes_executable_grant(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    venv = fake_home / "tools" / "agy-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n")
    exe = venv / "bin" / "agy"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
    }
    res = subprocess.run(
        [sys.executable, "-m", "aisan.cli.main", "agy", "--explain", str(repo)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert str(venv.resolve()) in res.stdout


@needs_seatbelt
async def test_darwin_run_interactive_warns_on_allow_local_loopback(tmp_path, capsys):
    from aisan.session import LauncherFlags, run_interactive

    repo = tmp_path / "repo"
    repo.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    spec = antigravity(repo, state=state)
    rc = await run_interactive(
        client="antigravity",
        harness="antigravity",
        executable="true",
        repo=repo,
        state=state,
        spec=spec,
        command=lambda _box: ["/usr/bin/true"],
        binary=lambda: Path("/usr/bin/true"),
        flags=LauncherFlags(),
        explain_only=False,
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING: box allows localhost TCP on macOS" in err


@pytest.mark.live
@needs_seatbelt
def test_real_claude_cli_matches_keychain_service_derivation(tmp_path, monkeypatch):
    from aisan.egress.claude_plan import _keychain_service
    from aisan.presets.claude_code import claude_code_binary

    claude = claude_code_binary()
    if claude is None:
        pytest.skip("claude is not installed")
    sec_log = tmp_path / "security.log"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "security"
    shim.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {sec_log}\nexit 44\n')
    shim.chmod(0o755)

    # 1) Custom CLAUDE_CONFIG_DIR: must match `Claude Code-credentials-<sha8>`.
    custom_cfg = tmp_path / "custom-claude-cfg"
    custom_cfg.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_cfg))
    monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
    expected_custom = _keychain_service(custom_cfg)

    env = {
        **os.environ,
        "PATH": f"{shim_dir}:{os.environ['PATH']}",
        "CLAUDE_CONFIG_DIR": str(custom_cfg),
    }
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    subprocess.run(
        [str(claude), "auth", "status"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    logged = sec_log.read_text() if sec_log.exists() else ""
    assert f"-s {expected_custom}" in logged, logged

    # 2) Default `~/.claude` with CLAUDE_CONFIG_DIR unset: must match `Claude Code-credentials`.
    sec_log.unlink(missing_ok=True)
    fake_home = tmp_path / "home"
    (fake_home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    expected_default = _keychain_service(fake_home / ".claude")
    assert expected_default == "Claude Code-credentials"

    env_def = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{shim_dir}:{os.environ['PATH']}",
    }
    env_def.pop("CLAUDE_CONFIG_DIR", None)
    env_def.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    subprocess.run(
        [str(claude), "auth", "status"],
        env=env_def,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    logged_def = sec_log.read_text() if sec_log.exists() else ""
    assert re.search(rf"-s {re.escape(expected_default)}(?:\s|$)", logged_def), (
        logged_def
    )
