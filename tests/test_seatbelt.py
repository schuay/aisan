# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pty
import shutil
import socket
import subprocess
import sys
import termios
from pathlib import Path

import pytest
from conftest import needs_seatbelt

from aisan.gitbinds import git_binds, git_host_files
from aisan.sandbox import (
    RO,
    RW,
    Bind,
    BindOver,
    GuardViolation,
    MountConflict,
    Overlay,
    PinnedSymlink,
    Sandbox,
    Seal,
    SymlinkDestination,
    WriteRegex,
)
from aisan.seatbelt import (
    SeatbeltSandbox,
    seatbelt_lexpath,
    seatbelt_path,
    through_darwin_system_symlink,
)
from aisan.spec import BoxSpec, Grant


def _run_sb(
    sb: SeatbeltSandbox,
    cmd: list[str],
    *,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*sb.wrapper(), *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _python_binds() -> tuple[Bind, ...]:
    """Read-only binds needed for `sys.executable` inside a minimal Seatbelt box."""
    from aisan.launch import launcher_binds

    return tuple(b for b in launcher_binds() if isinstance(b, Bind))


def _box_env(tmp_dir: Path) -> tuple[tuple[str, str], ...]:
    return (
        ("HOME", str(Path.home())),
        ("TMPDIR", str(tmp_dir)),
        ("XDG_CONFIG_HOME", str(tmp_dir)),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("PATH", "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"),
        ("GIT_AUTHOR_NAME", "Test"),
        ("GIT_AUTHOR_EMAIL", "test@example.invalid"),
        ("GIT_COMMITTER_NAME", "Test"),
        ("GIT_COMMITTER_EMAIL", "test@example.invalid"),
    )


# --- Pure policy, validation, and path-normalization tests ---


def test_firmlink_and_root_symlink_normalization():
    assert through_darwin_system_symlink(Path("/tmp/aisan-501")) == Path(
        "/private/tmp/aisan-501"
    )
    assert through_darwin_system_symlink(Path("/var/folders/x")) == Path(
        "/private/var/folders/x"
    )
    assert through_darwin_system_symlink(Path("/etc/hosts")) == Path(
        "/private/etc/hosts"
    )
    assert through_darwin_system_symlink(
        Path("/System/Volumes/Data/Users/alice/repo")
    ) == Path("/Users/alice/repo")
    assert through_darwin_system_symlink(
        Path("/System/Volumes/Data/private/tmp/x")
    ) == Path("/private/tmp/x")
    assert (
        seatbelt_path(Path("/System/Volumes/Data/Users/nonexistent-aisan-u/work"))
        == "/Users/nonexistent-aisan-u/work"
    )


def test_control_characters_and_invalid_endpoints_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="control character"):
        seatbelt_path(Path("/tmp/bad\nline"))
    with pytest.raises(ValueError, match="absolute"):
        seatbelt_lexpath(Path("."))
    with pytest.raises(ValueError, match="absolute"):
        seatbelt_lexpath(Path("/tmp/a/.."))
    with pytest.raises(ValueError, match="absolute"):
        SeatbeltSandbox(root=tmp_path, profile_path=Path("relative.sb"))
    with pytest.raises(ValueError, match="absolute"):
        SeatbeltSandbox(root=tmp_path, unix_sockets=(Path("rel.sock"),))
    with pytest.raises(ValueError, match="invalid Seatbelt tcp_port"):
        SeatbeltSandbox(root=tmp_path, tcp_ports=("*",))
    with pytest.raises(ValueError, match="invalid Seatbelt tcp_port"):
        SeatbeltSandbox(root=tmp_path, tcp_ports=(0,))
    with pytest.raises(ValueError, match="invalid Seatbelt tcp_port"):
        SeatbeltSandbox(root=tmp_path, tcp_ports=(True,))
    with pytest.raises(ValueError, match="unknown mach_profiles"):
        SeatbeltSandbox(root=tmp_path, mach_profiles=("securityd",))


@needs_seatbelt
def test_quote_escaping_in_rendered_sbpl(tmp_path):
    sb = SeatbeltSandbox(
        root=tmp_path,
        binds=(Bind(Path('/tmp/quote"inject'), RO, optional=True),),
    )
    quoted_dir = tmp_path / 'dir"with\\quotes'
    quoted_dir.mkdir()
    rendered = SeatbeltSandbox(
        root=tmp_path, binds=(Bind(quoted_dir, RO),)
    ).render_sbpl()
    assert r"dir\"with\\quotes" in rendered
    assert sb.render_sbpl().startswith("(version 1)\n")


def test_unsupported_linux_constructs_are_refused_by_name(tmp_path):
    src = tmp_path / "src"
    src.write_text("x")
    with pytest.raises(ValueError, match=r"BindOver .* is not supported"):
        SeatbeltSandbox(root=tmp_path, binds=(BindOver(src, tmp_path / "dst"),)).tree()

    ov = tmp_path / "ov"
    ov.mkdir()
    with pytest.raises(ValueError, match=r"Overlay .* is not supported"):
        SeatbeltSandbox(root=tmp_path, binds=(Overlay(ov),)).tree()

    with pytest.raises(ValueError, match=r"root_tmpfs .* is not supported"):
        SeatbeltSandbox(root=tmp_path, root_tmpfs=1024)


def test_bwrap_refuses_macos_only_constructs(tmp_path):
    src = tmp_path / "src"
    src.write_text("x")
    with pytest.raises(ValueError, match="PinnedSymlink"):
        Sandbox(
            root=tmp_path,
            binds=(PinnedSymlink(src, tmp_path / "dst"),),
            use_cgroup=False,
        ).entries()
    with pytest.raises(ValueError, match="WriteRegex"):
        Sandbox(
            root=tmp_path,
            binds=(WriteRegex(tmp_path, (r"/\.lock",)),),
            use_cgroup=False,
        ).entries()


def test_write_regex_validates_anchored_suffixes(tmp_path):
    with pytest.raises(ValueError, match="at least one suffix"):
        WriteRegex(tmp_path, ())
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("/",))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("relative.lock",))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("/bad\nregex",))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ('/bad"quote',))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("/(?:noncapturing)",))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("/store/../evil",))
    with pytest.raises(ValueError, match="POSIX ERE"):
        WriteRegex(tmp_path, ("/*.lock",))
    with pytest.raises(ValueError, match="cannot contain '\"'"):
        WriteRegex(tmp_path / 'dir"quote', ("/stamp",))


def test_mach_profiles_and_grants_validate_known_names(tmp_path):
    with pytest.raises(ValueError, match="unknown mach_profiles"):
        Grant(mach_profiles=("securityd",))
    with pytest.raises(ValueError, match="unknown mach_profiles"):
        BoxSpec(
            root=tmp_path,
            binds=(),
            tmpfs=(),
            env=(),
            egress=(),
            unshare_net=True,
            mach_profiles=("unknown",),
        )
    assert bool(Grant(allow_local_loopback=True))
    assert bool(Grant(mach_profiles=("codex-cfprefs",)))


@needs_seatbelt
def test_tree_invariants_apfs_case_folding_and_write_regex_merges(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()

    # Stricter mode wins: RO beats WriteRegex, and WriteRegex beats RW.
    sb = SeatbeltSandbox(
        root=wt,
        binds=(
            Bind(shared, RW, guard=False),
            WriteRegex(shared, (r"/\.lock",), guard=False),
            Bind(shared, RO, guard=False),
        ),
    )
    assert [m.op for m in sb.resolve() if m.dst == Path(seatbelt_path(shared))] == [
        "ro"
    ]

    sb_wr = SeatbeltSandbox(
        root=wt,
        binds=(
            Bind(shared, RW, guard=False),
            WriteRegex(shared, (r"/\.lock",), guard=False),
        ),
    )
    assert [m.op for m in sb_wr.resolve() if m.dst == Path(seatbelt_path(shared))] == [
        "write-regex"
    ]

    # Distinct WriteRegex suffix sets at the same path conflict rather than widening.
    with pytest.raises(MountConflict):
        SeatbeltSandbox(
            root=wt,
            binds=(
                WriteRegex(shared, (r"/\.lock",)),
                WriteRegex(shared, (r"/stamp",), guard=False),
            ),
        ).tree()

    # WriteRegex on system surface or regular file fails cleanly.
    with pytest.raises(MountConflict):
        SeatbeltSandbox(
            root=wt, binds=(WriteRegex(Path("/usr"), (r"/\.lock",)),)
        ).tree()
    reg_file = tmp_path / "regular.txt"
    reg_file.write_text("x")
    with pytest.raises(NotADirectoryError):
        SeatbeltSandbox(root=wt, binds=(WriteRegex(reg_file, (r"/\.lock",)),)).resolve()

    # Plain bind below guard is refused, including via APFS case variation.
    guarded = tmp_path / "guarded"
    (guarded / "sub").mkdir(parents=True)
    with pytest.raises(GuardViolation):
        SeatbeltSandbox(
            root=wt,
            binds=(Bind(guarded, RO), Bind(guarded / "sub", RW, guard=False)),
        ).tree()
    upper_guarded = tmp_path / "GUARDED"
    if upper_guarded.exists():  # case-insensitive APFS
        with pytest.raises(GuardViolation):
            SeatbeltSandbox(
                root=wt,
                binds=(
                    Bind(guarded, RO),
                    Bind(upper_guarded / "sub", RW, guard=False),
                ),
            ).tree()

    # Mount conflict (Seal vs RW).
    with pytest.raises(MountConflict):
        SeatbeltSandbox(
            root=wt,
            binds=(Seal(shared), Bind(shared, RW)),
        ).tree()

    # Symlinks refused both inside RW root and for unmounted RW/PinnedSymlink ancestors.
    outside = tmp_path / "outside"
    outside.mkdir()
    (wt / "link").symlink_to(outside)
    with pytest.raises(SymlinkDestination):
        SeatbeltSandbox(root=wt, binds=(Bind(wt / "link" / "cfg", RO),)).resolve()
    unmounted_link = tmp_path / "unmounted_link"
    unmounted_link.symlink_to(guarded)
    with pytest.raises(SymlinkDestination):
        SeatbeltSandbox(
            root=wt, binds=(Bind(unmounted_link, RW, guard=False),)
        ).resolve()
    with pytest.raises(GuardViolation):
        SeatbeltSandbox(
            root=wt,
            binds=(Bind(guarded, RO), Bind(unmounted_link / "sub", RO, guard=False)),
        ).tree()
    with pytest.raises(SymlinkDestination):
        SeatbeltSandbox(
            root=wt,
            binds=(PinnedSymlink(reg_file, wt / "link" / "auth.json"),),
        ).resolve()

    # Colliding, nested, sealed, or unstaged PinnedSymlinks are refused.
    other_file = tmp_path / "other.txt"
    other_file.write_text("y")
    with pytest.raises(MountConflict):
        SeatbeltSandbox(
            root=wt,
            binds=(
                PinnedSymlink(reg_file, wt / "auth.json"),
                PinnedSymlink(other_file, wt / "auth.json"),
            ),
        ).tree()
    with pytest.raises(MountConflict):
        SeatbeltSandbox(
            root=wt,
            binds=(
                PinnedSymlink(shared, wt / "skills"),
                PinnedSymlink(reg_file, wt / "skills" / "a"),
            ),
        ).tree()
    sealed_dir = tmp_path / "sealed_sym"
    sealed_dir.mkdir()
    with pytest.raises(GuardViolation):
        SeatbeltSandbox(
            root=wt,
            binds=(Seal(sealed_dir), PinnedSymlink(reg_file, sealed_dir / "a")),
        ).tree()
    (wt / "not_a_symlink.json").write_text("plain")
    with pytest.raises(ValueError, match="staged as a symlink"):
        SeatbeltSandbox(
            root=wt,
            binds=(PinnedSymlink(reg_file, wt / "not_a_symlink.json"),),
        ).render_sbpl()


@needs_seatbelt
def test_exposure_and_seal_checks_catch_leaks_including_case_and_system_literals(
    tmp_path,
):
    wt = tmp_path / "wt"
    wt.mkdir()
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    cred = secret_dir / "token.json"
    cred.write_text("{}")

    # Direct or ancestor exposure via WriteRegex, PinnedSymlink, and system literals.
    sb_regex = SeatbeltSandbox(root=wt, binds=(WriteRegex(secret_dir, (r"/\.lock",)),))
    assert sb_regex.exposed_path((cred,)) is not None

    sb_sym = SeatbeltSandbox(root=wt, binds=(PinnedSymlink(cred, wt / "auth.json"),))
    assert sb_sym.exposed_path((cred,)) is not None

    sb_base = SeatbeltSandbox(root=wt)
    assert sb_base.exposed_path((Path("/private/etc/hosts"),)) is not None
    assert sb_base.exposed_path((cred,)) is None

    upper_secret = tmp_path / "SECRETS"
    if upper_secret.exists():  # case-insensitive APFS
        sb_case = SeatbeltSandbox(root=wt, binds=(Bind(upper_secret, RO),))
        assert sb_case.exposed_path((cred,)) is not None

    # Alias republishing a sealed directory is refused.
    sealed = tmp_path / "sealed"
    (sealed / "sib").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(sealed / "sib")
    sb_seal_leak = SeatbeltSandbox(
        root=wt,
        binds=(Seal(sealed), Bind(alias, RO)),
    )
    with pytest.raises(ValueError, match="republish the sealed directory"):
        sb_seal_leak.render_sbpl()


@needs_seatbelt
def test_security_server_and_securityd_are_never_in_rendered_sbpl(tmp_path):
    for unshare in (True, False):
        sbpl = SeatbeltSandbox(
            root=tmp_path,
            unshare_net=unshare,
            allow_local_loopback=True,
            mach_profiles=("codex-cfprefs",),
        ).render_sbpl()
        assert "SecurityServer" not in sbpl
        assert "securityd" not in sbpl
        assert "(deny process-info*)" in sbpl
        assert "com.apple.cfprefsd.agent" in sbpl
        if unshare:
            assert "trustd" not in sbpl
        else:
            assert "com.apple.trustd.agent" in sbpl


@needs_seatbelt
def test_wrapper_deduplicates_env_guards_env_flags_and_supports_profile_file(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    prof_file = tmp_path / "box.sb"
    sb = SeatbeltSandbox(
        root=wt,
        env=(("A", "1"), ("B", "2"), ("A", "override")),
        profile_path=prof_file,
    )
    argv = sb.wrapper()
    assert argv[:5] == ["/usr/bin/env", "-i", "-C", seatbelt_path(wt), "--"]
    assert "A=override" in argv
    assert "A=1" not in argv
    assert argv[-3:] == ["/usr/bin/sandbox-exec", "-f", str(prof_file)]
    assert prof_file.stat().st_mode & 0o777 == 0o600

    for bad_key in ("", "-S/bin/sh", "-C/tmp", "BAD=KEY", "BAD\x00KEY"):
        with pytest.raises(ValueError, match="invalid environment"):
            SeatbeltSandbox(root=wt, env=((bad_key, "1"),)).wrapper()


# --- Live macOS `sandbox-exec` enforcement tests ---


@needs_seatbelt
def test_live_confinement_clearenv_cwd_home_and_write_denial(tmp_path, monkeypatch):
    wt = tmp_path / "wt"
    outside = tmp_path / "outside"
    scratch = tmp_path / "scratch"
    for d in (wt, outside, scratch):
        d.mkdir()
    (outside / "secret.txt").write_text("top-secret")
    monkeypatch.setenv("HOST_LEAK_VAR", "leaked")

    sb = SeatbeltSandbox(
        root=wt,
        binds=(*_python_binds(), Bind(scratch, RW)),
        env=_box_env(scratch),
        unshare_net=True,
    )
    code = (
        "import os, pwd\n"
        "assert 'HOST_LEAK_VAR' not in os.environ\n"
        f"assert os.getcwd() == {seatbelt_path(wt)!r}\n"
        "assert pwd.getpwuid(os.getuid()).pw_name\n"
        "(open('inside.txt', 'w')).write('ok')\n"
        "for bad in [\n"
        f"    lambda: open({str(outside / 'secret.txt')!r}).read(),\n"
        f"    lambda: open({str(outside / 'plant.txt')!r}, 'w'),\n"
        "    lambda: os.listdir(os.environ['HOME']),\n"
        "]:\n"
        "    try:\n"
        "        bad(); raise SystemExit('UNEXPECTED_ALLOW')\n"
        "    except PermissionError:\n"
        "        pass\n"
        "print('CONFINED_OK')\n"
    )
    res = _run_sb(sb, [sys.executable, "-c", code])
    assert res.returncode == 0, res.stderr
    assert "CONFINED_OK" in res.stdout
    assert (wt / "inside.txt").read_text() == "ok"


@needs_seatbelt
def test_live_ro_pin_under_rw_and_deeper_rw_hole(tmp_path):
    wt = tmp_path / "wt"
    scratch = tmp_path / "scratch"
    ro_dir = wt / "pinned"
    rw_hole = ro_dir / "hole"
    rw_hole.mkdir(parents=True)
    scratch.mkdir()
    ro_file = wt / "locked.cfg"
    ro_file.write_text("immutable")
    (ro_dir / "keep.txt").write_text("keep")

    sb = SeatbeltSandbox(
        root=wt,
        binds=(
            *_python_binds(),
            Bind(scratch, RW),
            Bind(ro_file, RO),
            Bind(ro_dir, RO),
            Bind(rw_hole, RW),
        ),
        env=_box_env(scratch),
        unshare_net=True,
    )
    code = (
        "import os\n"
        f"assert open({str(ro_file)!r}).read() == 'immutable'\n"
        f"open({str(rw_hole / 'writable.txt')!r}, 'w').write('hole-ok')\n"
        "for bad in [\n"
        f"    lambda: open({str(ro_file)!r}, 'w'),\n"
        f"    lambda: os.unlink({str(ro_file)!r}),\n"
        f"    lambda: os.link({str(ro_file)!r}, 'hardlink.cfg'),\n"
        f"    lambda: open({str(ro_dir / 'new.txt')!r}, 'w'),\n"
        f"    lambda: os.rename({str(ro_dir)!r}, 'pinned.bak'),\n"
        f"    lambda: os.rename({str(rw_hole)!r}, {str(ro_dir / 'hole.bak')!r}),\n"
        "]:\n"
        "    try:\n"
        "        bad(); raise SystemExit('PIN_BYPASSED')\n"
        "    except PermissionError:\n"
        "        pass\n"
        "print('RO_PIN_OK')\n"
    )
    res = _run_sb(sb, [sys.executable, "-c", code])
    assert res.returncode == 0, res.stderr
    assert "RO_PIN_OK" in res.stdout
    assert (rw_hole / "writable.txt").read_text() == "hole-ok"


@needs_seatbelt
def test_live_seal_hides_siblings_blocks_existence_probe_and_allows_c_realpath(
    tmp_path,
):
    wt = tmp_path / "wt"
    scratch = tmp_path / "scratch"
    sealed = tmp_path / "worktrees"
    sib = sealed / "sibling"
    hole = sealed / "nested" / "current"
    for d in (wt, scratch, sib, hole):
        d.mkdir(parents=True)
    secret = sib / "config.worktree"
    secret.write_text("secret")
    target = hole / "index"
    target.write_text("ok")

    sb = SeatbeltSandbox(
        root=wt,
        binds=(
            *_python_binds(),
            Bind(scratch, RW),
            Seal(sealed, allow=(hole,)),
            Bind(hole, RW),
        ),
        env=_box_env(scratch),
        unshare_net=True,
    )
    code = (
        "import os, subprocess\n"
        f"assert not os.access({str(secret)!r}, os.F_OK)\n"
        f"assert not os.access({str(sib)!r}, os.F_OK)\n"
        "for bad in [\n"
        f"    lambda: os.listdir({str(sealed)!r}),\n"
        f"    lambda: os.listdir({str(sealed / 'nested')!r}),\n"
        f"    lambda: os.stat({str(secret)!r}),\n"
        f"    lambda: os.mkdir({str(sealed / 'planted')!r}),\n"
        "]:\n"
        "    try:\n"
        "        bad(); raise SystemExit('SEAL_LEAK')\n"
        "    except PermissionError:\n"
        "        pass\n"
        f"rp = subprocess.run(['/bin/realpath', {str(target)!r}], capture_output=True, text=True)\n"
        "assert rp.returncode == 0, rp.stderr\n"
        f"open({str(hole / 'lock')!r}, 'w').write('1')\n"
        f"os.unlink({str(hole / 'lock')!r})\n"
        "print('SEAL_OK')\n"
    )
    res = _run_sb(sb, [sys.executable, "-c", code])
    assert res.returncode == 0, res.stderr
    assert "SEAL_OK" in res.stdout


@needs_seatbelt
def test_live_git_binds_block_rename_and_missing_ancestor_while_commits_and_gc_work(
    tmp_path,
):
    main = tmp_path / "main"
    main.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.name", "Host"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.email", "h@e.invalid"], cwd=main, check=True)
    (main / "f.txt").write_text("base\n")
    subprocess.run(["git", "add", "f.txt"], cwd=main, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=main, check=True)

    # Create a sibling worktree with an unreferenced detached commit, plus the
    # boxed worktree.
    sib = tmp_path / "sib"
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(sib)], cwd=main, check=True
    )
    (sib / "sib.txt").write_text("sibling-only\n")
    subprocess.run(["git", "add", "sib.txt"], cwd=sib, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "sib"], cwd=sib, check=True)
    sib_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=sib,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "job", str(wt), "HEAD"],
        cwd=main,
        check=True,
    )
    # Create EnsurePath guard targets just as `Box._ensure_host_paths` does.
    for e in git_host_files(wt, pin_packs=True):
        if e.is_dir:
            e.path.mkdir(parents=True, exist_ok=True)
        else:
            e.path.parent.mkdir(parents=True, exist_ok=True)
            e.path.touch(exist_ok=True)

    sb = SeatbeltSandbox(
        root=wt,
        binds=(*_python_binds(), Bind(scratch, RW), *git_binds(wt, pin_packs=True)),
        env=_box_env(scratch),
        unshare_net=True,
    )
    code = (
        "import os, subprocess\n"
        f"main_git = {str(main / '.git')!r}\n"
        "for bad in [\n"
        "    lambda: os.rename('.git', '.git.bak'),\n"
        "    lambda: open(main_git + '/config', 'a'),\n"
        "    lambda: open(main_git + '/hooks/pre-commit', 'w'),\n"
        "    lambda: os.rename(main_git + '/objects', main_git + '/objects.bak'),\n"
        "    lambda: os.rename(main_git + '/objects/info', main_git + '/objects/info.bak'),\n"
        "    lambda: os.rmdir(main_git + '/objects/info'),\n"
        "]:\n"
        "    try:\n"
        "        bad(); raise SystemExit('GIT_PIN_BYPASSED')\n"
        "    except PermissionError:\n"
        "        pass\n"
        "open('work.txt', 'w').write('change\\n')\n"
        "subprocess.run(['git', 'add', 'work.txt'], check=True)\n"
        "subprocess.run(['git', 'commit', '-q', '-m', 'boxed'], check=True)\n"
        "subprocess.run(['git', 'gc', '--prune=now'], check=False)\n"
        "print('GIT_OK')\n"
    )
    res = _run_sb(sb, [sys.executable, "-c", code])
    assert res.returncode == 0, res.stderr
    assert "GIT_OK" in res.stdout
    alive = subprocess.run(["git", "cat-file", "-e", sib_sha], cwd=main, check=False)
    assert alive.returncode == 0

    # Also verify missing-ancestor `file-write-create` protection when an
    # intermediate directory above a pin does not exist at launch.
    plain = tmp_path / "plain"
    plain.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=plain, check=True)
    for e in git_host_files(plain):
        if e.is_dir:
            e.path.mkdir(parents=True, exist_ok=True)
        else:
            e.path.parent.mkdir(parents=True, exist_ok=True)
            e.path.touch(exist_ok=True)
    sb_plain = SeatbeltSandbox(
        root=plain,
        binds=(*_python_binds(), Bind(scratch, RW), *git_binds(plain)),
        env=_box_env(scratch),
        unshare_net=True,
    )
    # Render SBPL, then remove `objects/info` before launch to simulate an
    # absent intermediate directory at runtime.
    sbpl = sb_plain.render_sbpl()
    shutil.rmtree(plain / ".git" / "objects" / "info")
    plant_attempt = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-p",
            sbpl,
            sys.executable,
            "-c",
            (
                "import os\n"
                f"objs = {str(plain / '.git' / 'objects')!r}\n"
                "os.makedirs(objs + '/info.evil')\n"
                "open(objs + '/info.evil/alternates', 'w').write('/evil')\n"
                "try:\n"
                "    os.rename(objs + '/info.evil', objs + '/info')\n"
                "    raise SystemExit('MISSING_ANCESTOR_BYPASSED')\n"
                "except PermissionError:\n"
                "    print('MISSING_ANCESTOR_BLOCKED')\n"
            ),
        ],
        cwd=plain,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plant_attempt.returncode == 0, plant_attempt.stderr
    assert "MISSING_ANCESTOR_BLOCKED" in plant_attempt.stdout


@needs_seatbelt
def test_live_pinned_symlink_and_write_regex(tmp_path):
    wt = tmp_path / "wt"
    state = tmp_path / "state"
    rt = tmp_path / "rt"
    cache = tmp_path / "cache"
    sibling_cache = tmp_path / "cache_sibling"
    (cache / "store" / "env1").mkdir(parents=True)
    for d in (wt, state, rt, sibling_cache):
        d.mkdir()

    tok_src = rt / "token.json"
    tok_src.write_text('{"token":"placeholder"}')
    tok_dst = state / "auth.json"
    tok_dst.symlink_to(tok_src)

    pyvenv = cache / "store" / "env1" / "pyvenv.cfg"
    pyvenv.write_text("home = /usr\n")

    sb = SeatbeltSandbox(
        root=wt,
        binds=(
            *_python_binds(),
            Bind(state, RW),
            Bind(rt, RO),
            PinnedSymlink(tok_src, tok_dst),
            WriteRegex(
                cache,
                (r"/store/\.[^/]+\.lock|store/[^/]+/stamp",),
            ),
        ),
        env=_box_env(state),
        unshare_net=True,
    )
    code = (
        "import os, stat\n"
        f"assert 'placeholder' in open({str(tok_dst)!r}).read()\n"
        f"open({str(state / 'cfg.tmp')!r}, 'w').write('ok')\n"
        f"os.rename({str(state / 'cfg.tmp')!r}, {str(state / 'cfg.json')!r})\n"
        f"open({str(cache / 'store' / '.cpython.lock')!r}, 'w').write('l')\n"
        f"open({str(cache / 'store' / 'env1' / 'stamp')!r}, 'w').write('s')\n"
        "for bad in [\n"
        f"    lambda: open({str(tok_dst)!r}, 'w'),\n"
        f"    lambda: os.unlink({str(tok_dst)!r}),\n"
        f"    lambda: os.lchflags({str(tok_dst)!r}, stat.UF_IMMUTABLE),\n"
        f"    lambda: os.rename({str(state / 'cfg.json')!r}, {str(tok_dst)!r}),\n"
        f"    lambda: open({str(pyvenv)!r}, 'w'),\n"
        f"    lambda: open({str(cache / 'evil.lock')!r}, 'w'),\n"
        f"    lambda: open({str(sibling_cache)!r} + 'store/x/stamp', 'w'),\n"
        "]:\n"
        "    try:\n"
        "        bad(); raise SystemExit('UNEXPECTED_WRITE')\n"
        "    except (PermissionError, FileNotFoundError):\n"
        "        pass\n"
        "print('SYM_AND_REGEX_OK')\n"
    )
    res = _run_sb(sb, [sys.executable, "-c", code])
    assert res.returncode == 0, res.stderr
    assert "SYM_AND_REGEX_OK" in res.stdout


@needs_seatbelt
def test_live_network_tcp_port_unix_socket_and_local_loopback(tmp_path):
    wt = tmp_path / "wt"
    rt = tmp_path / "rt"
    scratch = tmp_path / "scratch"
    for d in (wt, rt, scratch):
        d.mkdir()

    allowed_tcp = socket.socket()
    allowed_tcp.bind(("127.0.0.1", 0))
    allowed_tcp.listen(2)
    allowed_port = allowed_tcp.getsockname()[1]

    blocked_tcp = socket.socket()
    blocked_tcp.bind(("127.0.0.1", 0))
    blocked_tcp.listen(1)
    blocked_port = blocked_tcp.getsockname()[1]

    allowed_unix_path = rt / "ok.sock"
    blocked_unix_path = rt / "no.sock"
    u_ok = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    u_ok.bind(str(allowed_unix_path))
    u_ok.listen(1)
    u_no = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    u_no.bind(str(blocked_unix_path))
    u_no.listen(1)

    try:
        sb = SeatbeltSandbox(
            root=wt,
            binds=(*_python_binds(), Bind(scratch, RW), Bind(rt, RO)),
            env=_box_env(scratch),
            unshare_net=True,
            tcp_ports=(allowed_port,),
            unix_sockets=(allowed_unix_path,),
        )
        code = (
            "import socket\n"
            f"socket.create_connection(('127.0.0.1', {allowed_port}), 2).close()\n"
            "u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            f"u.connect({seatbelt_lexpath(allowed_unix_path)!r}); u.close()\n"
            "for bad in [\n"
            f"    lambda: socket.create_connection(('127.0.0.1', {blocked_port}), 1),\n"
            "    lambda: socket.create_connection(('1.1.1.1', 443), 1),\n"
            f"    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', {allowed_port})),\n"
            "]:\n"
            "    try:\n"
            "        bad(); raise SystemExit('NET_LEAK')\n"
            "    except OSError:\n"
            "        pass\n"
            "u2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "try:\n"
            f"    u2.connect({seatbelt_lexpath(blocked_unix_path)!r}); raise SystemExit('UNIX_LEAK')\n"
            "except PermissionError:\n"
            "    u2.close()\n"
            "print('NET_STRICT_OK')\n"
        )
        res = _run_sb(sb, [sys.executable, "-c", code])
        assert res.returncode == 0, res.stderr
        assert "NET_STRICT_OK" in res.stdout

        # With `allow_local_loopback=True`, ephemeral in-box server bind+connect
        # works while external IPs remain blocked.
        sb_lo = SeatbeltSandbox(
            root=wt,
            binds=(*_python_binds(), Bind(scratch, RW)),
            env=_box_env(scratch),
            unshare_net=True,
            allow_local_loopback=True,
        )
        lo_code = (
            "import socket\n"
            "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
            "c = socket.create_connection(('127.0.0.1', srv.getsockname()[1]), 2)\n"
            "a, _ = srv.accept(); c.close(); a.close(); srv.close()\n"
            "for bad in [\n"
            "    lambda: socket.create_connection(('1.1.1.1', 443), 1),\n"
            "    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 53)),\n"
            "]:\n"
            "    try:\n"
            "        bad(); raise SystemExit('LOOPBACK_LEAK')\n"
            "    except OSError:\n"
            "        pass\n"
            "print('LOCAL_LOOPBACK_OK')\n"
        )
        res_lo = _run_sb(sb_lo, [sys.executable, "-c", lo_code])
        assert res_lo.returncode == 0, res_lo.stderr
        assert "LOCAL_LOOPBACK_OK" in res_lo.stdout
    finally:
        for s in (allowed_tcp, blocked_tcp, u_ok, u_no):
            s.close()
        allowed_unix_path.unlink(missing_ok=True)
        blocked_unix_path.unlink(missing_ok=True)


@needs_seatbelt
def test_live_host_ipc_keychain_procargs_and_tiocsti_are_blocked(tmp_path):
    wt = tmp_path / "wt"
    scratch = tmp_path / "scratch"
    wt.mkdir()
    scratch.mkdir()

    sb = SeatbeltSandbox(
        root=wt,
        binds=(*_python_binds(), Bind(scratch, RW)),
        env=_box_env(scratch),
        unshare_net=True,
    )
    for cmd in (
        ["/usr/bin/open", "-a", "Calculator"],
        ["/usr/bin/pbpaste"],
        ["/bin/launchctl", "list"],
        ["/usr/bin/security", "find-generic-password", "-s", "aisan-nonexistent"],
    ):
        assert _run_sb(sb, cmd).returncode != 0

    if Path("/etc/resolv.conf").exists():
        assert _run_sb(sb, ["/bin/cat", "/etc/resolv.conf"]).returncode == 0

    # Verify host process KERN_PROCARGS2 inspection is denied (`process-info*`).
    host_pid = os.getpid()
    proc_probe = _run_sb(
        sb,
        [
            sys.executable,
            "-c",
            (
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "def procargs(pid):\n"
                "    mib = (ctypes.c_int * 3)(1, 49, pid)\n"  # CTL_KERN=1, KERN_PROCARGS2=49
                "    sz = ctypes.c_size_t(65536)\n"
                "    buf = ctypes.create_string_buffer(65536)\n"
                "    return libc.sysctl(mib, 3, buf, ctypes.byref(sz), None, 0)\n"
                "assert procargs(os.getpid()) == 0\n"
                f"assert procargs({host_pid}) != 0\n"
                "print('PROCARGS_BLOCKED')\n"
            ),
        ],
    )
    assert proc_probe.returncode == 0, proc_probe.stderr
    assert "PROCARGS_BLOCKED" in proc_probe.stdout

    # Verify TIOCSTI is blocked by Seatbelt (`hid-control`) even on a live PTY.
    master_fd, slave_fd = pty.openpty()
    try:
        tty_name = os.ttyname(slave_fd)
        tiocsti = getattr(termios, "TIOCSTI", 0x80017472)
        probe = subprocess.run(
            [
                *sb.wrapper(),
                sys.executable,
                "-c",
                (
                    "import fcntl, os\n"
                    f"fd = os.open({tty_name!r}, os.O_RDWR)\n"
                    "try:\n"
                    f"    fcntl.ioctl(fd, {tiocsti}, b'X')\n"
                    "    raise SystemExit('TIOCSTI_ALLOWED')\n"
                    "except OSError:\n"
                    "    print('TIOCSTI_BLOCKED')\n"
                    "finally:\n"
                    "    os.close(fd)\n"
                ),
            ],
            stdin=slave_fd,
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        assert "TIOCSTI_BLOCKED" in probe.stdout
    finally:
        os.close(master_fd)
        os.close(slave_fd)
