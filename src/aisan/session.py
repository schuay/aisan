# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Shared lifecycle helpers for interactive session launchers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import userbinds
from .box import Box
from .egress import known_credential_paths
from .egress.base import Backend, EgressProfile, PreflightError, credential_overlap
from .explain import assembly_refusal, explain
from .launch import exit_status, interpreter_roots
from .presets import EGRESS_PROFILES, GRANTS
from .private import private_root
from .sandbox import (
    DARWIN_DEFAULT_BINS,
    RO,
    Bind,
    BindOver,
    BindSpec,
    host_platform,
)
from .seatbelt import darwin_system_ro_roots, seatbelt_lexpath, seatbelt_path
from .session_mcp import MCPAllowlist, SessionMCP, _shared_roots, mcp_ro_binds
from .spec import SECURITY_SERVER, BoxSpec, Grant
from .statedir import (
    StateBusyError,
    is_state_dir_locked,
    lock_state_dir,
    planted_credentials,
    prepare_state_dir,
    write_sealed,
)
from .upstream import Transport

_TERM_PASSTHROUGH = ("TERM", "COLORTERM", "LANG", "LC_ALL")


class LaunchRefused(Exception):
    """A launcher refusal and the exit code to report."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def mcp_launcher_binds(mcp: SessionMCP | None) -> tuple[Path, ...]:
    """Resolve MCP launcher binds and convert resolution errors to refusals.

    Resolution fails before the box starts when a command is missing, its tool
    root can't be proven, or a required bind would expose a credential store.
    """
    if mcp is None:
        return ()
    try:
        return mcp_ro_binds(mcp)
    except (FileNotFoundError, ValueError) as e:
        raise LaunchRefused(f"refused: {e}") from e


def upstream_transport() -> Transport:
    """Read the host proxies' upstream route from the host environment."""
    try:
        return Transport.from_env()
    except ValueError as e:
        raise LaunchRefused(f"refused: {e}") from e


def git_config_binds() -> list[BindSpec]:
    """Bind the global Git config file without exposing adjacent credentials.

    The file supplies commit identity. Binding its directory could expose the
    plaintext credential store. Resolve the host source through
    ``XDG_CONFIG_HOME``, then bind it to the default path used by the box's
    cleared environment. Return no binds when the host has no global config.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    host = (Path(xdg) if xdg else Path.home() / ".config") / "git" / "config"
    if not host.is_file():
        return []
    if host_platform() == "darwin":
        return [Bind(host.resolve(), RO)]
    box_dst = Path.home() / ".config" / "git" / "config"
    if host.resolve() == box_dst.resolve():
        return [Bind(host, RO)]
    return [BindOver(host, box_dst)]


def git_config_env() -> tuple[tuple[str, str], ...]:
    """Return `GIT_CONFIG_GLOBAL` on Darwin when `git_config_binds()` matches."""
    if host_platform() != "darwin":
        return ()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    host = (Path(xdg) if xdg else Path.home() / ".config") / "git" / "config"
    if not host.is_file():
        return ()
    return (("GIT_CONFIG_GLOBAL", seatbelt_path(host.resolve())),)


_DARWIN_DEFAULT_BINS_SET = frozenset(DARWIN_DEFAULT_BINS)


def _node_package_mounts(
    real: Path, shared: frozenset[Path]
) -> tuple[list[Path], list[Path]]:
    """Return `(mounts, path_dirs)` when `real` resolves inside `node_modules`."""
    for idx, part in enumerate(real.parts):
        if part != "node_modules":
            continue
        nm_dir = Path(*real.parts[: idx + 1])
        rem = real.parts[idx + 1 :]
        if not rem:
            return [], []
        pkg_root = (
            nm_dir / rem[0] / rem[1]
            if rem[0].startswith("@") and len(rem) >= 2
            else nm_dir / rem[0]
        )
        if not (pkg_root / "package.json").is_file() or pkg_root in shared:
            return [], []
        mounts = [pkg_root]
        extra_paths: list[Path] = []
        if nm_dir.parent.name == "lib":
            node_bin = nm_dir.parent.parent / "bin"
            if (node_bin / "node").is_file() and node_bin not in shared:
                mounts.append(node_bin)
                extra_paths.append(node_bin)
        return mounts, extra_paths
    return [], []


def _symlink_hops(entry: Path) -> list[Path]:
    """Return each symlink path along `entry`'s chain without its parent dir."""
    hops: list[Path] = []
    seen: set[Path] = set()
    p = entry
    for _ in range(10):
        try:
            if not p.is_symlink() or p in seen:
                break
            seen.add(p)
            hops.append(p)
            target = Path(os.readlink(p))
            p = target if target.is_absolute() else (p.parent / target)
        except OSError:
            break
    return hops


def _darwin_target_roots(
    real: Path, shared: frozenset[Path]
) -> tuple[list[Path], list[Path]]:
    """Locate self-contained tool roots for `real` by explicit layout markers."""
    node_mounts, node_paths = _node_package_mounts(real, shared)
    if node_mounts:
        return node_mounts, node_paths
    for anc in real.parents:
        if anc in shared or anc in (Path("/"), Path("/opt"), Path("/Applications")):
            continue
        if (anc / "pyvenv.cfg").is_file():
            mounts = [anc]
            for py_name in ("python", "python3"):
                py = anc / "bin" / py_name
                if py.is_symlink():
                    mounts.extend(_symlink_hops(py))
                    mounts.extend(p for p in interpreter_roots(py) if p not in shared)
            return mounts, []
        if anc.suffix == ".app":
            return [anc], []
    return [real], []


def darwin_executable_grant(
    executable_path: Path | None, *, platform: str | None = None
) -> Grant:
    """Grant read-only visibility and PATH discovery for a Darwin client binary.

    Homebrew and system binaries sit inside Seatbelt's fixed read-only surface
    and default PATH. Self-contained tool roots (`pyvenv.cfg` virtualenvs,
    `node_modules/<pkg>` + `bin/node` prefixes, or `.app` bundles) mount their
    explicit root; standalone binaries mount only the resolved executable and
    its symlink hops so shared trees such as `~/.cargo/bin` or `~/.local/bin`
    are never mounted whole.
    """
    target_platform = host_platform() if platform is None else platform
    if target_platform != "darwin" or executable_path is None:
        return Grant()
    bin_dir = executable_path.parent
    real = executable_path.resolve()
    sys_roots = darwin_system_ro_roots()
    shared = _shared_roots(Path.home().resolve())

    def covered_by_surface(target: Path) -> bool:
        canon = Path(
            seatbelt_lexpath(target) if target.is_symlink() else seatbelt_path(target)
        )
        return any(canon == r or canon.is_relative_to(r) for r in sys_roots)

    refused_stores = (
        *known_credential_paths(),
        *userbinds._refused_key_stores(),
    )
    mount_candidates: list[Path] = []
    extra_path_dirs: list[Path] = []
    if not covered_by_surface(executable_path):
        for hop in _symlink_hops(executable_path):
            if not covered_by_surface(hop) and hop not in shared:
                mount_candidates.append(hop)
    if not covered_by_surface(real):
        target_mounts, target_paths = _darwin_target_roots(real, shared)
        for cand in target_mounts:
            if not covered_by_surface(cand) and cand not in shared:
                mount_candidates.append(cand)
        extra_path_dirs.extend(target_paths)

    candidates = [
        p
        for p in dict.fromkeys(mount_candidates)
        if credential_overlap((p,), refused_stores) is None
    ]
    dir_roots = tuple(
        Path(seatbelt_path(p)) for p in candidates if not p.is_symlink() and p.is_dir()
    )
    safe_mounts = tuple(
        p
        for p in candidates
        if not (
            p.is_symlink()
            and any(Path(seatbelt_lexpath(p)).is_relative_to(d) for d in dir_roots)
        )
    )
    binds = tuple(Bind(p, RO, guard=False) for p in safe_mounts)
    raw_paths = [
        p
        for p in dict.fromkeys((bin_dir, *extra_path_dirs))
        if str(p) not in _DARWIN_DEFAULT_BINS_SET
    ]
    return Grant(binds=binds, path=tuple(raw_paths))


def repo_key(repo: Path) -> str:
    """Return a stable key that distinguishes repositories with the same name."""
    resolved = repo.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:32]
    name = resolved.name or "root"
    return f"{name}-{digest}"


def state_dir(client: str, repo: Path, *, home: Path | None = None) -> Path:
    """Return the persistent state directory for a client and repository."""
    base = home if home is not None else Path.home()
    return base / ".cache" / f"aisan-{client}" / repo_key(repo)


def mirror_user_memory(source: Path, dst: Path) -> None:
    """Copy host instructions into the boxed client's redirected config.

    The state directory is writable in the box, so ``write_sealed`` prevents a
    planted destination symlink from redirecting the host-side write. Refresh
    the copy at launch, but preserve the destination when the source is absent.
    """
    if source.exists():
        write_sealed(dst, source.read_bytes())


def box_id(client: str, repo: Path) -> str:
    """Return a unique box ID containing a stable repository hint."""
    return f"{client}-i-{repo_key(repo)}-{secrets.token_hex(8)}"


def interactive_parser(prog: str, executable: str) -> argparse.ArgumentParser:
    """Build the common strict argument parser for interactive launchers."""
    parser = argparse.ArgumentParser(
        prog=prog,
        allow_abbrev=False,
        epilog=f"everything after a literal `--` is forwarded to {executable}.",
    )
    parser.add_argument(
        "repo", nargs="?", default=".", help="git repo to root the box at"
    )
    parser.add_argument(
        "--binds",
        type=Path,
        metavar="FILE",
        action="append",
        help="user bind spec, TOML (keys: ro, rw, overlay, path, mcp, include);"
        " merged with the preset's binds: deeper paths win, the stricter mode"
        " wins at one path, and entries under a preset guard are refused. `mcp`"
        " names the host MCP servers this box may start; without it, none do."
        " Repeatable (see aisan.userbinds)",
    )
    parser.add_argument(
        "--egress",
        metavar="NAME",
        action="append",
        choices=sorted(EGRESS_PROFILES),
        help="add a named egress profile's backends and mounts to the box"
        f" (one of: {', '.join(sorted(EGRESS_PROFILES))}). Repeatable; a"
        " profile serving a proxy needs the isolated network, so not with"
        " --net",
    )
    parser.add_argument(
        "--grant",
        metavar="NAME",
        action="append",
        choices=sorted(GRANTS),
        help="add a named grant: the mounts, PATH entries and environment a"
        " tree needs inside a box with no network"
        f" (one of: {', '.join(sorted(GRANTS))}). Repeatable",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the resolved profile and exit; nothing runs",
    )
    parser.add_argument(
        "--net",
        action="store_true",
        help="share host networking (internet, LAN/VPN, and host loopback)",
    )
    return parser


def parse_interactive_args(
    parser: argparse.ArgumentParser, argv: list[str]
) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher args before ``--`` and preserve payload args after it."""
    if "--" not in argv:
        return parser.parse_args(argv), []
    split = argv.index("--")
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def mcp_notice(mcp: SessionMCP) -> str | None:
    """Describe what the host MCP declarations did in this box, or return None.

    Name servers that carry environment values because those values may include
    credentials readable by the box. Report withheld servers too: a box that
    names none is the default, and the operator otherwise has no sign that the
    host declared any. Name each withheld server's command, since that is one
    of the tokens a spec can select it by.
    """
    lines: list[str] = []
    if mcp.enabled:
        lines.append(
            f"NOTE: {len(mcp.names)} host MCP server(s) start inside the box:"
            f" {', '.join(mcp.names)}."
        )
        if mcp.env_names:
            lines.append(
                "      Declared environment values travel with them"
                f" ({', '.join(mcp.env_names)}); any credential there is readable"
                " in the box."
            )
    if mcp.withheld:
        withheld = ", ".join(f"{name} ({command})" for name, command in mcp.withheld)
        lines.append(
            f"NOTE: {len(mcp.withheld)} host MCP server(s) withheld; no bind"
            f" spec names them: {withheld}."
        )
        lines.append(
            "      Name one in a --binds spec's `mcp` key to start it in the"
            " box, by server name, command, or the command's full path."
        )
    if mcp.unmatched:
        lines.append(
            "NOTE: bind spec `mcp` entries name no local stdio server this host"
            f" declares: {', '.join(mcp.unmatched)}. Nothing starts for them;"
            " a remote or disabled declaration is never imported."
        )
    return "\n".join(lines) or None


def terminal_env() -> tuple[tuple[str, str], ...]:
    """Return terminal settings to restore after clearing the environment."""
    return tuple(
        (key, os.environ[key]) for key in _TERM_PASSTHROUGH if key in os.environ
    )


@dataclass(frozen=True)
class LauncherFlags:
    """One launch's resolved ``--egress``, ``--grant`` and ``--binds`` inputs.

    Reading is separate from applying because the MCP allowlist lives in the
    bind specs and a launcher needs it before it builds a spec: the selected
    servers decide which launcher binds and PATH entries the preset gets.
    """

    profiles: tuple[tuple[str, EgressProfile], ...] = ()
    grants: tuple[Grant, ...] = ()
    users: tuple[userbinds.UserSpec, ...] = ()
    #: The spec files as written, for the `--explain` report.
    files: tuple[Path, ...] = ()

    @property
    def mcp(self) -> MCPAllowlist:
        """Return the MCP servers the bind specs name, in file order."""
        return MCPAllowlist(
            tuple(dict.fromkeys(entry for user in self.users for entry in user.mcp))
        )

    def apply(self, spec: BoxSpec) -> BoxSpec:
        """Add the profiles, grants and user binds to ``spec``.

        Bind order carries no precedence. Each item applies on its own because
        `with_path_prefix` prepends and `with_env` appends: composing a
        flattened list instead would reverse PATH precedence between two grants.
        """
        for name, profile in self.profiles:
            try:
                spec = spec.with_egress(list(profile.backends))
            except ValueError as e:
                raise LaunchRefused(f"--egress {name}: {e}") from e
            spec = spec.with_binds(list(profile.binds))
        for grant in self.grants:
            spec = (
                spec.with_binds(list(grant.binds))
                .with_path_prefix(grant.path)
                .with_env(grant.env)
                .with_ensure(grant.ensure)
                .with_local_loopback(grant.allow_local_loopback)
                .with_mach_profiles(grant.mach_profiles)
            )
        for user in self.users:
            spec = spec.with_binds(user.binds).with_path_prefix(user.path)
        return spec


def resolve_launcher_flags(
    repo: Path,
    *,
    base_egress: tuple[Backend, ...],
    unshare_net: bool,
    egress_profiles: list[str] | None,
    grants: list[str] | None,
    binds: list[Path] | None,
) -> LauncherFlags:
    """Read the launcher's profile, grant and bind-spec flags.

    Report unavailable optional profiles and grants, and raise `LaunchRefused`
    for invalid configuration. Nothing is composed here; `LauncherFlags.apply`
    does that once the launcher has a spec.
    """
    # Ignore duplicate names from repeatable flags.
    profiles: list[tuple[str, EgressProfile]] = []
    for name in dict.fromkeys(egress_profiles or []):
        profile = EGRESS_PROFILES[name](repo)
        if not profile:
            # Optional profiles may be unavailable for this repository.
            print(
                f"NOTE: egress profile {name} found nothing to serve in {repo};"
                " the box gets no route from it.",
                file=sys.stderr,
            )
            continue
        # Each backend defines whether its transport is safe on host loopback.
        is_darwin = host_platform() == "darwin"
        if is_darwin:
            unsupported = [
                b.name
                for b in profile.backends
                if not b.supports_shared_net
                and not (unshare_net and b.supports_direct_unix_socket)
            ]
            if unsupported:
                if not unshare_net:
                    raise LaunchRefused(
                        f"--egress {name} needs an isolated box: {', '.join(unsupported)}"
                        " cannot serve unauthenticated on host networking. Drop --net."
                    )
                raise LaunchRefused(
                    f"--egress {name}: {', '.join(unsupported)} requires a Linux"
                    " network namespace and is not supported on macOS Seatbelt."
                )
        else:
            stranded = [b.name for b in profile.backends if not b.supports_shared_net]
            if stranded and not unshare_net:
                raise LaunchRefused(
                    f"--egress {name} needs the box's own network: {', '.join(stranded)}"
                    " would otherwise answer on the host's loopback, unauthenticated."
                    " Drop --net."
                )
        if profile.notice:
            print(profile.notice, file=sys.stderr)
        profiles.append((name, profile))

    # Apply each named grant once.
    resolved: list[Grant] = []
    for name in dict.fromkeys(grants or []):
        grant = GRANTS[name]()
        if not grant:
            # An unavailable optional tool grant doesn't prevent launching.
            print(
                f"NOTE: grant {name} found nothing on this host;"
                " the box gets none of it.",
                file=sys.stderr,
            )
            continue
        if SECURITY_SERVER in grant.mach_profiles and unshare_net:
            raise LaunchRefused(
                f"--grant {name} needs --net: it only serves TLS trust lookups."
            )
        resolved.append(grant)

    # The credential guard needs every backend the box will carry.
    egress = (*base_egress, *(b for _, profile in profiles for b in profile.backends))
    users: list[userbinds.UserSpec] = []
    for spec_file in binds or []:
        try:
            users.append(userbinds.load(spec_file, egress=egress))
        except ValueError as e:
            raise LaunchRefused(f"binds: {e}") from e
        except OSError as e:
            # Report unreadable files through the same launcher refusal path.
            raise LaunchRefused(f"binds: cannot read {spec_file}: {e}") from e
    return LauncherFlags(
        tuple(profiles), tuple(resolved), tuple(users), tuple(binds or [])
    )


async def run_interactive(
    *,
    client: str,
    harness: str,
    executable: str,
    repo: Path,
    state: Path,
    spec: BoxSpec,
    command: Callable[[Box], list[str]],
    binary: Callable[[], Path | None],
    flags: LauncherFlags,
    explain_only: bool,
    prepare: Callable[[], None] | None = None,
    mcp: SessionMCP | None = None,
) -> int:
    """Apply launcher options, then explain or run the box."""
    is_darwin = host_platform() == "darwin"
    found_binary: Path | None = None
    if not explain_only:
        found_binary = binary()
    exe_candidate = (
        (Path(hit) if (hit := shutil.which(executable)) else None)
        if explain_only and is_darwin
        else found_binary
    )
    if isinstance(spec, BoxSpec):
        exe_grant = darwin_executable_grant(exe_candidate)
        if exe_grant:
            spec = (
                spec.with_binds(list(exe_grant.binds))
                .with_path_prefix(exe_grant.path)
                .with_env(exe_grant.env)
            )
    try:
        spec = flags.apply(spec)
    except LaunchRefused as e:
        print(e, file=sys.stderr)
        return e.code

    notice = mcp_notice(mcp) if mcp is not None else None
    if explain_only:
        if notice:
            print(notice, file=sys.stderr)
        inputs = (
            ("harness", harness),
            ("repo", str(repo)),
            (
                "binds",
                ", ".join(map(str, flags.files)) if flags.files else "(none)",
            ),
        )
        locked = is_darwin and isinstance(spec, BoxSpec) and is_state_dir_locked(state)
        state_guard = contextlib.nullcontext() if locked else staged_directory(state)
        box = Box(spec, box_id=box_id(client, repo))
        try:
            with state_guard, box.staged(stage_symlinks=not locked):
                print(
                    explain(
                        box,
                        inputs=inputs,
                        color=sys.stdout.isatty(),
                    ),
                    end="",
                )
                # Probe while generated bind-over sources still exist.
                refused = assembly_refusal(box) is not None
        except (
            ValueError,
            FileNotFoundError,
            FileExistsError,
            NotADirectoryError,
        ) as exc:
            print(f"\n== BOX ASSEMBLY REFUSED ==\n  {exc}\n", end="")
            return 2
        return 2 if refused else 0

    if found_binary is None:
        tool = "sandbox-exec" if is_darwin else "bwrap"
        print(
            f"no `{executable}` on PATH: this would be an exec failure inside {tool}",
            file=sys.stderr,
        )
        return 2
    box = Box(spec, box_id=box_id(client, repo))

    # A credential planted in persistent state would enter every later box read-write.
    planted = planted_credentials(
        state,
        client,
        allowed_symlink_root=private_root() if is_darwin else None,
    )
    if planted:
        print(
            "refused: the state directory holds a credential written inside a"
            f" box: {', '.join(map(str, planted))}\n"
            "aisan never writes these; a session with host networking logged in"
            " from inside the box, and every later box for this repository"
            " would get that credential rw.\n"
            "delete the file, then log in on the HOST if the login is one you"
            " want.",
            file=sys.stderr,
        )
        return 2

    state_ctx = lock_state_dir(state) if is_darwin else contextlib.nullcontext()
    try:
        with state_ctx:
            prepare_state_dir(state)
            if prepare is not None:
                prepare()
            async with box:
                if not box.spec.unshare_net:
                    print(
                        "WARNING: box shares host networking: internet, LAN/VPN, and"
                        " host-local services are reachable; configured model"
                        " credential files are not mounted.",
                        file=sys.stderr,
                    )
                elif (
                    is_darwin
                    and isinstance(box.spec, BoxSpec)
                    and box.spec.allow_local_loopback
                ):
                    print(
                        "WARNING: box allows localhost TCP on macOS: external"
                        " networks and DNS are blocked, but unauthenticated"
                        " host-local TCP services on 127.0.0.1/::1 are reachable.",
                        file=sys.stderr,
                    )
                if notice:
                    print(notice, file=sys.stderr)
                done = await asyncio.to_thread(
                    subprocess.run,
                    box.command(command(box)),
                    env={**os.environ, **box.env},
                    check=False,
                )
                # Convert a negative signal return code to shell exit status.
                return exit_status(done.returncode)
    except StateBusyError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except PreflightError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError, FileExistsError, NotADirectoryError) as e:
        # Bind sources can disappear and mount resolution can still fail after preflight.
        print(f"refused: {e}", file=sys.stderr)
        return 2


@contextmanager
def staged_directory(path: Path) -> Iterator[None]:
    """Create ``path`` for inspection and remove newly created empty directories.

    Cleanup uses ``rmdir``, so it preserves existing directories and any files
    created concurrently.
    """
    created: list[Path] = []
    cursor = path
    while not cursor.exists():
        created.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        for directory in created:
            with contextlib.suppress(OSError):
                directory.rmdir()
