# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Launch a payload with isolated relays or shared-network credentials.

In isolated mode, ``relays.json`` lists the relay listeners. The launcher starts
them, spawns the payload, and returns its status. In shared-network mode,
``client-env.json`` supplies the proxy port and token. The launcher adds them to
the environment and executes the payload without exposing the token in argv.

The isolated launcher must stay alive because its event loop serves the relays.
It therefore spawns the payload and forwards signals to it.

Bubblewrap remains PID 1 so it can reap orphaned grandchildren. The launcher is
PID 2 and waits only for its direct child.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

from .proxy import relay
from .runtime import CLIENT_ENV_NAME, read_client_env, read_manifest
from .sandbox import RO, Bind, BindSpec, system_ro_roots, through_system_symlink

# Forward terminal and supervisor termination signals to the payload.
_FORWARD = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def interpreter_roots(python: Path) -> list[Path]:
    """Return every root traversed by an interpreter symlink chain.

    Each root is inferred from path depth, assuming ``<root>/bin/<exe>``. The
    assumption holds for an installation prefix and fails for a link farm such
    as a personal ``~/bin``, where the inferred root is the home directory. The
    two are indistinguishable by shape, so a caller must establish that the
    executable really sits in a prefix before trusting the result.

    Use this only for a foreign interpreter, whose ``sys.prefix`` cannot be read
    without executing it. `launcher_binds` asks its own interpreter instead.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(root: Path) -> None:
        if root not in seen:
            seen.add(root)
            roots.append(root)

    p = python
    for _ in range(10):  # Bound cycles and unexpectedly long chains.
        try:
            if len(p.parents) >= 2:
                add(p.parents[1])
            if not p.is_symlink():
                # Resolve directory symlinks that weren't visible as file links.
                add(p.resolve().parents[1] if len(p.parents) >= 2 else p.resolve())
                break
            target = Path(os.readlink(p))
            p = target if target.is_absolute() else (p.parent / target)
        except OSError:
            break
    return roots


def interpreter_chain_dirs(python: Path) -> list[Path]:
    """Return the directory holding every name on an interpreter's link chain.

    ``exec`` resolves the chain inside the box, so each name on it must exist at
    its own path. A hop through a directory outside every prefix has no other
    mount and would fail with ENOENT. Hops beneath a prefix are returned too:
    binding one twice is harmless, and whether the prefix bind reaches it is a
    question about the whole mount list that this function cannot answer.

    Directory symlinks in intermediate components need no entry of their own: a
    bind resolves its source, so the hop's directory carries the target's
    contents to the path the chain names. This preserves executable lookup,
    not stdlib lookup relative to the venv home; `launcher_binds` handles that.
    """
    dirs: list[Path] = []
    seen: set[Path] = set()

    p = python
    for _ in range(10):  # Bound cycles and unexpectedly long chains.
        try:
            if p.parent not in seen:
                seen.add(p.parent)
                dirs.append(p.parent)
            if not p.is_symlink():
                break
            target = Path(os.readlink(p))
            p = target if target.is_absolute() else (p.parent / target)
        except OSError:
            break
    return dirs


def own_source_root() -> Path | None:
    """This package's own source tree, when it is installed editable.

    A normal installation lives in site-packages and is covered by the venv
    bind. An editable installation uses a ``.pth`` file to reach source outside
    that bind, so the source root must also be mounted.

    Derive the root from this module so unrelated editable packages stay hidden.
    """
    root = Path(__file__).resolve().parent.parent
    # A site-packages parent is already covered by the venv bind.
    return root if (root / "aisan").is_dir() and not _in_site_packages(root) else None


def _in_site_packages(path: Path) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in path.parts)


def launcher_binds(
    python: Path | None = None,
    prefixes: tuple[Path, Path] | None = None,
    *,
    home: Path | None = None,
) -> list[BindSpec]:
    """Return read-only binds for the in-box aisan launcher.

    The launcher runs inside the box and requires its interpreter, that
    interpreter's runtime trees, and its imported source even when the payload
    doesn't use Python. Including these binds in the launcher policy avoids
    relying on payload-specific mounts.

    Two separate requirements produce the list:

    * ``exec`` walks the interpreter's symlink chain, so every name on it must
      exist in the box. `interpreter_chain_dirs` supplies those directories.
    * CPython locates its stdlib, ``pyvenv.cfg``, and site-packages under
      ``sys.prefix`` and ``sys.base_prefix``. Read both from the interpreter
      rather than inferring a root from path depth. The venv's configured home
      can use an alias absent from those prefixes; include its parent only
      when it resolves to the reported base prefix.

    Omit a path the fixed system surface already mounts. Re-declaring one adds
    no mount, so leaving it in would misreport the policy. A path under one of
    the surface's merged-`/usr` links, such as `/bin`, is renamed to the
    directory the box has: the sandbox refuses a destination through a link.

    Always bind `own_source_root`. Asking whether another bind already covers it
    would compare path prefixes, and containment does not answer whether a path
    is readable: a later tmpfs can mask a bound ancestor. One nested read-only
    bind costs less than a predicate that cannot be right.

    `prefixes` and `home` override the interpreter's own report so tests can
    describe another layout. Production reads `sys`, including the venv home
    recorded by CPython's site initialization in `sys._home`.
    """
    exe = python or Path(sys.executable)
    prefix, base = prefixes or (Path(sys.prefix), Path(sys.base_prefix))
    configured_home = home or getattr(sys, "_home", None)
    roots = [prefix, base]
    if configured_home:
        alias = Path(configured_home).parent
        # A bin-only bind loses the alias's adjacent stdlib. Path depth alone
        # cannot prove a prefix: a personal ~/bin would select the whole home.
        if alias.resolve(strict=True) == base.resolve(strict=True):
            roots.append(alias)
    system = system_ro_roots()
    binds: list[BindSpec] = []
    seen: set[Path] = set()
    for hop in (*roots, *interpreter_chain_dirs(exe)):
        path = through_system_symlink(hop)
        # An alias is a separate publication, even when its source is /usr.
        if path in seen or path in system:
            continue
        seen.add(path)
        binds.append(Bind(path, RO))
    own = own_source_root()
    if own is not None:
        binds.append(Bind(own, RO))
    return binds


def launch_prefix(runtime_dir: Path, python: Path | None = None) -> list[str]:
    """The argv prefix that supplies a payload's egress transport.

    An absolute ``python -m`` path selects the interpreter covered by
    ``launcher_binds`` without a PATH lookup or shim. A box may contain several
    venv bin directories whose order would otherwise choose the aisan version.
    """
    exe = python or Path(sys.executable)
    return [str(exe), "-m", f"{__package__}.launch", str(runtime_dir), "--"]


def exit_status(code: int) -> int:
    """A `wait` status as the number a shell would report.

    ``wait`` and ``subprocess.run`` report a signal as ``-N``, while shells
    report ``128 + N``. Convert it so callers see the same status with or without
    the launcher.
    """
    return code if code >= 0 else 128 - code


async def _run(runtime_dir: Path, cmd: list[str]) -> int:
    """Inject shared state and exec, or serve isolated relays around `cmd`."""
    if (runtime_dir / CLIENT_ENV_NAME).exists():
        env = {**os.environ, **read_client_env(runtime_dir)}
        os.execvpe(cmd[0], cmd, env)  # noqa: S606 - argv exec, no shell

    servers = []
    for entry in read_manifest(runtime_dir):
        # Fail startup if a relay can't listen; every call to that backend would
        # otherwise fail later with a connection error.
        servers.append(
            await relay.serve(Path(str(entry["socket"])), int(str(entry["port"])))
        )
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        loop = asyncio.get_running_loop()
        for sig in _FORWARD:
            # Keep relays alive while the payload handles the forwarded signal.
            loop.add_signal_handler(sig, _forward, proc, sig)
        try:
            code = await proc.wait()
        finally:
            for sig in _FORWARD:
                loop.remove_signal_handler(sig)
        return exit_status(code)
    finally:
        for server in servers:
            server.close()


def _forward(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    # The payload may exit before the signal is forwarded.
    with contextlib.suppress(ProcessLookupError):
        proc.send_signal(sig)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: aisan-box-launch <runtime_dir> -- <cmd...>"""
    usage = "usage: aisan-box-launch <runtime_dir> -- <cmd...>"
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or "--" not in args:
        sys.exit(usage)
    sep = args.index("--")
    cmd = args[sep + 1 :]
    if sep != 1 or not cmd:
        sys.exit(usage)
    sys.exit(asyncio.run(_run(Path(args[0]), cmd)))


if __name__ == "__main__":
    main()
