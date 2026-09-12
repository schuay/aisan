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
from .sandbox import RO, Bind, BindSpec

# Forward terminal and supervisor termination signals to the payload.
_FORWARD = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def interpreter_roots(python: Path) -> list[Path]:
    """Return every root traversed by an interpreter symlink chain.

    uv may connect a venv interpreter to a versioned installation through an
    unversioned directory symlink. Binding only the resolved target leaves that
    intermediate name absent from the box, so bind every hop.
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


def launcher_binds(python: Path | None = None) -> list[BindSpec]:
    """Return read-only binds for the in-box aisan launcher.

    The launcher runs inside the box and requires its interpreter, venv, and
    imported source even when the payload doesn't use Python. Including these
    binds in the launcher policy avoids relying on payload-specific mounts.

    The binds cover the unresolved venv, the interpreter symlink chain, and the
    package source root for editable installations.

    ``pyvenv.cfg`` distinguishes a venv from a system interpreter whose parent
    directory is already covered by ``interpreter_roots``.
    """
    exe = python or Path(sys.executable)
    venv = exe.parent.parent
    binds = [Bind(venv, RO)] if (venv / "pyvenv.cfg").is_file() else []
    covered = {Path(b.path) for b in binds}
    for root in interpreter_roots(exe):
        if root not in covered:
            covered.add(root)
            binds.append(Bind(root, RO))
    own = own_source_root()
    if own is not None and not any(own.is_relative_to(p) for p in covered):
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
