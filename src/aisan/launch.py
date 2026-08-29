# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The in-box launcher: relay isolated egress or inject shared-network state.

In isolated mode, `relays.json` says what to listen on and where to splice it;
the launcher binds every relay, spawns the payload, and returns its status. In
shared-network mode, `client-env.json` carries the dynamic proxy port and token;
the launcher adds them to the environment and execs the payload without
starting relays or exposing the token in argv.

Spawn, not exec, and that is the whole reason this is a process at all. The
relays live in THIS process's event loop, so replacing the image would take them
with it. The cost of spawning is one extra process in the box and the obligation
to forward signals, both of which are paid here.

The launcher is pid 2, not pid 1: bwrap keeps pid 1 for its own reaper (see
`box.py` on why `--as-pid-1` must never be passed), so an orphaned grandchild is
reaped there and this process only ever waits on its own child.
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

# Signals forwarded to the payload. Everything a terminal or a supervisor sends
# to end or interrupt a job: the launcher is in the middle of that path and a
# signal that stops here leaves the payload running with nobody watching.
_FORWARD = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def interpreter_roots(python: Path) -> list[Path]:
    """Every root a python symlink hops THROUGH, not just its final target.

    uv keeps two names for one interpreter: a versioned dir
    (cpython-3.12.12-linux-x86_64-gnu) and an unversioned symlink to it
    (cpython-3.12-...). A venv's bin/python points at the UNVERSIONED one, so
    binding only `.resolve()` binds the versioned dir and leaves the name the
    venv actually names absent in the box -- and bwrap then fails the exec with
    `execvp <venv>/bin/python3: No such file or directory`, which reads as a
    missing venv rather than a missing hop. Walk the chain and bind each root.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(root: Path) -> None:
        if root not in seen:
            seen.add(root)
            roots.append(root)

    p = python
    for _ in range(10):  # cycle/rogue-chain guard; real chains are 1-2 hops
        try:
            if len(p.parents) >= 2:
                add(p.parents[1])
            if not p.is_symlink():
                # The interpreter root itself may be reached through a symlinked
                # DIRECTORY rather than a symlinked file: uv writes
                # <...>/cpython-3.12 -> <...>/cpython-3.12.12 and the venv points
                # into the alias, so following only file links stops one hop
                # short of the real tree. resolve() closes any such gap.
                add(p.resolve().parents[1] if len(p.parents) >= 2 else p.resolve())
                break
            target = Path(os.readlink(p))
            p = target if target.is_absolute() else (p.parent / target)
        except OSError:
            break
    return roots


def own_source_root() -> Path | None:
    """This package's own source tree, when it is installed editable.

    A non-editable dist lives in site-packages and rides the venv bind; an
    EDITABLE one leaves only a .pth behind, and its actual code sits in a
    checkout the venv bind does not cover -- so `python -m aisan.launch` in the
    box dies with `No module named aisan`, which names the module and says
    nothing about the mount that is missing.

    Just this package, found from the module rather than by scanning every
    editable distribution in the venv. Nothing above this package is imported
    to reach the launcher, so no unrelated source tree needs to be mounted into
    the box.
    """
    root = Path(__file__).resolve().parent.parent
    # The parent of the package dir is a source root only for an editable
    # install (<checkout>/src/aisan -> <checkout>/src). Installed normally it is
    # site-packages, which the venv bind already covers, and binding it again
    # would be a second mount of the same tree under a different name.
    return root if (root / "aisan").is_dir() and not _in_site_packages(root) else None


def _in_site_packages(path: Path) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in path.parts)


def launcher_binds(python: Path | None = None) -> list[BindSpec]:
    """Read-only binds every box with egress needs: aisan's own runtime.

    A spec requirement, not a consumer's convenience. The launcher runs INSIDE
    the box, so aisan's interpreter, the venv, and the trees aisan's own imports
    resolve through have to be visible there -- and a box whose payload is a
    shell script, or a compiled binary, or another language's toolchain has no
    other reason to contain a Python. Stating it here is what keeps "aisan
    works" from silently depending on the consumer having bound its own
    interpreter for its own reasons -- a dependency that holds right up until the
    consumer stops needing a Python in the box, and then breaks aisan.

    Three parts, none redundant: the venv dir unresolved (bin/python is a
    symlink out of it), the interpreter chain it points at, and -- for an
    editable install only -- this package's own source root, which coincides
    with the venv only when it is installed normally.
    """
    exe = python or Path(sys.executable)
    venv = exe.parent.parent
    binds = [Bind(venv, RO), *(Bind(r, RO) for r in interpreter_roots(exe))]
    own = own_source_root()
    if own is not None and not own.is_relative_to(venv):
        binds.append(Bind(own, RO))
    return binds


def launch_prefix(runtime_dir: Path, python: Path | None = None) -> list[str]:
    """The argv prefix that supplies a payload's egress transport.

    `python -m` at an absolute interpreter path rather than the installed
    `aisan-box-launch` console script: it names exactly the interpreter
    `launcher_binds` bound, with no PATH lookup and no shim to resolve. In a box
    the PATH deliberately holds several venvs' bin dirs, and picking the
    launcher by name there would make which aisan runs a function of bind order.
    The console script still exists, for an operator inside `box shell`.
    """
    exe = python or Path(sys.executable)
    return [str(exe), "-m", f"{__package__}.launch", str(runtime_dir), "--"]


def exit_status(code: int) -> int:
    """A `wait` status as the number a shell would report.

    A process killed by signal N has no exit code; `wait` and `subprocess.run`
    hand it back as `-N`, while a shell reports `128 + N`. Normalising here means
    a caller reading the status sees the same number with a launcher in the
    middle as it would without one.
    """
    return code if code >= 0 else 128 - code


async def _run(runtime_dir: Path, cmd: list[str]) -> int:
    """Inject shared state and exec, or serve isolated relays around `cmd`."""
    if (runtime_dir / CLIENT_ENV_NAME).exists():
        env = {**os.environ, **read_client_env(runtime_dir)}
        os.execvpe(cmd[0], cmd, env)  # noqa: S606 - argv exec, no shell

    servers = []
    for entry in read_manifest(runtime_dir):
        # Bind failures propagate: a relay that is not listening does not mean a
        # degraded box, it means every call through that backend fails with a
        # connection error, and startup is a far better place to learn that than
        # the middle of a turn.
        servers.append(
            await relay.serve(Path(str(entry["socket"])), int(str(entry["port"])))
        )
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        loop = asyncio.get_running_loop()
        for sig in _FORWARD:
            # Forward rather than die: the payload owns the job, and killing the
            # launcher first would tear the relays out from under a process that
            # is still trying to finish. It gets the signal, we wait for it, and
            # the relays close after it is gone.
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
    # Raced with the payload exiting. Nothing to signal, nothing to fix.
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
