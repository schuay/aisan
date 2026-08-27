# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The V8 job profile: what a sandboxed agent turn on a V8 worktree may touch.

This is the preset the whole app runs on, and the reference for what a preset
is: it names binds, an environment and limits, and returns a `BoxSpec`. It
starts nothing, reads no config file, and holds no state -- so a caller whose
case differs by one mount adjusts the result (`with_binds`) instead of forking
the function.

The profile: the worktree rw at its real absolute path (remote-exec resolves
inputs by absolute path), the dep symlink TARGETS ro (binding the symlinks alone
would either break the build or hand the agent the writable main tree), a
restricted slice of the main checkout's .git so in-worktree git keeps working,
depot_tools ro, and whatever the consumer adds.

The box holds no credential of its own. Both egress routes -- the model and the
remote build -- run through host-side backends that mint from the host's own
durable credential and attach it to requests the box never sees. What the box
gets is an endpoint. When the RBE backend is absent the build degrades to the
offline fallback (autoninja -o), so remote execution is a fast path, not a
requirement.

Network is the box's own (`unshare_net`): no route off the machine, and a
loopback that is not the host's. The backends stay reachable because each listens
on a UNIX socket in the box's runtime dir -- a filesystem object crosses a
namespace, a port does not -- and the launcher relays loopback to it in-box.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, external_symlink_targets, git_binds
from ..sandbox import RO, Bind, BindSpec, Overlay
from ..spec import DEFANG_ENV, BoxSpec, Limits

# `git cl presubmit` fetches the live CL description and validates OWNERS
# against Gerrit, both over authenticated /a/ endpoints. Those need an SSO
# cookie the box deliberately does not have (no ~/.gitcookies, /run unbound), so
# in-box they do not fail on the patch -- they fail on git-remote-sso with
# "SSO login required", which reads as a broken tree and burns revise rounds
# chasing it. depot_tools' own switch skips exactly the network half
# (git_cl.py, presubmit_support.py, presubmit_canned_checks.py all gate on it),
# leaving the local checks -- lint, formatting, copyright -- which are the ones
# an agent can act on anyway.
#
# This narrows what presubmit proves: a green in-box run says nothing about
# OWNERS or the CL description. That is the right trade here, because the CQ
# re-runs the full set and is the authority; the in-box run is a fast local
# sanity check, not a gate.
_PRESUBMIT_ENV = (("PRESUBMIT_SKIP_NETWORK", "1"),)


def _vpython_cache() -> Path:
    """vpython's venv store on this host, the path depot_tools derives itself.

    `~/.cache/vpython-root.<uid>` (depot_tools/utils.py names the same shape).
    Read from the real home rather than the box's blanked one, because what is
    wanted is the HOST's warm copy -- see _tool_cache_overlays for why.
    """
    return Path.home() / ".cache" / f"vpython-root.{os.getuid()}"


def _tool_cache_overlays() -> list[Path]:
    """Warm tool caches the box may write to and cannot change.

    Only vpython's today. Every `git cl` invocation runs through it, and it is
    the one thing standing between the agent and both `git cl format` and `git cl
    presubmit` -- 14 of the ~45 friction entries the fleet reported were this.
    Three bindings are possible and two are wrong:

      ro       -- fails. vpython takes a lock file even to READ ("failed to
                  acquire read lock ... read-only file system"), which is the
                  error the agents kept reporting.
      absent   -- worse. vpython tries to rebuild the venv, and in a box with no
                  network that HANGS rather than failing (measured: no output,
                  killed at 120s).
      rw       -- worst. It is ~4 GB of interpreters and ~21k Python files that
                  every box EXECUTES, shared across jobs and with the host, so a
                  writable bind is a cross-job code-execution channel that also
                  outlives the job.

    The overlay is the fourth: warm to read, writable inside, discarded on exit.
    Skipped when absent so a host that has never run vpython still builds a
    profile -- the box then behaves as it did before this existed.
    """
    cache = _vpython_cache()
    return [cache] if cache.is_dir() else []


def sisoenv_path(worktree: Path) -> Path:
    """Where a V8 checkout keeps the .sisoenv the RBE backend binds over.

    Here rather than in the backend because it is a fact about the CHECKOUT, and
    a consumer boxing something else keeps its own copy of that fact.
    """
    return (worktree / "build" / "config" / "siso" / ".sisoenv").resolve()


def v8_job(
    worktree: Path,
    *,
    egress: tuple[Backend, ...] = (),
    extra_ro: tuple[Path, ...] = (),
    extra_path: tuple[str, ...] = (),
    depot_tools: Path | None = None,
    unshare_net: bool = False,
    cgroup_slice: str = "",
    tmp_size_mb: int = 4096,
    home_size_mb: int = 1024,
    memory_max: str = "32G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """The spec for one job's agent turns on `worktree`.

    `extra_ro` is the consumer's: the control dir, the agent's Python runtime,
    the MCP launchers, whatever else its own deployment needs visible. Optional
    binds throughout -- a dep symlink target or an operator extra_ro that has
    gone away is a bind to skip, not a profile to refuse -- unlike the .git pins,
    which are guards and must exist.
    """
    if depot_tools is None:
        autoninja = shutil.which("autoninja")
        depot_tools = Path(autoninja).parent if autoninja else None
    ro = [*external_symlink_targets(worktree), *extra_ro]
    if depot_tools is not None:
        ro.append(depot_tools)
    home = Path.home()
    # depot_tools stays first (build tooling precedence); extra_path carries the
    # consumer's launcher dirs, so MCP servers named bare in a config resolve.
    parts = [str(depot_tools)] if depot_tools else []
    path_env = ":".join([*parts, *extra_path, "/usr/bin"])
    # The whole mount policy, in the order bwrap applies it and later winning.
    # Read top to bottom: warm caches, then everything the box may read, then
    # what it may write (the rw root goes down inside resolve(), between the
    # two), then the .git pins on top of the .git it may write. The Box appends
    # its own three -- the runtime dir, aisan's interpreter, the backends'
    # bind-overs -- after all of these.
    binds: list[BindSpec] = [
        *(Overlay(p) for p in _tool_cache_overlays()),
        *(Bind(p, RO, optional=True) for p in ro),
        *git_binds(worktree, pin_packs=unshare_net),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        # HOME is blanked at its real path: the worktree and control-dir binds
        # land on top, tools that insist on writing under $HOME (vpython caches)
        # get a size-capped scratch, and nothing else of the real home exists.
        # "/tmp" is the mount POINT of the box's own tmpfs, not a path this
        # code writes to on the host -- the box gets a private /tmp, which is
        # the opposite of the insecure-shared-tmp the rule looks for.
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            # The box's environment is exactly this tuple (--clearenv, then
            # --setenv per entry) plus each backend's client vars, so the
            # noninteractive defaults have to be named here or the box does not
            # have them. AI_AGENT is the one that shows: siso keys off it to drop
            # per-action build spam.
            *DEFANG_ENV.items(),
            ("HOME", str(home)),
            ("PATH", path_env),
            *GC_ENV,
            *_PRESUBMIT_ENV,
        ),
        egress=egress,
        unshare_net=unshare_net,
        limits=Limits(
            memory_max=memory_max,
            cpu_quota=cpu_quota,
            tasks_max=tasks_max,
            slice_unit=cgroup_slice,
        ),
    )


def v8_job_default(worktree: Path) -> BoxSpec:
    """The preset as `explain --dry-run` invokes it: a worktree and nothing else.

    Deliberately without egress. A registry entry has to be describable on a
    host with no deployment, and a backend constructed from a project id nobody
    supplied would describe a box nobody runs. What this shows is the mount
    policy, which is what a reviewer reading a bind-list diff came for; the
    egress half is two extra bind-overs and a socket bind, and `probe` is where
    those get exercised for real.
    """
    return v8_job(worktree)
