# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The Claude Code profile: an agent CLI in a box, holding no credential.

The payload is an external CLI rather than Python code in this package. Its
mount and environment requirements are recorded here because they are facts
about Claude Code rather than the confinement mechanism.

The profile: the worktree rw at its real absolute path, a rw STATE dir the CLI
owns, an optional ro CONFIG dir, `unshare_net` and the Anthropic backend. There
is no bind for node, for the CLI, or for its package tree -- see below.

**The credential is not in the box, and that is the whole safety claim.**
`~/.claude/.credentials.json` is never bound, and the mechanism is subtraction:
bwrap composes a box out of what it is GIVEN, so a path nobody names is a path
that does not exist there. It is not the HOME tmpfs that hides it -- measured, by
building this box with that tmpfs removed, where the credential is still absent
and HOME holds only what other binds happened to create. The tmpfs is there to
give tools a writable scratch at the real HOME path, not to conceal anything, and
reading it as the safety mechanism would put the claim on the wrong line.

What the box gets instead is a base URL pointing at the in-box relay and a
per-box relay token, in whichever variable matches the credential the proxy will
attach (`CLAUDE_CODE_OAUTH_TOKEN` for the host's subscription login,
`ANTHROPIC_API_KEY` for a static key -- see `egress.anthropic`). The real
credential is attached host-side, on the far side of a namespace the box cannot
cross.
`test_the_credential_is_not_in_the_box` asserts it from inside a real box,
because a preset that grew the bind later would otherwise pass every other test
in the suite.

No node bind, no CLI bind. Measured, not assumed: `/usr` is bound ro
unconditionally (`_SYSTEM_ARGS`), and on a normal install `claude`, `node` and
`@anthropic-ai/claude-code` all live under it -- so a box with no extra binds
already resolves and runs the CLI. An install somewhere else (an npm prefix
under $HOME, a nodenv shim) is NOT covered by that, and would need its package
root and interpreter in `extra_ro`; `claude_code_binary` finds the path so a
caller can check rather than discover it as an exec failure.

The config/state split. `~/.claude` is one directory doing two jobs on the host,
and the two want opposite bindings -- so the preset separates them by mechanism
rather than by hoping:

  STATE (rw)  -- `CLAUDE_CONFIG_DIR`. Measured: the CLI writes `.claude.json`,
                 `projects/`, `sessions/`, `telemetry/` and `.last-cleanup` here
                 within one `-p` turn, so this half cannot be ro. It is also
                 where a `CLAUDE.md` is read from (a marker instruction placed
                 there was obeyed), which means `CLAUDE_CONFIG_DIR` alone is
                 BOTH halves and cannot be the ro one.
  CONFIG (ro) -- a `settings.json` passed with `--settings`. Measured readable
                 from a read-only bind: a settings file whose `env` block named
                 a dead port made the turn fail to connect, which an unread file
                 could not have done. This is the half an operator controls and
                 the agent must not edit.

The seam is therefore the `--settings` FLAG, not an environment variable, and
`claude_code_argv` builds the argv that uses it -- the preset returns a spec, so
the flag belongs to a helper the caller composes with, not to the spec.

`--new-session` is deliberately NOT set, and this preset does not decide the TTY
question. bwrap's `--new-session` protects against TIOCSTI injection back into
the terminal that launched the box, which matters for a payload the operator
sits in front of; it also detaches the controlling terminal, and whether an
interactive Claude Code survives that is unvalidated here. Every measurement
behind this file used `-p` (headless), where it does not arise. So the option
stays unset, an interactive caller should treat it as an open question, and
there is no test here pretending to have settled it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds, git_host_files
from ..sandbox import RO, RW, Bind, BindSpec
from ..spec import DEFANG_ENV, BoxSpec, Limits

# Turn off the CLI's own retry ladder. A box whose host half is missing gets a
# connection reset on the relay port immediately, but the client retries with
# backoff and the failure surfaces minutes later (measured: no answer at 180s,
# versus RC 1 in 1.0s with this set). For a supervised, headless turn a fast
# refusal is worth more than a retry -- the supervisor can retry a whole turn,
# and it can see that it did.
#
# Not a security control. It is set here because an unattended box that hangs
# looks exactly like an unattended box that is working.
_RETRY_ENV = (("CLAUDE_CODE_MAX_RETRIES", "0"),)

# Turn off the CLI's own updater. Under `unshare_net` the npm reach cannot
# succeed, so the startup attempt only buys a delay and a red line in the status
# bar -- measured: "Auto-updating..." then "Auto-update failed - Try claude
# doctor", gone with this set and nothing else changed.
#
# Not a security control either: the box has no route to the registry with or
# without it. It is here so a failure the operator cannot act on stops being
# reported as one.
_UPDATER_ENV = (("DISABLE_AUTOUPDATER", "1"),)


def claude_code_binary() -> Path | None:
    """The `claude` entry point on this host, or None.

    Returned rather than raised on: a preset is a pure function and a host
    without the CLI can still describe the profile. A caller that intends to RUN
    it should check -- an absent binary is an exec failure inside bwrap, which
    reads as a broken box rather than a missing package.
    """
    found = shutil.which("claude")
    return Path(found) if found else None


def claude_code_argv(
    prompt: str, *, settings: Path | None = None, extra: tuple[str, ...] = ()
) -> list[str]:
    """A headless one-shot turn, with the ro settings file if there is one.

    Here rather than in the spec because `--settings` is the read-only half of
    the config split and the split is this preset's decision -- but argv is the
    caller's, so this composes with `Box.command` instead of hiding inside it.
    """
    argv = ["claude"]
    if settings is not None:
        argv += ["--settings", str(settings)]
    return [*argv, *extra, "-p", prompt]


def claude_code(
    worktree: Path,
    *,
    state: Path,
    egress: tuple[Backend, ...] = (),
    config: Path | None = None,
    extra_ro: tuple[Path, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 2048,
    home_size_mb: int = 1024,
    memory_max: str = "32G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """The spec for one Claude Code turn on `worktree`.

    `state` is required and has no default: it is the one directory the agent
    writes its own session history into, so it belongs to the JOB, and a preset
    that picked it would have every box in a fleet sharing one. `config` is
    optional -- a box with no settings file is a box running the CLI's defaults,
    which is a real configuration rather than a broken one.
    """
    home = Path.home()
    binds: list[BindSpec] = [
        *(Bind(p, RO, optional=True) for p in extra_ro),
        # The .git policy for a linked worktree: the common dir rw so git works,
        # its steering files pinned ro, sibling worktrees sealed away, and this
        # session's own dir punched back through. Empty for a plain checkout,
        # whose .git is inside the rw root already.
        #
        # `pin_packs` under unshare_net: the seal removes the siblings' HEAD and
        # index as reachability roots, so an in-box `git gc` would prune objects
        # only they reference out of the SHARED store (measured -- gitbinds
        # documents it). A ro objects/pack stops that; it also stops any in-box
        # fetch, which a box with its own network namespace cannot do anyway.
        *git_binds(worktree, pin_packs=unshare_net),
        # After the ro binds: the state dir is the box's own and nothing may
        # shadow it. `config` stays ro even though it sits under the same HOME
        # tmpfs -- the agent reads its policy, it does not edit it.
        *([Bind(config, RO)] if config is not None else []),
        # No bind when the state dir is already inside the rw root: it is
        # writable there by the root's own mount, and binding it again would
        # require it to EXIST at assembly time -- which a caller that expects
        # the CLI to create it has no reason to arrange. Verified: with no bind
        # of its own, a state dir under the root is created from inside the box
        # and its contents are on the host afterwards.
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        # A writable scratch at the real HOME path, for tools that insist on
        # writing under $HOME. NOT what hides the credential -- that is the
        # absence of any bind naming it, which holds with this tmpfs removed
        # (measured). The worktree and state binds land on top.
        #
        # "/tmp" is the mount POINT of the box's private tmpfs, not a host path
        # written to here.
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            *DEFANG_ENV.items(),
            *GC_ENV,
            ("HOME", str(home)),
            ("PATH", "/usr/bin"),
            # The rw half of the config split. Also where a CLAUDE.md is read
            # from, so it is not "just" state -- see the module docstring.
            ("CLAUDE_CONFIG_DIR", str(state)),
            *_RETRY_ENV,
            *_UPDATER_ENV,
            *extra_env,
        ),
        egress=egress,
        unshare_net=unshare_net,
        ensure=git_host_files(worktree, pin_packs=unshare_net),
        limits=Limits(
            memory_max=memory_max,
            cpu_quota=cpu_quota,
            tasks_max=tasks_max,
            slice_unit=cgroup_slice,
        ),
    )


def claude_code_default(worktree: Path) -> BoxSpec:
    """The preset as `explain --dry-run` invokes it: a worktree and nothing else.

    Egress-less, like `depot_tools_job_default`, and for the same reason -- a registry
    entry has to be describable on a host with no deployment. The state dir
    defaults to one INSIDE the worktree here, which a real caller should not do:
    it puts the agent's session history in the tree the agent is editing. A dry
    run has no job to own it, and inventing a path outside the root would put a
    host directory in the audit artifact.
    """
    return claude_code(worktree, state=worktree / ".aisan-claude-state")
