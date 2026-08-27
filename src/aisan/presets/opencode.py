# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The opencode profile: an agent CLI in a box, holding no credential.

The payload is straightforward to mount because opencode ships as a single ELF
under `/usr/bin` (measured, 1.18.18: 183 MB, Bun-compiled, no node tree, no
package root), and `/usr` is bound ro unconditionally -- so unlike the Claude
Code profile there is no interpreter question to answer and no `extra_ro` a
nonstandard install forces on the caller.

**The credential is not in the box, by the same subtraction as Claude Code.**
The host's opencode keeps its keys in `~/.local/share/opencode/auth.json`;
nothing binds it or any ancestor other than the HOME tmpfs, so in the box it
does not exist -- and the client never reaches for it, because
`OPENCODE_AUTH_CONTENT` (set by the backend) is a COMPLETE in-memory auth
store: measured, when it is set opencode does not read the file at all. The
real key is attached host-side, past the namespace the box cannot cross.
`test_the_credential_is_not_in_the_box` asserts it from inside a real box.

**The XDG seam, and the one redirect that is deliberately absent.** opencode
writes under three XDG roots, and only one needs to survive the box:
`XDG_DATA_HOME` (the session db, logs), which is pointed at the rw state dir
so sessions persist across runs. `XDG_STATE_HOME` (instance locks) and
`XDG_CACHE_HOME` stay at their $HOME defaults -- the tmpfs -- because
ephemeral is the CORRECT scope for a per-box lock and for a cache whose
freshness is a host decision. `XDG_CONFIG_HOME` is not redirected either, and
that one is a trap rather than a preference: git reads
`$XDG_CONFIG_HOME/git/config` INSTEAD of `~/.config/git/config` when it is
set, so redirecting it would silently bypass the ro bind an interactive
launcher puts at `~/.config/git` (documented git behavior, config.txt). Left
unset, the config root lands on the tmpfs -- ephemeral, which also means the
agent cannot plant config in it that outlives the box.

**The catalog bind, and why it is default-on.** The binary embeds a model
catalog that LAGS the fetched one -- measured: `glm-5.3` was refused as
unknown while the host's fetched catalog listed it, `OPENCODE_DISABLE_MODELS_FETCH`
set in both cases -- so a model pin that works on the host can fail in a box
freshly upgraded. The host's catalog is therefore bound ro at its real path
(the one opencode reads when `XDG_CACHE_HOME` is unset; measured), optional so
a host that has never run opencode still builds the profile -- it falls back
to the embedded list, which is strictly worse but never broken.

The inline config the backend injects (`OPENCODE_CONFIG_CONTENT`) outranks the
PROJECT config in opencode's precedence, which is the property that keeps the
route non-negotiable from inside the repo: the agent can edit the worktree's
`opencode.json`, and that file cannot re-point the provider's base URL.
Containment never rested on it -- `unshare_net` is the boundary -- but a route
that holds without cooperation from inside the box is one less thing to
reason about.

The TTY question is inherited open from the Claude Code profile and stays
open here: `--new-session` (TIOCSTI hardening) is unset, and whether an
interactive TUI survives the detached controlling terminal is unvalidated.
Every measurement behind this file used `opencode run` (headless).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..egress.base import Backend
from ..gitbinds import GC_ENV, git_binds
from ..sandbox import RO, RW, Bind, BindSpec
from ..spec import DEFANG_ENV, BoxSpec, Limits

# Legibility knobs, not security controls. The network namespace already
# makes each of these things unable to reach its destination; what the flags
# buy is the failure MODE -- autoupdate and the models/LSP fetchers retry and
# hang on a network that is not coming back, which in a headless box looks
# exactly like a box that is working. Same reasoning as Claude Code's
# CLAUDE_CODE_MAX_RETRIES=0, stated in the same place.
_DISABLE_ENV = (
    ("OPENCODE_DISABLE_AUTOUPDATE", "1"),
    ("OPENCODE_DISABLE_MODELS_FETCH", "1"),
    ("OPENCODE_DISABLE_LSP_DOWNLOAD", "1"),
)


# Where the HOST's opencode keeps the fetched provider catalog. XDG-aware for
# the same reason the backend's paths are: this is wherever the host's own
# opencode actually put it.
def _default_catalog() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "opencode" / "models.json"


def opencode_binary() -> Path | None:
    """The `opencode` entry point on this host, or None.

    Returned rather than raised on, exactly as `claude_code_binary`: a preset
    is a pure function and a host without the binary can still describe the
    profile. An absent binary is an exec failure inside bwrap, which reads as
    a broken box rather than a missing package -- a caller that intends to RUN
    should check.
    """
    found = shutil.which("opencode")
    return Path(found) if found else None


def opencode(
    worktree: Path,
    *,
    state: Path,
    egress: tuple[Backend, ...] = (),
    extra_ro: tuple[Path, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    models: Path | None = None,
    unshare_net: bool = True,
    cgroup_slice: str = "",
    tmp_size_mb: int = 2048,
    home_size_mb: int = 1024,
    memory_max: str = "8G",
    cpu_quota: str = "",
    tasks_max: int = 4096,
) -> BoxSpec:
    """The spec for an opencode session on `worktree`.

    `state` is required and has no default, for the same reason as Claude
    Code's: it is the one directory the agent's own session history lives in,
    so it belongs to the job. `models` is the provider catalog to bind ro --
    None means the host's fetched copy (an optional bind, so a host without
    one still builds); the embedded catalog opencode falls back to is a
    snapshot that lags, see the module docstring.
    """
    home = Path.home()
    catalog = models if models is not None else _default_catalog()
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
        # shadow it. Same rule as the Claude Code preset's.
        *([] if state.is_relative_to(worktree) else [Bind(state, RW)]),
        # The catalog, pinned ro on top of the HOME tmpfs at the path opencode
        # reads by default. Last: it is a host fact, not a policy, and it
        # shadows nothing the caller asked for.
        Bind(catalog, RO, optional=True),
    ]
    return BoxSpec(
        root=worktree,
        binds=tuple(binds),
        # A writable scratch at the real HOME path, for tools that insist on
        # writing under $HOME. NOT what hides the credential -- that is the
        # absence of any bind naming it. The state and catalog binds land on
        # top. "/tmp" is the mount POINT of the box's private tmpfs, not a
        # host path written to here.
        tmpfs=(("/tmp", tmp_size_mb << 20), (str(home), home_size_mb << 20)),  # noqa: S108
        env=(
            *DEFANG_ENV.items(),
            *GC_ENV,
            ("HOME", str(home)),
            ("PATH", "/usr/bin"),
            # The one XDG redirect that persists: sessions, logs, the db.
            ("XDG_DATA_HOME", str(state)),
            *_DISABLE_ENV,
            *extra_env,
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


def opencode_default(worktree: Path) -> BoxSpec:
    """The preset as `explain --dry-run` invokes it: a worktree and nothing
    else.

    Egress-less, like `claude_code_default`, and for the same reason -- a
    registry entry has to be describable on a host with no deployment. The
    state dir defaults to one INSIDE the worktree here, which a real caller
    should not do: it puts the agent's session history in the tree the agent
    is editing. A dry run has no job to own it.
    """
    return opencode(worktree, state=worktree / ".aisan-opencode-state")
