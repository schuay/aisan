# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""What a box is: an ordered bind list, an environment, limits, and egress.

`BoxSpec` is pure data and deliberately **non-defaulting**. Every field must be
written at the call site, so a reviewer reading one knows what is mounted
without simulating default resolution -- which is the property that makes a
profile auditable, and the property `Box(...)` would lose the moment it grew a
convenience default. Convenience lives one layer up, in `presets`: pure
functions returning this same object, so a preset can never express something
you could not write by hand.

The spec is the whole policy but not the whole box. What it deliberately does
NOT carry is anything derived from the box's identity -- the runtime directory,
the socket paths, the relay manifest, the launcher prefix. Those are the Box's,
because they exist only once a box is running and two specs that differ only in
them are the same policy. `box.py` is where the two meet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .egress.base import Backend
from .sandbox import BindSpec

_BACKEND_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

# Noninteractive defaults for any box that runs build tooling. Not a bwrap flag
# and not a mount, so nothing here enforces it: a spec's `env` is the box's WHOLE
# environment (--clearenv, then --setenv per entry), which means a default not
# named in a spec is a default that box does not have. Merged in by whoever
# builds the spec -- see presets/v8_job.py -- rather than added on the way to the
# box, because a profile that states its environment and then silently receives
# more of it is a profile nobody can read.
#
# The pager/prompt entries stop a tool blocking on a terminal that is not there.
# AI_AGENT is read by autoninja/siso, which drop per-action progress when it is
# non-empty -- any value will do, so this one is free to say who set it, and it
# says aisan because that is what built the box.
DEFANG_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "AI_AGENT": "aisan",
}


@dataclass(frozen=True)
class Limits:
    """Resource caps applied through the systemd transient scope.

    Separated from the mount policy because they answer a different question --
    "how much of this machine may the box use" rather than "what may it see" --
    and because a caller tuning one almost never touches the other. Every field
    has a neutral value (empty string / 0) meaning "leave systemd's default",
    which is why this is the one part of a spec that may be defaulted: an unset
    limit is not an unstated mount, it is a cap that does not exist.

    `use_cgroup=False` drops the scope entirely, for a host with no user manager
    (a container, a bare chroot) or an operator inspecting a profile without
    tripping a cgroup policy.
    """

    memory_max: str = ""
    cpu_quota: str = ""
    tasks_max: int = 0
    slice_unit: str = ""
    use_cgroup: bool = True


@dataclass(frozen=True)
class BoxSpec:
    """One box's complete policy: what it may touch, and how it reaches out.

    Frozen and non-defaulting on everything that describes the box's contents.
    `with_binds` / `with_egress` return a NEW spec rather than mutating, so a
    preset's result can be adjusted without the preset's own guarantees being
    quietly editable -- and so "the spec that built this box" stays a single
    value somebody can print.
    """

    # The rw root, bound at its real absolute path and also the box's cwd.
    root: Path
    # Everything else the box may touch, in mount order: later wins. The rule is
    # bwrap's own, and it is the only precedence rule in the model.
    binds: tuple[BindSpec, ...]
    # (mount point, size in bytes), mounted before the root and the binds.
    tmpfs: tuple[tuple[str, int], ...]
    # The COMPLETE environment inside the box. Nothing is merged on the way to
    # the box, so what a spec says the environment is, is what it is -- with one
    # exception the Box owns and documents: the per-backend client variables,
    # which cannot be written here because they name a port table the Box
    # resolves. See Box.env.
    env: tuple[tuple[str, str], ...]
    # The egress backends this box gets. Each is a host half holding a
    # credential and a client endpoint selected by the network mode; an empty
    # tuple is a box with no credential-aware route.
    egress: tuple[Backend, ...]
    # Its own network namespace: no general route off the machine, and a
    # loopback that is not the host's. False shares the host's complete network.
    unshare_net: bool
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        # Egress without unshare_net is valid only for a backend that supplies
        # the authenticated, kernel-assigned host-loopback transport. A backend
        # with only the fixed relay port would either collide with another box or
        # expose an unauthenticated credential capability on host loopback. The
        # all-or-none rule lives here so no caller can assemble that unsafe mix.
        unsupported = [b.name for b in self.egress if not b.supports_shared_net]
        if self.egress and not self.unshare_net and unsupported:
            raise ValueError(
                "egress without unshare_net requires shared-network support"
                f" from every backend (unsupported: {unsupported})"
            )
        names = [b.name for b in self.egress]
        if any(
            not isinstance(name, str) or _BACKEND_NAME_RE.fullmatch(name) is None
            for name in names
        ):
            raise ValueError(
                f"egress backend names must be safe unique socket-file components"
                f" (names: {names!r})"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"egress backend names collide: {names}")
        ports = [b.port for b in self.egress]
        if self.unshare_net and len(set(ports)) != len(ports):
            # A collision presents as one backend mysteriously unreachable (its
            # relay bound first and the other's client found the wrong server),
            # which is a long way from the typo that caused it.
            raise ValueError(
                f"egress ports collide: {[(b.name, b.port) for b in self.egress]}"
            )

    def with_binds(self, extra: list[BindSpec]) -> BoxSpec:
        """This spec with `extra` appended to the bind list.

        Concatenation, with no precedence logic to invent: later-wins is already
        the semantics, so appending is exactly "these binds are applied after
        everything the preset asked for". Prepending is deliberately not offered
        -- a bind that has to come first is a bind whose position matters, and
        that belongs in a spec written by hand rather than in a combinator.
        """
        return replace(self, binds=(*self.binds, *extra))

    def with_egress(self, extra: list[Backend]) -> BoxSpec:
        """This spec with `extra` backends added. Order is irrelevant here --
        backends are addressed by port, not by position -- but the port
        injectivity check in __post_init__ still runs, so adding a backend that
        collides with one already present fails at composition."""
        return replace(self, egress=(*self.egress, *extra))
