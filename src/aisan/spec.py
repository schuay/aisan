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

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .egress.base import _NAME_RE as _BACKEND_NAME_RE
from .egress.base import Backend
from .sandbox import BindSpec, EnsurePath

# Noninteractive defaults for any box that runs build tooling. Not a bwrap flag
# and not a mount, so nothing here enforces it: a spec's `env` is the box's WHOLE
# environment (--clearenv, then --setenv per entry), which means a default not
# named in a spec is a default that box does not have. Merged in by whoever
# builds the spec -- see presets/depot_tools_job.py -- rather than added on the
# way to the
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
class Grant:
    """What one `--grant NAME` contributes to a box: the mounts, PATH entries
    and environment some tree needs to work inside a box that has no network.

    Deliberately not named for tools: a tool tree is the first case, but the
    shape fits anything aisan can know how to hand a box -- a CA bundle and the
    variable naming it, a device node and the library path that finds it. What
    every one of them has in common is that it WIDENS the box, which is why the
    flag says grant.

    The sibling of `EgressProfile`, and the split between them is what each is a
    fact ABOUT. An egress profile is a fact about a checkout -- an RBE project,
    where a tree keeps its config -- so it is a function of the box's root. A
    grant is a fact about the HOST: where depot_tools is installed, where
    vpython keeps its venvs. It takes no root, and a box rooted anywhere gets
    the same one.

    `env` is the half no bind can express, and the reason this type exists
    rather than a list of mounts. A tool tree bound read-only into a box with no
    route does not merely lack the network -- it tries to USE it, on every
    invocation, and fails in its own vocabulary rather than in the box's. The
    value that turns that off is knowledge about the tree, so it belongs beside
    the mounts that need it and not in each operator's config file.
    """

    binds: tuple[BindSpec, ...] = ()
    path: tuple[Path, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # PATH through `env` would be applied AFTER `path` and win, silently
        # replacing the box's PATH with this grant's idea of it rather than
        # prepending to it. The field exists so that cannot be written by hand.
        if any(k == "PATH" for k, _ in self.env):
            raise ValueError("a grant names PATH through `path`, not through `env`")

    def __bool__(self) -> bool:
        """Whether this grant contributes anything. A tree absent from this host
        resolves to an empty grant rather than a refusal -- the same shape
        `EgressProfile` uses, and the launcher says so rather than building a box
        that silently lacks what was asked for."""
        return bool(self.binds or self.path or self.env)


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
    # Host paths to create (empty) before the binds resolve and remove again if
    # this box created them. The .git guard pins name real host files that may
    # be absent on a fresh checkout; see EnsurePath for why this is declared
    # rather than done in the preset that computes the binds.
    ensure: tuple[EnsurePath, ...] = ()

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

    def with_path_prefix(self, dirs: tuple[Path, ...]) -> BoxSpec:
        """This spec with `dirs` prepended to the box's PATH.

        The mount-completing env combinator, and it exists because PATH is the
        variable whose values are DIRECTORIES: a bind the box cannot resolve by
        name is half a grant, and completing it adds no reach the bind did not.
        Values that are not directories go through `with_env`, which is a wider
        grant and says so.

        Prepended, not appended: a caller adding a tool tree means the box to
        use that one. An absent PATH is set rather than refused, so this
        composes with a spec whose environment does not name one.

        The LAST entry, not the first. `env` is an ordered tuple of pairs
        rather than a mapping, and a preset naming a variable that its caller
        then overrides through `extra_env` leaves two -- which bwrap resolves
        the way this does, by `--setenv` running in order. Rewriting the first
        would prepend to the value the box does not use.
        """
        if not dirs:
            return self
        # The invariant this claims -- a PATH prefix names box directories -- was
        # enforced nowhere: a relative dir resolves against the box's cwd, and a
        # dir containing os.pathsep smuggles a second PATH entry past every
        # coverage check. Both are refused here so no caller (userbinds or a
        # direct one) can assert the invariant while breaking it.
        for d in dirs:
            if os.pathsep in str(d):
                raise ValueError(
                    f"PATH prefix dir {d} contains {os.pathsep!r}, which would"
                    " smuggle a second PATH entry"
                )
            if not Path(d).is_absolute():
                raise ValueError(f"PATH prefix dir {d} is not absolute")
        env = list(self.env)
        last = max((i for i, (k, _) in enumerate(env) if k == "PATH"), default=None)
        value = env[last][1] if last is not None else ""
        entries = [*(str(d) for d in dirs), *(value.split(os.pathsep) if value else [])]
        # First occurrence wins, which is how PATH lookup already works: a
        # repeat of an earlier entry resolves nothing and only makes the
        # rendered profile harder to read. It shows up when a caller names a
        # grant the preset already applied.
        path = os.pathsep.join(dict.fromkeys(entries))
        if last is None:
            env.append(("PATH", path))
        else:
            env[last] = ("PATH", path)
        return replace(self, env=tuple(env))

    def with_env(self, extra: tuple[tuple[str, str], ...]) -> BoxSpec:
        """This spec with `extra` env pairs appended, later winning.

        Wider than `with_path_prefix`: these are VALUES, so nothing here is
        checkable the way a PATH dir is checkable against the mounts. That is
        why it exists for aisan's own named grants -- resolved through a
        registry in this repo, applied by the flag that selected them -- and why
        the user bind format still refuses an env key. A value a person writes
        into an unreviewed file is a secret this box cannot see it carrying.

        Appended rather than merged, because `env` is an ordered tuple that
        bwrap replays through `--setenv` in order: a later pair overriding an
        earlier one is the semantics, and rewriting the earlier would hide from
        a reader that two callers named the same variable.
        """
        if not extra:
            return self
        return replace(self, env=(*self.env, *extra))

    def with_egress(self, extra: list[Backend]) -> BoxSpec:
        """This spec with `extra` backends added. Order is irrelevant here --
        backends are addressed by port, not by position -- but the port
        injectivity check in __post_init__ still runs, so adding a backend that
        collides with one already present fails at composition."""
        return replace(self, egress=(*self.egress, *extra))
