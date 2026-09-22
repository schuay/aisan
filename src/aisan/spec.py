# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The complete policy for a box: binds, environment, limits, and egress.

`BoxSpec` has no implicit policy defaults. Every field is visible at the call
site, so a reviewer can tell what the box mounts without reproducing default
resolution. Presets provide convenience by constructing the same data type and
cannot express anything that a caller could not write directly.

Runtime state derived from a box's identity does not belong in the policy. The
runtime directory, socket paths, relay manifest, and launcher prefix exist only
while a `Box` is running. `box.py` combines that state with the spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .egress.base import _NAME_RE as _BACKEND_NAME_RE
from .egress.base import Backend
from .private import nested_root
from .sandbox import BindSpec, EnsurePath

# Defaults for noninteractive build tools. A spec's `env` is the complete box
# environment (`--clearenv` followed by its `--setenv` entries), so the code
# constructing the spec must add these explicitly. Applying them later would
# make the rendered profile incomplete. See presets/depot_tools_job.py.
#
# The pager and prompt settings prevent tools from waiting for an unavailable
# terminal. autoninja and siso suppress per-action progress when AI_AGENT is
# nonempty; its value identifies aisan as the source.
DEFANG_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "AI_AGENT": "aisan",
}

# An inner aisan needs a private root outside the sealed host default. Keep this
# separate from DEFANG_ENV because it serves nested boxes.
# The spec applies it explicitly so `explain` shows the effective value.
NESTING_ENV = {"AISAN_PRIVATE_ROOT": str(nested_root())}


@dataclass(frozen=True)
class Limits:
    """Resource caps applied through the systemd transient scope.

    Limits are separate from the mount policy because they control resource use
    instead of visibility. Empty strings and zero retain systemd's defaults.
    These neutral values make `Limits` safe to default even though mount policy
    must always be explicit.

    Set `use_cgroup=False` on hosts without a user manager, such as containers
    and bare chroots, or when inspecting a profile without creating a scope.
    """

    memory_max: str = ""
    cpu_quota: str = ""
    tasks_max: int = 0
    slice_unit: str = ""
    use_cgroup: bool = True


@dataclass(frozen=True)
class Grant:
    """Mounts, PATH entries, and environment added by one `--grant NAME`.

    Grants cover any host resource aisan knows how to supply, including tool
    trees, CA bundles with their environment variables, or device nodes with
    library paths. Each grant widens the box's access.

    An `EgressProfile` depends on the checkout, such as the RBE project named in
    its configuration. A grant describes the host, such as the locations of
    depot_tools and vpython environments, and therefore does not depend on the
    box root.

    `env` carries requirements that mounts cannot express. For example, an
    offline tool may need an environment variable that disables its updater.
    Keep these settings with the tool's mounts so operators don't duplicate them.
    """

    binds: tuple[BindSpec, ...] = ()
    path: tuple[Path, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # A PATH in `env` would override the entries from `path` instead of
        # prepending them.
        if any(k == "PATH" for k, _ in self.env):
            raise ValueError("a grant names PATH through `path`, not through `env`")

    def __bool__(self) -> bool:
        """Return whether this grant contributes anything.

        A resource absent from the host resolves to an empty grant. The launcher
        reports that result instead of building a box missing the requested
        resource.
        """
        return bool(self.binds or self.path or self.env)


@dataclass(frozen=True)
class BoxSpec:
    """One box's complete policy: what it may touch, and how it reaches out.

    Every field that controls the box's contents is explicit and immutable.
    The `with_*` methods return new specs, which preserves the original preset
    and leaves one printable value describing the box.
    """

    # The writable root, bound at its absolute host path and used as the cwd.
    root: Path
    # All other mounts. Order carries no meaning; overlaps resolve by depth,
    # strictness, and guards as described in `sandbox`.
    binds: tuple[BindSpec, ...]
    # (mount point, size in bytes). Binds below a tmpfs land on top of it.
    tmpfs: tuple[tuple[str, int], ...]
    # The complete environment inside the box. `Box.env` adds only per-backend
    # client variables, whose values depend on ports resolved at runtime.
    env: tuple[tuple[str, str], ...]
    # Host-side egress backends and their network-mode-specific client endpoints.
    egress: tuple[Backend, ...]
    # True creates a network namespace with private loopback and no external route.
    unshare_net: bool
    limits: Limits = field(default_factory=Limits)
    # Empty host paths to create before bind resolution and remove afterward if
    # this box created them. See `EnsurePath` for the .git guard use case.
    ensure: tuple[EnsurePath, ...] = ()
    # Host paths this box must not be able to read under any name. Use this for
    # data kept out by subtraction, which has no mount to seal: the box refuses
    # to start if the assembled mount list publishes any of them. A path that is
    # present in the box but emptied belongs in a `Seal`, which carries the same
    # guarantee for its own contents.
    confidential: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        # Shared networking requires authenticated, kernel-assigned loopback
        # endpoints. A fixed relay port could collide with another box or expose
        # an unauthenticated credential capability on the host.
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
            # Without this check, one relay binds first and the other backend's
            # client reaches the wrong server.
            raise ValueError(
                f"egress ports collide: {[(b.name, b.port) for b in self.egress]}"
            )

    def with_binds(self, extra: list[BindSpec]) -> BoxSpec:
        """Return this spec with `extra` added to the bind list.

        Position in the list carries no precedence. An added bind at a deeper
        path wins below it; at an existing path the stricter mode wins; at or
        below a guard it is refused.
        """
        return replace(self, binds=(*self.binds, *extra))

    def with_path_prefix(self, dirs: tuple[Path, ...]) -> BoxSpec:
        """Return this spec with `dirs` prepended to the box's PATH.

        This complements a mount by making its executables discoverable. PATH
        entries add no filesystem access beyond their corresponding binds.
        Other environment values go through `with_env`.

        New entries take precedence over the existing PATH. If the spec has no
        PATH, this method creates one.

        `env` may contain a variable more than once because bwrap applies its
        `--setenv` arguments in order. This method updates the last PATH entry,
        which is the effective one.
        """
        if not dirs:
            return self
        # Relative paths depend on the box cwd, and `os.pathsep` would encode an
        # unchecked second entry. Reject both before mount-coverage checks.
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
        # PATH lookup uses the first occurrence, so later duplicates have no
        # effect and only obscure the rendered profile.
        path = os.pathsep.join(dict.fromkeys(entries))
        if last is None:
            env.append(("PATH", path))
        else:
            env[last] = ("PATH", path)
        return replace(self, env=tuple(env))

    def with_env(self, extra: tuple[tuple[str, str], ...]) -> BoxSpec:
        """Return this spec with `extra` environment pairs appended.

        Arbitrary values cannot be checked against the mount policy as PATH
        entries can. This method therefore serves aisan's reviewed, named grants;
        user bind files cannot add environment variables that may contain
        secrets.

        Appending preserves bwrap's ordered `--setenv` behavior and shows when
        two callers set the same variable. The later value wins.
        """
        if not extra:
            return self
        return replace(self, env=(*self.env, *extra))

    def with_egress(self, extra: list[Backend]) -> BoxSpec:
        """Return this spec with `extra` backends added.

        Backends are addressed by port, so their order does not matter.
        `__post_init__` rejects any port collision introduced by composition.
        """
        return replace(self, egress=(*self.egress, *extra))
