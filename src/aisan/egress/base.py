# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""What a backend is, and what a box may ask of one.

A backend is one route off the machine, in two halves that never meet: a HOST
half holding the credential, checking the allowlist and dialing upstream, and an
IN-BOX half that is a loopback listener splicing to a UNIX socket. The socket is
what crosses the network namespace, being a filesystem object; the port is what
a client that can only dial host:port can use.

The split this contract encodes is transport vs backend. A transport is
`(unix socket) -> (allowlist) -> (inject credential) -> (upstream)`; a backend is
the allowlist, the upstream host, the credential source, and any files the box
needs in order to dial it. Adding a second model provider is a backend; adding a
protocol nobody speaks yet is a transport, and only that costs real code.

Three things every backend must be able to answer, and one it may:

- `port` -- the fixed in-box loopback port it answers on. Fixed is safe because
  the box's loopback is private to it; see `spec.BoxSpec.__post_init__` for why
  it is also required to be safe.
- `preflight()` -- can the credential be obtained AT ALL, right now. Asked
  before bwrap starts, because once the box is up it owns the TTY and an
  interactive reauth has nowhere to prompt. The failure carries the exact
  command to fix it.
- `serve(runtime_dir)` -- the host half, for the life of the box.
- `refused` -- the mint failure this backend hit, or None. Optional in the sense
  that the default is honest for a backend that cannot fail this way; it is on
  the PUBLIC surface because "did this build fail for lack of a credential" has
  a caller (the offline build fallback) whose only alternative is matching
  someone else's stderr for phrases like "code = Unauthenticated".
"""

from __future__ import annotations

import abc
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ..sandbox import BindSpec


class PreflightError(RuntimeError):
    """A backend cannot obtain its credential, and what to run about it.

    A distinct type rather than a message convention because the fix command is
    the payload: a caller renders it to an operator, and losing it turns "run
    this" into "something went wrong with authentication", which is the failure
    an operator has to guess their way out of.
    """

    def __init__(self, backend: str, reason: str, fix: str) -> None:
        super().__init__(f"{backend}: {reason}\n  fix: {fix}")
        self.backend = backend
        self.reason = reason
        self.fix = fix


# The key value every backend hands its box. Distinct from a real key by
# construction, and self-describing: it shows up in the box's environment, in
# `explain` output and in a snapshot, so it should say what it is to whoever
# reads it there. Its only job is to stop the client looking for a credential
# the box does not have -- the value is deliberately meaningless.
#
# One constant for every backend rather than one each: it names aisan's side of
# the same contract wherever a client reads it, and a backend that minted a
# DIFFERENT-looking placeholder would be a backend whose placeholder could be
# mistaken for a policy of its own.
PLACEHOLDER_KEY = "aisan-placeholder-not-a-credential"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class Backend(abc.ABC):
    """One egress route: the host half, plus what the box needs to dial it."""

    #: Stable identifier, used for the socket file name and in the manifest.
    name: str
    #: The in-box loopback port. Fixed per backend; see the module docstring.
    port: int
    #: Filesystem locations holding this backend's credential. The Box refuses
    #: a spec bind whose source contains one of them, because the
    #: credential-absence claim is subtraction -- a bind naming the credential
    #: (or any ancestor of it) undoes that from inside the spec. Empty, the
    #: default, for a backend whose credential never touches disk: one minted
    #: in memory, or none at all.
    #:
    #: Note the file is the HOST's and may hold more than this route needs --
    #: opencode's `auth.json` carries every provider's key -- so exposing it
    #: is exposure even for a provider the box does not use. That is why the
    #: check is over paths, not over "the key this backend attaches".
    credentials: tuple[Path, ...] = ()

    def socket_path(self, runtime_dir: Path) -> Path:
        """Where this backend listens, inside the box's runtime dir.

        Derived from the name so the host half and the manifest cannot name two
        different files -- but note that the box never derives it: the launcher
        reads the path out of `relays.json`. This is the host side's own
        convenience, not a shared derivation.
        """
        if not isinstance(self.name, str) or _NAME_RE.fullmatch(self.name) is None:
            raise ValueError(
                f"backend name {self.name!r} is not a safe socket-file component"
            )
        return runtime_dir / f"{self.name}.sock"

    def client_env(self) -> dict[str, str]:
        """Environment the box's CLIENT reads to find this backend.

        The client, never the relay: the relay's wiring comes from the manifest.
        This is the one place a backend's in-box port becomes a string, and it
        is deliberately the application's vocabulary (`SISO_REAPI_ADDRESS`), not
        the sandbox's -- aisan's contract is "your backend answers on loopback
        port N" and mapping N to a variable name is what the consumer knows.
        """
        return {}

    def box_binds(self, runtime_dir: Path) -> list[BindSpec]:
        """Extra mounts this backend needs in the box, e.g. a bind-over of a
        config file the box would otherwise read from the host copy. Empty for a
        backend whose client needs nothing but an endpoint."""
        return []

    def prepare(self, runtime_dir: Path) -> None:
        """Write whatever `box_binds` names as a source, before bwrap resolves.

        Separate from `serve` because a bind-over source that does not exist
        fails the whole profile: it must be on disk before `wrapper()` is
        called, and `serve` is where the socket appears, which is later.
        """

    async def preflight(self) -> None:
        """Raise PreflightError if the credential cannot be obtained now.

        Default: nothing to check. A backend that mints MUST override this --
        the alternative is discovering the problem when the box already owns the
        terminal, which is where an interactive reauth has nowhere to go.
        """

    @property
    def refused(self) -> Exception | None:
        """The mint failure this backend hit during the box's life, or None."""
        return None

    @abc.abstractmethod
    @asynccontextmanager
    async def serve(self, runtime_dir: Path) -> AsyncIterator[None]:
        """Run the host half on `socket_path(runtime_dir)` for the block.

        Raises on a failed start rather than degrading. A backend that is not
        listening does not mean a slower box, it means every call through it
        fails -- and a box that fails here fails with a legible reason instead
        of an agent silently unable to think, or a build silently dialing a
        dead port.
        """
        raise NotImplementedError
        yield  # pragma: no cover -- makes this an async generator for typing


def credential_exposure(
    binds: tuple[BindSpec, ...], backends: tuple[Backend, ...]
) -> tuple[Path, Backend, Path] | None:
    """The first (bind source, backend, credential) where a bind's source
    contains a backend's credential, or None.

    Containment, not equality: binding the credential's PARENT exposes it just
    as surely as binding the file, and equality is the containment edge case.
    Resolved on both sides, so a symlinked home compares against its real
    target -- the same rule `_strict_ancestor` applies in the sandbox, for the
    same reason.

    A predicate rather than a check that raises, so each caller words its own
    refusal in its own terms: the Box names the spec and the box id, the user
    bind loader names the file and the key the path came from. What must NOT
    be duplicated is the decision, and a predicate makes the decision the one
    thing both callers share.

    `Bind.path`, `BindOver.src` and `Overlay.path` are the sources; a `Seal`
    has none (it mounts nothing FROM the host, which is its point).
    """
    for spec in binds:
        src = getattr(spec, "path", None) or getattr(spec, "src", None)
        if src is None:
            continue
        try:
            src_r = src.resolve()
        except OSError:
            src_r = src
        for backend in backends:
            for cred in backend.credentials:
                try:
                    cred_r = cred.resolve()
                except OSError:
                    cred_r = cred
                if cred_r.is_relative_to(src_r):
                    return src, backend, cred
    return None
