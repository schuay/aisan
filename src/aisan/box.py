# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""One lifecycle bracket: `async with Box(spec, box_id=...) as box`.

`__aenter__` creates the runtime dir, preflights and starts every backend,
writes the selected transport's launcher control file, and hands back a Box
that can produce the wrapper argv and launch prefix; `__aexit__` unwinds it.

The order inside `__aenter__` is the part worth stating, because it is a real
constraint and not a preference:

1. runtime dir -- every backend writes into it.
2. per backend: preflight, then prepare, then serve. Preflight before ANY bwrap
   because luci reauth is interactive and the box owns the TTY once it is up;
   prepare before wrapper() because a bind-over source that does not exist fails
   the whole profile.
3. relay manifest or protected client environment, because the launcher reads it.
4. only then may `wrapper()` be called. It resolves the binds, and the sources
   have to be on disk by then.

`box_id` is opaque: a string aisan hashes for the runtime dir name and otherwise
never interprets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Self

from .egress.base import Backend, BackendActivation
from .runtime import (
    cleanup_runtime_dir,
    prepare_runtime_dir,
    runtime_bind,
    runtime_dir,
    write_client_env,
    write_manifest,
)
from .sandbox import RO, RW, Bind, Mount, Sandbox
from .spec import BoxSpec


def _rw_grant_covers(path: Path, spec: BoxSpec) -> bool:
    """Whether the spec already grants `path` writable, as the root or through
    a Bind whose mode is RW.

    Containment, not equality: the grant that matters is an ancestor of the
    launcher path, because the launcher's requirement is only visibility and an
    rw ancestor supplies it in full. Optional RW binds do not count -- one that
    cannot be stat'd is dropped at resolve time, and the launcher requirement
    would go with it.

    The consumer's own RO pins inside the root are not this function's business.
    Those are the consumer overriding the consumer (the .git dance in
    Sandbox.resolve's ordering), and later-wins stays theirs to use; this
    predicate only disarms the LIBRARY's defaults, never a bind the spec wrote.
    """

    def within(p: Path, ancestor: Path) -> bool:
        # Compare mount destinations, not their host-side targets. Two paths
        # that resolve to the same directory still name distinct destinations
        # in the box; launcher_binds() deliberately preserves both names for
        # interpreter symlink chains.
        return p.absolute().is_relative_to(ancestor.absolute())

    if within(path, spec.root):
        return True
    return any(
        isinstance(b, Bind) and b.mode is RW and not b.optional and within(path, b.path)
        for b in spec.binds
    )


log = logging.getLogger(__name__)


def _undo_host_paths(files: tuple[Path, ...], dirs: tuple[Path, ...]) -> None:
    """Remove ensure-paths a box created: files first, then directories deepest
    first, and only while empty -- a concurrent box that put something real in
    one keeps it, the same best-effort rule `staged_directory` uses."""
    for f in files:
        with suppress(OSError):
            f.unlink()
    for d in sorted(set(dirs), key=lambda p: len(p.parts), reverse=True):
        with suppress(OSError):
            d.rmdir()


class Box:
    """A running box: the mount policy resolved, the egress halves serving.

    Not a context manager over a Sandbox -- a Box IS the lifecycle. `wrapper()`
    and `launch_prefix()` are only meaningful between `__aenter__` and
    `__aexit__`, because both name paths that exist for exactly that long.
    """

    def __init__(self, spec: BoxSpec, *, box_id: str) -> None:
        self.spec = spec
        self.box_id = box_id
        self.runtime_dir = runtime_dir(box_id)
        self._stack: AsyncExitStack | None = None
        self._activations: dict[Backend, BackendActivation] = {}

    def _needs_relays(self) -> bool:
        return bool(self.spec.egress) and self.spec.unshare_net

    def _stage(self, stack: ExitStack | AsyncExitStack) -> None:
        """Everything that has to exist on disk before `wrapper()` may resolve:
        the runtime dir, each backend's bind-over sources, and the spec's
        ensure-paths (the .git guard files a fresh checkout may lack).

        Shared with `staged()` below, which is the inspector's way in. Split out
        so there is ONE statement of what a resolvable box needs -- an inspector
        that staged a little differently from the real path would report a
        profile nobody ever runs, which is precisely the drift the triad exists
        to catch. Every source this creates is registered for cleanup on the
        same stack, so the run path unwinds it at box exit and the inspector's
        short-lived stack leaves the host exactly as it found it.
        """
        self._ensure_host_paths(stack)
        prepare_runtime_dir(self.box_id)
        # Registered before anything can fail, so a backend that raises halfway
        # still leaves no directory behind.
        stack.callback(cleanup_runtime_dir, self.box_id)
        for backend in self.spec.egress:
            backend.prepare(self.runtime_dir)
            # Whatever prepare() just wrote goes back out here rather than in the
            # backend, because THIS is where it was written and the directory
            # only disappears once it is empty. The sources are not guessed: they
            # are the bind-over srcs the backend itself names, filtered to the
            # ones inside the runtime dir -- a backend binding over something it
            # does not own is not ours to remove.
            for spec in backend.box_binds(self.runtime_dir):
                src = getattr(spec, "src", None)
                if src is not None and src.parent == self.runtime_dir:
                    stack.callback(src.unlink, missing_ok=True)

    def _ensure_host_paths(self, stack: ExitStack | AsyncExitStack) -> None:
        """Create the spec's ensure-paths, empty, registering their undo.

        Files are touched, directories mkdir'd, and each path that did not
        already exist -- including any parent this had to create -- is recorded
        so the single cleanup callback removes exactly what this box added and
        nothing that was already there. Empty is equivalent to absent for every
        file this concerns (an empty git config or hooks dir steers nothing), so
        removing them on the way out cannot change what host-side git sees.
        """
        created_files: list[Path] = []
        created_dirs: list[Path] = []

        def ensure_dir(d: Path) -> None:
            cursor = d
            while not cursor.exists():
                created_dirs.append(cursor)
                if cursor == cursor.parent:
                    break
                cursor = cursor.parent
            d.mkdir(parents=True, exist_ok=True)

        for e in self.spec.ensure:
            if e.is_dir:
                ensure_dir(e.path)
            else:
                ensure_dir(e.path.parent)
                if not e.path.exists():
                    created_files.append(e.path)
                    e.path.touch()
        if created_files or created_dirs:
            stack.callback(_undo_host_paths, tuple(created_files), tuple(created_dirs))

    async def __aenter__(self) -> Self:
        stack = AsyncExitStack()
        try:
            for backend in self.spec.egress:
                # Preflight every backend before starting ANY of them, and before
                # anything is written: the point is to fail before bwrap, and a
                # box whose second backend has no credential is just as dead as
                # one whose first does not.
                await backend.preflight()
            self._stage(stack)
            activations: dict[Backend, BackendActivation] = {}
            for backend in self.spec.egress:
                if self.spec.unshare_net:
                    await stack.enter_async_context(backend.serve(self.runtime_dir))
                    activation = BackendActivation(backend.port, backend.client_env())
                else:
                    activation = await stack.enter_async_context(
                        backend.serve_shared(self.runtime_dir)
                    )
                activations[backend] = activation
            if not self.spec.unshare_net and activations:
                client_env: dict[str, str] = {}
                for activation in activations.values():
                    client_env.update(activation.client_env)
                path = write_client_env(self.runtime_dir, client_env)
                stack.callback(path.unlink, missing_ok=True)
            if self._needs_relays():
                path = write_manifest(self.runtime_dir, self._manifest())
                # Unlinked with everything else: a file left in the runtime dir
                # means its rmdir never succeeds and every box leaks a directory
                # under the system temp dir.
                stack.callback(path.unlink, missing_ok=True)
        except BaseException:
            await stack.aclose()
            raise
        self._activations = activations
        self._stack = stack
        return self

    def _manifest(self) -> list[dict[str, object]]:
        return [
            {
                "socket": str(b.socket_path(self.runtime_dir)),
                "port": b.port,
                "name": b.name,
            }
            for b in self.spec.egress
        ]

    async def __aexit__(self, *exc) -> None:
        stack, self._stack = self._stack, None
        self._activations = {}
        if stack is not None:
            await stack.aclose()

    def activation(self, backend: Backend) -> BackendActivation:
        """The backend's live per-Box endpoint, only inside the async bracket."""
        try:
            return self._activations[backend]
        except KeyError as e:
            raise RuntimeError("backend activation is not live in this Box") from e

    @contextmanager
    def staged(self) -> Iterator[Box]:
        """The box's files on disk, but nothing serving and nothing minted.

        What an inspector wants: `wrapper()` resolves only if every bind-over
        source exists, but describing a profile must not mint a credential or
        open a socket. This is the same `_stage` the real path runs, so what
        `explain` prints is the argv the box would get -- the fidelity the
        inspector triad rests on.
        """
        with ExitStack() as stack:
            self._stage(stack)
            yield self

    @property
    def env(self) -> dict[str, str]:
        """Environment serialized into bwrap argv.

        Isolated mode includes each backend's meaningless placeholder values.
        Shared mode deliberately does not: its live ports and proxy tokens are
        injected by the launcher from the protected runtime file, so they never
        appear in a host process command line.
        """
        env = dict(self.spec.env)
        if self.spec.unshare_net:
            for backend in self.spec.egress:
                env.update(backend.client_env())
        return env

    def _sandbox(self) -> Sandbox:
        """The spec plus what egress costs, as a resolvable Sandbox.

        Three additions, and each is exactly what the spec deliberately does not
        carry -- something derived from the box's IDENTITY, which does not exist
        until there is a box:

        - aisan's own runtime, ro, in EVERY box and not just the ones with
          egress. A spec requirement rather than the consumer's favour (see
          launch.launcher_binds), and unconditional so that "aisan's runtime is
          present" is an invariant of a Box rather than a property of a
          particular spec. Conditioning it on egress would make adding a backend
          to a working box change what is mounted in it, which is the class of
          coupling this whole extraction exists to remove. RO is the DEFAULT,
          not the requirement: a launcher path the spec already grants writable
          (a box rooted at aisan's own checkout is the case that forced this)
          keeps the spec's grant, and the launcher bind for it is dropped
          below -- a default does not downgrade an explicit grant.
        - the runtime dir, ro, so isolated relays can reach sockets and the
          shared-network launcher can read its protected client environment.
          The box has no business changing either control surface.
        - each backend's own mounts, whose sources live in the runtime dir.

        Appended LAST, so a consumer that binds one of these paths differently
        still loses to the box's own version. That is the right way round: a box
        whose sockets are shadowed is a box whose egress silently does not work.
        The launcher binds are the one exception, and it is a DROP rather than
        an ordering: their winning is never load-bearing -- the launcher reads
        and execs, it does not write -- while their shadowing a writable grant
        subtracts a freedom the spec stated.

        Composed on the UNRESOLVED bind list and resolved once, never by
        concatenating two resolved lists: a Seal resolves to a tmpfs now plus a
        deferred --remount-ro at the very END, and gluing two resolved lists puts
        one seal's remount before the other's holes are punched. There is exactly
        one resolve() call per box and it is inside the Sandbox below.
        """
        # The credential-absence invariant, over the SPEC's own binds: every
        # other source in the list below is library-authored (launcher binds,
        # the runtime dir, backend bind-overs), but a spec is caller input --
        # and the one thing a spec must not be able to state is "and also mount
        # the credential". Same class of rule as egress-requires-unshare_net:
        # a library exporting an unsafe combination hands a caller an unsafe
        # box, and one that refuses to build it cannot. Here rather than in
        # BoxSpec.__post_init__ because the backends own the paths.
        from .egress.base import credential_exposure

        # `root` is also a host bind, even though Sandbox models it in its own
        # field rather than in `binds`. Audit it through the same predicate so
        # choosing $HOME (or another credential-containing ancestor) as the
        # writable root cannot bypass the guard.
        exposure = credential_exposure(
            (Bind(self.spec.root, RW), *self.spec.binds), self.spec.egress
        )
        if exposure is not None:
            src, backend, cred = exposure
            raise ValueError(
                f"box {self.box_id}: root or spec bind {src} would expose the"
                f" {backend.name} backend's credential at {cred} -- the"
                " credential stays out of the box by subtraction, and a bind"
                " naming it (or an ancestor of it) undoes that"
            )

        # Imported here, not at module scope: `launch` is executed in the box as
        # `python -m ...aisan.launch`, and runpy warns (and re-executes the
        # module) when the package __init__ has already imported it. __init__
        # imports this module, so a top-level import here would put launch on
        # that chain.
        from .launch import launcher_binds

        # A launcher bind states a REQUIREMENT -- aisan importable, the
        # interpreter exec'able -- and RO is the least-privilege way to meet
        # it, not a policy over the consumer's tree. Where the spec already
        # grants one of these paths writable (the root, or an RW bind), that
        # grant both meets the requirement and says more trust than the
        # default, and re-appending RO after it downgrades an explicit grant
        # to an implementation detail. The case in point: a box rooted at
        # aisan's own checkout, where `<root>/src` and `<root>/.venv` came
        # out read-only and the session existed to edit them. DROPPED rather
        # than reordered, because the runtime dir and backend bind-overs
        # appended below must keep winning (a shadowed socket dir is
        # silently dead egress) while a launcher bind's win here would only
        # subtract writability.
        binds = list(self.spec.binds)
        binds += [
            b
            for b in launcher_binds()
            if not (
                isinstance(b, Bind)
                and b.mode is RO
                and _rw_grant_covers(b.path, self.spec)
            )
        ]
        if self.spec.egress:
            binds.append(runtime_bind(self.box_id))
            for backend in self.spec.egress:
                binds += backend.box_binds(self.runtime_dir)
        return Sandbox(
            root=self.spec.root,
            binds=tuple(binds),
            tmpfs=self.spec.tmpfs,
            env=tuple(self.env.items()),
            memory_max=self.spec.limits.memory_max,
            cpu_quota=self.spec.limits.cpu_quota,
            tasks_max=self.spec.limits.tasks_max,
            slice_unit=self.spec.limits.slice_unit,
            use_cgroup=self.spec.limits.use_cgroup,
            unshare_net=self.spec.unshare_net,
        )

    def mounts(self) -> list[Mount]:
        """The resolved mount list, in the order bwrap will apply it."""
        return self._sandbox().resolve()

    def wrapper(self) -> list[str]:
        """The argv prefix: [systemd-run ...] bwrap ... -- ready to have the
        payload appended.

        Checks that the runtime dir really is mounted when there are backends.
        `_sandbox` adds that bind, so this is not checking the spec -- it is
        checking that the bind SURVIVED resolution. It is optional (a box being
        explained may have no runtime dir yet), so a wrapper() called outside
        the bracket, before `_stage` created the directory, silently drops it
        and produces a box whose relays have no sockets, or whose shared-network
        launcher has no client environment. That presents as every model call
        failing inside a box that otherwise looks correct, far from the missing
        `async with` that caused it.
        """
        sandbox = self._sandbox()
        if self.spec.egress and not any(
            m.dst == self.runtime_dir and m.op == "ro" for m in sandbox.resolve()
        ):
            raise ValueError(
                f"box {self.box_id}: spec has egress backends but does not bind"
                f" the runtime dir {self.runtime_dir} ro -- the launcher would"
                " have no egress control data"
            )
        return sandbox.wrapper()

    def launch_prefix(self) -> list[str]:
        """The argv prefix that supplies the selected egress transport.

        Empty when there is no egress: no backends means no relays or protected
        client environment, so the payload need not carry a Python launcher.
        """
        if not self.spec.egress:
            return []
        from .launch import launch_prefix

        return launch_prefix(self.runtime_dir)

    def command(self, payload: list[str]) -> list[str]:
        """The complete argv: wrapper, launcher, payload.

        The one call site that has to know these three compose in this order, so
        no consumer has to.

        An argv, never a command string, and that is an invariant rather than a
        convenience. A caller handed a string has to spawn it through a shell, and
        the shell it uses is the HOST's -- so the payload's bytes get a round of
        host-side word splitting, globbing and substitution before bwrap has run
        anything. The confinement is downstream of that, which means a `$(...)` in
        a payload executes outside the box. Returning argv keeps every caller on
        `shell=False`, where the bytes are arguments and nothing interprets them.
        This also matches sandbox-runtime's `wrapWithSandboxArgv` design; the
        implementation here is independent.
        """
        return [*self.wrapper(), *self.launch_prefix(), *payload]

    @property
    def refused(self) -> Exception | None:
        """The first backend credential refusal of this box's life, or None."""
        for backend in self.spec.egress:
            if backend.refused is not None:
                return backend.refused
        return None
