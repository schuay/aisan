# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Manage a box and its host-side egress services as one async context.

Entering the context performs these steps in order:

1. Preflight every backend before bwrap can take over the terminal, because
   credential repair such as `luci-auth` reauthentication may be interactive.
2. Create the runtime directory and prepare each backend's bind sources.
3. Start the backends and write either the relay manifest or protected shared
   client environment consumed by the launcher.
4. Resolve the wrapper only after every bind source exists.

Exiting unwinds all resources. `box_id` is an opaque string used only to derive
the runtime directory name.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Self

from . import seatbelt as _seatbelt
from .egress import known_credential_paths
from .egress.base import Backend, BackendActivation, credential_overlap
from .private import private_root
from .runtime import (
    cleanup_runtime_dir,
    cleanup_scratch_dir,
    prepare_runtime_dir,
    prepare_scratch_dir,
    profile_path,
    runtime_bind,
    runtime_dir,
    scratch_bind,
    scratch_dir,
    write_client_env,
    write_manifest,
)
from .sandbox import (
    RO,
    RW,
    Bind,
    BindOver,
    BindSpec,
    Mount,
    PinnedSymlink,
    Sandbox,
    SandboxBackend,
    Seal,
    host_platform,
)
from .seatbelt import SeatbeltSandbox, seatbelt_path, through_darwin_system_symlink
from .spec import BoxSpec
from .statedir import is_runtime_symlink


def _rw_grant_covers(path: Path, spec: BoxSpec) -> bool:
    """Return whether the spec already grants `path` writable.

    The host-bound writable root or a mandatory writable ancestor provides the
    visibility required by the launcher. Optional binds do not count because
    resolution may omit them along with that visibility, and a tmpfs root hides
    the host directory it is mounted over.

    This predicate suppresses only the library's default launcher binds. It does
    not alter read-only pins explicitly ordered by the caller.
    """

    def within(p: Path, ancestor: Path) -> bool:
        # Compare literal mount destinations. Symlink aliases may resolve to one
        # host directory while remaining distinct paths required inside the box.
        return p.absolute().is_relative_to(ancestor.absolute())

    if not spec.root_tmpfs and within(path, spec.root):
        return True
    return any(
        isinstance(b, Bind) and b.mode is RW and not b.optional and within(path, b.path)
        for b in spec.binds
    )


log = logging.getLogger(__name__)


def _undo_host_paths(files: tuple[Path, ...], dirs: tuple[Path, ...]) -> None:
    """Remove created files, then created directories from deepest to shallowest.

    Leave nonempty directories intact in case another box has used them.
    """
    for f in files:
        with suppress(OSError):
            f.unlink()
    for d in sorted(set(dirs), key=lambda p: len(p.parts), reverse=True):
        with suppress(OSError):
            d.rmdir()


def _unlink_staged_symlink(dst: Path, expected_src: Path) -> None:
    """Remove a staged `PinnedSymlink` only if it is still a link to `expected_src`."""
    with suppress(OSError):
        if dst.is_symlink() and Path(os.readlink(dst)) == expected_src:
            dst.unlink(missing_ok=True)


_PLATFORMS = frozenset({"linux", "darwin"})


def _default_platform() -> str:
    """Return the active confinement platform identifier."""
    return host_platform()


class Box:
    """A running mount policy with its host-side egress services.

    `wrapper()` and `launch_prefix()` are valid only inside the async context
    because they refer to resources created and removed by that context.
    """

    def __init__(
        self,
        spec: BoxSpec,
        *,
        box_id: str,
        platform: str | None = None,
    ) -> None:
        resolved = _default_platform() if platform is None else platform
        if resolved not in _PLATFORMS:
            raise ValueError(
                f"unsupported box platform {resolved!r}; expected one of"
                f" {sorted(_PLATFORMS)}"
            )
        self.spec = spec
        self.box_id = box_id
        self.platform = resolved
        self.runtime_dir = runtime_dir(box_id)
        self.scratch_dir = scratch_dir(box_id)
        self._staged_profile_path: Path | None = None
        self._verify_symlinks: bool = True
        self._stack: AsyncExitStack | None = None
        self._activations: dict[Backend, BackendActivation] = {}

    def _clear_staged_profile(self) -> None:
        self._staged_profile_path = None
        self._verify_symlinks = True

    def _uses_shared_tcp(self, backend: Backend) -> bool:
        """Return whether `backend` serves authenticated TCP on host loopback.

        macOS has no network namespaces, so HTTP backends always use
        authenticated host-loopback TCP while Seatbelt pins outbound ports.
        Backends whose clients can dial a Unix socket directly keep their
        private socket in isolated mode.
        """
        if self.platform == "darwin":
            return not (self.spec.unshare_net and backend.supports_direct_unix_socket)
        return not self.spec.unshare_net

    def _validate_platform_egress(self) -> None:
        unsupported = [
            b.name
            for b in self.spec.egress
            if self._uses_shared_tcp(b) and not b.supports_shared_net
        ]
        if unsupported:
            raise ValueError(
                f"box {self.box_id}: backend(s) {unsupported} require a network"
                f" namespace and cannot serve on {self.platform}"
            )

    def _needs_relays(self) -> bool:
        return (
            self.platform != "darwin"
            and bool(self.spec.egress)
            and self.spec.unshare_net
        )

    def _needs_launcher(self) -> bool:
        return self._needs_relays() or any(
            self._uses_shared_tcp(b) for b in self.spec.egress
        )

    def _activated_endpoints(
        self,
    ) -> tuple[tuple[int | str, ...], tuple[Path, ...]]:
        """Return `(tcp_ports, unix_sockets)` for sandbox network rules."""
        tcp_ports = tuple(
            self._activations[b].port if b in self._activations else f"<PORT:{b.name}>"
            for b in self.spec.egress
            if self._uses_shared_tcp(b)
        )
        unix_sockets = tuple(
            b.socket_path(self.runtime_dir)
            for b in self.spec.egress
            if not self._uses_shared_tcp(b)
        )
        return tcp_ports, unix_sockets

    def _backend_binds(self, backend: Backend) -> list[BindSpec]:
        """Adapt backend bind-over/symlink entries to `self.platform`."""
        raw_list = backend.box_binds(self.runtime_dir, platform=self.platform)
        adapted: list[BindSpec] = []
        for item in raw_list:
            if self.platform == "linux" and isinstance(item, PinnedSymlink):
                adapted.append(BindOver(item.src, item.dst))
            else:
                adapted.append(item)
        return adapted

    def _stage(
        self, stack: ExitStack | AsyncExitStack, *, stage_symlinks: bool = True
    ) -> None:
        """Create every on-disk source required before wrapper resolution.

        This includes ensure-paths, the runtime directory, backend bind-over
        sources, and Darwin scratch/symlink staging. Both live boxes and
        `staged()` inspection use this method so the inspector resolves the
        same profile. Every created resource registers cleanup on the supplied
        stack.
        """
        self._ensure_host_paths(stack)
        prepare_runtime_dir(self.box_id)
        # Register cleanup before backend preparation can fail.
        stack.callback(cleanup_runtime_dir, self.box_id)
        if self.platform == "darwin":
            prepare_scratch_dir(self.box_id)
            stack.callback(cleanup_scratch_dir, self.box_id)
            prof = profile_path(self.box_id)
            prof.unlink(missing_ok=True)
            self._staged_profile_path = prof
            self._verify_symlinks = stage_symlinks
            stack.callback(self._clear_staged_profile)
            stack.callback(prof.unlink, missing_ok=True)
        backend_binds: list[BindSpec] = []
        for backend in self.spec.egress:
            backend.prepare(self.runtime_dir)
            stack.callback(backend.cleanup_prepared, self.runtime_dir)
            # Remove prepared bind-over sources owned by this runtime directory.
            # Sources outside it belong to the backend's caller and remain intact.
            for spec in self._backend_binds(backend):
                backend_binds.append(spec)
                src = getattr(spec, "src", None)
                if src is not None and src.parent == self.runtime_dir:
                    stack.callback(src.unlink, missing_ok=True)
        if self.platform == "darwin" and stage_symlinks:
            self._stage_pinned_symlinks(
                stack,
                tuple(
                    b
                    for b in (*self.spec.binds, *backend_binds)
                    if isinstance(b, PinnedSymlink)
                ),
            )

    def _stage_pinned_symlinks(
        self,
        stack: ExitStack | AsyncExitStack,
        symlinks: tuple[PinnedSymlink, ...],
    ) -> None:
        """Stage `PinnedSymlink` entries without overwriting user-owned files."""
        if not symlinks:
            return
        created_dirs: list[Path] = []
        staged_links: list[tuple[Path, Path]] = []
        priv = private_root()

        def cleanup_staged() -> None:
            for dst, src in reversed(staged_links):
                _unlink_staged_symlink(dst, src)
            if created_dirs:
                _undo_host_paths((), tuple(created_dirs))

        stack.callback(cleanup_staged)
        for sym in symlinks:
            norm_parent = through_darwin_system_symlink(sym.dst).parent
            for p in (norm_parent, *norm_parent.parents):
                if p.is_symlink():
                    raise ValueError(
                        f"{p} is a symlink; refusing to stage {sym.dst} through it"
                    )
            cursor = norm_parent
            while not cursor.exists():
                created_dirs.append(cursor)
                if cursor == cursor.parent:
                    break
                cursor = cursor.parent
            norm_parent.mkdir(mode=0o700, parents=True, exist_ok=True)

            try:
                info = os.lstat(sym.dst)
            except FileNotFoundError:
                info = None
            if info is not None:
                if stat.S_ISREG(info.st_mode) and info.st_size == 0:
                    sym.dst.unlink()
                elif stat.S_ISLNK(info.st_mode):
                    target = Path(os.readlink(sym.dst))
                    if target == sym.src or is_runtime_symlink(sym.dst, priv):
                        sym.dst.unlink()
                    else:
                        raise FileExistsError(
                            f"refusing to overwrite unexpected symlink at {sym.dst}"
                        )
                else:
                    raise FileExistsError(
                        f"refusing to overwrite existing path at {sym.dst}"
                    )
            sym.dst.symlink_to(sym.src)
            staged_links.append((sym.dst, sym.src))

    def _ensure_host_paths(self, stack: ExitStack | AsyncExitStack) -> None:
        """Create missing ensure-paths empty and register their cleanup.

        Record every created file and parent directory so cleanup removes only
        this box's additions. For the Git guards involved, empty and absent have
        the same effect on host-side Git.
        """
        created_files: list[Path] = []
        created_dirs: list[Path] = []

        def ensure_dir(d: Path) -> None:
            # `mkdir(parents=True)` follows a symlink in any component, so a
            # box that replaced a directory on this path with a link would
            # have the next staging create the guard source wherever the link
            # points. Refuse a symlinked component inside anything the box can
            # write; components above that are the host's own.
            for p in (d, *d.parents):
                if _rw_grant_covers(p, self.spec) and p.is_symlink():
                    raise ValueError(
                        f"{p} is a symlink; refusing to create {d} through it"
                    )
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
                # The writable .git directory may contain an agent-planted
                # symlink. O_EXCL creates only an absent file, and O_NOFOLLOW
                # prevents touching a host target through that symlink.
                try:
                    fd = os.open(
                        e.path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                        0o600,
                    )
                    os.close(fd)
                    created_files.append(e.path)
                except FileExistsError:
                    pass
        if created_files or created_dirs:
            stack.callback(_undo_host_paths, tuple(created_files), tuple(created_dirs))

    async def __aenter__(self) -> Self:
        self._validate_platform_egress()
        stack = AsyncExitStack()
        try:
            for backend in self.spec.egress:
                # Preflight the complete set before starting services or writing
                # state, so one missing credential aborts the box cleanly.
                await backend.preflight()
            self._stage(stack)
            activations: dict[Backend, BackendActivation] = {}
            shared_backends = False
            client_env: dict[str, str] = {}
            for backend in self.spec.egress:
                if self._uses_shared_tcp(backend):
                    activation = await stack.enter_async_context(
                        backend.serve_shared(
                            self.runtime_dir, isolated=self.spec.unshare_net
                        )
                    )
                    shared_backends = True
                    client_env.update(activation.client_env)
                else:
                    await stack.enter_async_context(
                        backend.serve(self.runtime_dir, platform=self.platform)
                    )
                    act_env = (
                        backend.direct_unix_client_env(self.runtime_dir)
                        if self.platform == "darwin"
                        else backend.client_env()
                    )
                    activation = BackendActivation(backend.port, act_env)
                activations[backend] = activation
            if shared_backends:
                path = write_client_env(self.runtime_dir, client_env)
                stack.callback(path.unlink, missing_ok=True)
            if self._needs_relays():
                path = write_manifest(self.runtime_dir, self._manifest())
                # The runtime directory can be removed only after this file.
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
        """Return a backend's live endpoint within this box's async context."""
        try:
            return self._activations[backend]
        except KeyError as e:
            raise RuntimeError("backend activation is not live in this Box") from e

    @contextmanager
    def staged(self, *, stage_symlinks: bool = True) -> Iterator[Box]:
        """Stage the box for inspection without serving or minting credentials.

        Wrapper resolution requires every bind-over source. Reuse the live
        staging path to create those sources without opening sockets, ensuring
        `explain` reports the same mount profile a live box receives. When
        `stage_symlinks=False` (used while another session holds the state-dir
        lock), omit host `PinnedSymlink` mutation while rendering identical SBPL.
        """
        with ExitStack() as stack:
            self._stage(stack, stage_symlinks=stage_symlinks)
            yield self

    @property
    def env(self) -> dict[str, str]:
        """Return the environment serialized into the wrapper argv.

        Backends served over Unix sockets include static placeholder values
        here. Backends served on authenticated host loopback instead pass live
        ports and proxy tokens through the protected runtime file, keeping them
        out of host process command lines.
        """
        env = dict(self.spec.env)
        if self.platform == "darwin":
            if "TMPDIR" not in env:
                env["TMPDIR"] = seatbelt_path(self.scratch_dir)
            env.setdefault("CLAUDE_CODE_TMPDIR", env["TMPDIR"])
            env.setdefault("XDG_CACHE_HOME", f"{env['TMPDIR']}/xdg-cache")
            env.setdefault("NPM_CONFIG_CACHE", f"{env['TMPDIR']}/npm-cache")
            env.setdefault("GIT_CONFIG_GLOBAL", "/dev/null")
        for backend in self.spec.egress:
            if not self._uses_shared_tcp(backend):
                if self.platform == "darwin":
                    env.update(backend.direct_unix_client_env(self.runtime_dir))
                else:
                    env.update(backend.client_env())
        return env

    def _sandbox(self) -> SandboxBackend:
        """Combine the spec with runtime mounts and return a resolvable sandbox.

        A box adds four kinds of mounts whose paths depend on runtime identity:

        * aisan's interpreter and source runtime, read-only unless the spec
          already grants the path writable;
        * a seal over the private host root, hiding credential children and
          other boxes' runtime capabilities;
        * this box's runtime directory, read-only, for relay sockets or shared
          client state;
        * backend-specific mounts sourced from that directory.

        The seal, the runtime directory, and the backend mounts are guards, so
        the spec cannot mount anything at or below them; the sandbox refuses the
        profile instead. Launcher mounts are omitted when an explicit writable
        grant already provides the path, preserving that grant.
        """
        self._validate_platform_egress()
        # Keep this import off the package initialization path. `python -m`
        # otherwise finds `launch` already imported, warns, and re-executes it.
        from .launch import launcher_binds

        # Launcher binds provide visibility with the least privilege. Omit one
        # when the spec already grants its path writable; re-appending it would
        # unexpectedly downgrade that explicit grant. This matters when a box is
        # rooted at the aisan checkout and needs to edit its own source or venv.
        binds = list(self.spec.binds)
        lb_items = (
            launcher_binds()
            if self.platform == host_platform()
            else launcher_binds(platform=self.platform)
        )
        binds += [
            b
            for b in lb_items
            if not (
                isinstance(b, Bind)
                and b.mode is RO
                and _rw_grant_covers(b.path, self.spec)
            )
        ]
        # Hide the shared host-control namespace, then expose only this box's
        # runtime directory (and per-box scratch dir on Darwin).
        holes: list[Path] = []
        if self.platform == "darwin":
            holes.append(self.scratch_dir)
            binds.append(scratch_bind(self.box_id))
        if self.spec.egress:
            holes.append(self.runtime_dir)
        binds.append(
            Seal(
                private_root(),
                allow_missing=True,
                allow=tuple(holes),
            )
        )
        if self.spec.egress:
            binds.append(runtime_bind(self.box_id))
            for backend in self.spec.egress:
                binds += self._backend_binds(backend)

        if self.platform == "darwin":
            from .userbinds import _refused_key_stores

            priv_resolved = private_root().resolve()
            ext_sym_srcs = tuple(
                b.src.resolve()
                for b in binds
                if isinstance(b, PinnedSymlink)
                and not b.src.resolve().is_relative_to(priv_resolved)
            )
            if ext_sym_srcs:
                ks_hit = credential_overlap(ext_sym_srcs, _refused_key_stores())
                if ks_hit is not None:
                    raise ValueError(
                        f"box {self.box_id}: PinnedSymlink source {ks_hit[0]} would"
                        f" expose host key store {ks_hit[1]}"
                    )
                cr_hit = credential_overlap(ext_sym_srcs, known_credential_paths())
                if cr_hit is not None:
                    raise ValueError(
                        f"box {self.box_id}: PinnedSymlink source {cr_hit[0]} would"
                        f" expose host credential {cr_hit[1]}"
                    )
        sandbox: SandboxBackend
        if self.platform == "darwin":
            tcp_ports, unix_sockets = self._activated_endpoints()
            sandbox = SeatbeltSandbox(
                root=self.spec.root,
                binds=tuple(binds),
                env=tuple(self.env.items()),
                unshare_net=self.spec.unshare_net,
                root_tmpfs=self.spec.root_tmpfs,
                allow_local_loopback=self.spec.allow_local_loopback,
                mach_profiles=self.spec.mach_profiles,
                tcp_ports=tcp_ports,
                unix_sockets=unix_sockets,
                controlling_ttys=_seatbelt.darwin_controlling_ttys(),
                profile_path=self._staged_profile_path,
                verify_symlinks=self._verify_symlinks,
            )
        else:
            sandbox = Sandbox(
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
                root_tmpfs=self.spec.root_tmpfs,
            )

        # Every seal states that its contents stay out of the box, so an alias
        # to a sealed directory is refused here rather than at `wrapper`. This
        # covers the private host-control root -- other boxes' runtime sockets
        # and shared-network tokens -- along with the spec's own seals, and
        # catches aliases created by bind-over operations.
        hit = sandbox.sealed_exposure()
        if hit is not None:
            src, path = hit
            raise ValueError(
                f"box {self.box_id}: mount {src} would republish the sealed"
                f" directory {path} under a second name, where the seal does"
                " not reach"
            )

        # Check paths kept out by subtraction against the finished mount list,
        # including the fixed system surface and library-added binds. This runs
        # after composition because backends own their credential paths. One
        # call per group walks the mount list once for all of that group's
        # paths and still names the one it found.
        egress = self.spec.egress
        groups = [("path declared confidential", self.spec.confidential)]
        if self.platform == "darwin":
            from .userbinds import _refused_key_stores

            groups.append(("host key store", _refused_key_stores()))
        groups += [(f"{b.name} backend's credential", b.credentials) for b in egress]
        for what, paths in groups:
            hit = sandbox.exposed_path(paths)
            if hit is not None:
                src, target = hit
                raise ValueError(
                    f"box {self.box_id}: mount {src} would expose the {what}"
                    f" at {target} -- it stays out of the box by subtraction,"
                    " and this mount undoes that"
                )
        return sandbox

    def mounts(self) -> list[Mount]:
        """Return the resolved mount list for the active confinement backend."""
        sandbox = self._sandbox()
        if isinstance(sandbox, Sandbox):
            return sandbox.resolve()
        if isinstance(sandbox, SeatbeltSandbox):
            return sandbox._published()
        raise TypeError(f"unsupported sandbox backend: {type(sandbox)!r}")

    def wrapper(self) -> list[str]:
        """Return the confinement wrapper argv prefix for a payload.

        When egress is configured, verify that the optional runtime-directory
        bind survived resolution. Calling this outside the lifecycle, before
        `_stage` creates the directory, would otherwise omit the bind and leave
        the launcher without sockets or shared client state.
        """
        sandbox = self._sandbox()
        if self.platform == "darwin" and not self.scratch_dir.is_dir():
            raise ValueError(
                f"box {self.box_id}: Darwin scratch dir {self.scratch_dir} does"
                " not exist -- call wrapper() inside `async with box:` or"
                " `with box.staged():`"
            )
        if self.spec.egress:
            try:
                sandbox.assert_runtime_visible(self.runtime_dir)
            except ValueError as e:
                raise ValueError(f"box {self.box_id}: {e}") from e
        return sandbox.wrapper()

    def launch_prefix(self) -> list[str]:
        """Return the launcher argv prefix for the selected egress transport.

        A box without relays or protected shared client state can launch the
        payload directly.
        """
        if not self._needs_launcher():
            return []
        from .launch import launch_prefix

        return launch_prefix(self.runtime_dir)

    def command(self, payload: list[str]) -> list[str]:
        """Return the complete wrapper, launcher, and payload argv.

        Returning an argv keeps callers on `shell=False`. A command string would
        require a host shell, allowing word splitting, globbing, or substitution
        such as `$(...)` to execute before bwrap establishes confinement.
        """
        return [*self.wrapper(), *self.launch_prefix(), *payload]

    @property
    def refused(self) -> Exception | None:
        """Return the first backend credential refusal, if any."""
        for backend in self.spec.egress:
            if backend.refused is not None:
                return backend.refused
        return None
