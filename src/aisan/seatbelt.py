# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Compile a mount and network policy into a macOS Seatbelt (`sandbox-exec`) profile.

Seatbelt evaluates canonical vnode paths (`vn_getpath`) against an ordered list
of `(version 1)` SBPL rules where the last matching rule wins. Unlike Linux
`bwrap`, Seatbelt cannot remap paths (`BindOver`), mount copy-on-write overlays
(`Overlay`), or replace `/tmp` and `$HOME` with per-box `tmpfs` mounts.

It preserves `aisan`'s core guarantees through five rules:

* `(deny default)` hides the host home directory and every undeclared path;
  only the fixed macOS runtime surface and declared `Bind` trees are readable.
* Every path emitted into SBPL and checked for guard, conflict, or credential
  exposure is canonicalized to XNU's `F_GETPATH` spelling (resolving symlinks,
  APFS case-folding, `/tmp`/`/var`/`/etc` -> `/private/...`, and mapping APFS
  `/System/Volumes/Data` firmlinks back to their synthetic root paths).
* Policy mounts are emitted in canonical path-depth order: `RO` mounts emit an
  explicit `(deny file-write*)` so pins inside writable trees cannot be edited,
  and `Seal` directories deny read, existence-testing, and write access while
  permitting `file-read-metadata` along the exact ancestor chain to allowed
  holes so `realpath(3)` succeeds inside them.
* Every mount point and every writable directory above a mount or
  `PinnedSymlink` receives a final `(deny file-write-unlink file-write-create
  (literal ...))` rule, giving Seatbelt the rename-and-recreate protection that
  VFS mount points provide on Linux.
* Network isolation (`unshare_net=True`) denies `network*` and opens only the
  activated host-proxy TCP ports and Unix domain sockets (plus localhost TCP
  when `allow_local_loopback=True`). Neither `com.apple.SecurityServer` nor
  `com.apple.securityd.xpc` is ever permitted.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .policy_tree import (
    Bind,
    BindOver,
    BindSpec,
    Entry,
    GuardViolation,
    Mount,
    MountConflict,
    Op,
    Overlay,
    PinnedSymlink,
    Seal,
    SymlinkDestination,
    WriteRegex,
    _f_getpath,
    _strip_firmlink,
    check_guards,
    check_surface,
    check_symlink_free,
    entry_for,
    first_exposure,
    merge_entries,
    mount_for,
    resolved_path,
    validate_destination,
)
from .spec import _validate_mach_profiles

__all__ = [
    "SeatbeltSandbox",
    "darwin_controlling_ttys",
    "darwin_selected_xcode_app",
    "darwin_system_ro_roots",
    "seatbelt_lexpath",
    "seatbelt_path",
    "through_darwin_system_symlink",
]

_PORT_PLACEHOLDER_RE = re.compile(r"^<PORT:[^>\x00\r\n\"]+>$")
_DEV_TTYS_RE = re.compile(r"^/dev/ttys[0-9]+$")
_XCODE_SELECT_LINK = Path("/private/var/db/xcode_select_link")

# Signed immutable root symlinks on macOS.
_DARWIN_ROOT_SYMLINKS: tuple[tuple[Path, Path], ...] = (
    (Path("/") / "tmp", Path("/private/tmp")),
    (Path("/") / "var", Path("/private/var")),
    (Path("/") / "etc", Path("/private/etc")),
)

# Read-only system subpaths required by dyld, POSIX utilities, and toolchains.
_DARWIN_SYSTEM_SUBPATHS: tuple[Path, ...] = (
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/System"),
    Path("/System/Volumes/Preboot/Cryptexes"),
    Path("/Library/Apple"),
    Path("/private/etc/ssl"),
    Path("/private/var/select"),
    Path("/private/var/db/timezone"),
    Path("/dev/fd"),
)

# Read-only Apple developer toolchains (`xcrun` shims such as `/usr/bin/git`
# resolve `/private/var/db/xcode_select_link` into one of these trees).
_DARWIN_DEVELOPER_SUBPATHS: tuple[Path, ...] = (
    Path("/Library/Developer"),
    Path("/Applications/Xcode.app"),
)

# Read-only Homebrew subpaths when Homebrew is installed on Apple Silicon,
# intentionally excluding `/opt/homebrew/var` (service state and logs) and
# `/opt/homebrew/.git`.
_DARWIN_HOMEBREW_SUBPATHS: tuple[Path, ...] = (
    Path("/opt/homebrew/bin"),
    Path("/opt/homebrew/sbin"),
    Path("/opt/homebrew/lib"),
    Path("/opt/homebrew/include"),
    Path("/opt/homebrew/share"),
    Path("/opt/homebrew/opt"),
    Path("/opt/homebrew/Cellar"),
    Path("/opt/homebrew/Caskroom"),
    Path("/opt/homebrew/Frameworks"),
    Path("/opt/homebrew/etc/ca-certificates"),
    Path("/opt/homebrew/etc/openssl@3"),
    Path("/opt/homebrew/etc/openssl@4"),
)

_DARWIN_SYSTEM_LITERALS: tuple[Path, ...] = (
    Path("/"),
    Path("/dev"),
    Path("/Library/Preferences/com.apple.dt.Xcode.plist"),
    Path("/private/var/db/xcode_select_link"),
    Path("/private/etc/hosts"),
    Path("/private/etc/localtime"),
    Path("/private/etc/resolv.conf"),
    Path("/private/etc/passwd"),
    Path("/private/etc/group"),
    Path("/private/etc/services"),
    Path("/private/etc/protocols"),
    Path("/dev/null"),
    Path("/dev/zero"),
    Path("/dev/random"),
    Path("/dev/urandom"),
    Path("/dev/tty"),
    Path("/dev/ptmx"),
)

_SYSCTL_NAMES: tuple[str, ...] = (
    "hw.activecpu",
    "hw.busfrequency_compat",
    "hw.byteorder",
    "hw.cacheconfig",
    "hw.cachelinesize",
    "hw.cachelinesize_compat",
    "hw.cpufamily",
    "hw.cpufrequency_compat",
    "hw.cpusubfamily",
    "hw.cputhreadtype",
    "hw.cputype",
    "hw.l1dcachesize",
    "hw.l1dcachesize_compat",
    "hw.l1icachesize",
    "hw.l1icachesize_compat",
    "hw.l2cachesize",
    "hw.l2cachesize_compat",
    "hw.l3cachesize",
    "hw.l3cachesize_compat",
    "hw.logicalcpu",
    "hw.logicalcpu_max",
    "hw.machine",
    "hw.memsize",
    "hw.ncpu",
    "hw.nperflevels",
    "hw.packages",
    "hw.pagesize",
    "hw.pagesize_compat",
    "hw.physicalcpu",
    "hw.physicalcpu_max",
    "hw.tbfrequency_compat",
    "hw.vectorunit",
    "kern.argmax",
    "kern.bootargs",
    "kern.hostname",
    "kern.maxfiles",
    "kern.maxfilesperproc",
    "kern.maxproc",
    "kern.ngroups",
    "kern.osproductversion",
    "kern.osrelease",
    "kern.ostype",
    "kern.osvariant_status",
    "kern.osversion",
    "kern.secure_kernel",
    "kern.usrstack64",
    "kern.version",
    "kern.wq_limit_cooperative_threads",
    "machdep.cpu.brand_string",
    "security.mac.lockdown_mode_state",
    "sysctl.proc_cputype",
    "vm.loadavg",
)


def _check_clean_path(raw: str) -> None:
    if any(ch in raw for ch in "\x00\r\n"):
        raise ValueError(f"Seatbelt path contains forbidden control character: {raw!r}")


_ERE_METACHARS = frozenset(r".^$*+?()[]{}\|")


def _ere_escape(raw: str) -> str:
    """Escape POSIX ERE metacharacters without escaping non-special characters."""
    return "".join(f"\\{ch}" if ch in _ERE_METACHARS else ch for ch in raw)


def _sbpl_quote(value: str) -> str:
    """Quote a string literal for SBPL."""
    _check_clean_path(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _sbpl_regex(pattern: str) -> str:
    """Quote a POSIX ERE literal for SBPL's raw `#"..."` reader syntax.

    TinyScheme's `#"..."` reader does not interpret `\\"` escapes -- any `"`
    terminates the literal -- so `"` is rejected outright.
    """
    _check_clean_path(pattern)
    if '"' in pattern:
        raise ValueError(f"Seatbelt regex cannot contain '\"': {pattern!r}")
    return f'#"{pattern}"'


def through_darwin_system_symlink(path: Path) -> Path:
    """Map macOS root symlinks and APFS Data firmlinks to canonical prefixes.

    Only system-owned root prefixes (`/tmp`, `/var`, `/etc`, and
    `/System/Volumes/Data/<firmlink>`) are rewritten here; deeper components
    stay untouched until `check_symlink_free` has inspected them.
    """
    stripped = Path(_strip_firmlink(str(path)))
    for link, target in _DARWIN_ROOT_SYMLINKS:
        if stripped == link:
            return target
        if stripped.is_relative_to(link):
            return target / stripped.relative_to(link)
    return stripped


def seatbelt_path(path: Path) -> str:
    """Return XNU's canonical `vn_getpath` spelling for `path`.

    Resolves symlinks (including `/tmp`, `/var`, `/etc`), queries `F_GETPATH`
    on the deepest existing prefix when running on Darwin (normalizing APFS
    case-folding), and maps APFS `/System/Volumes/Data` firmlinks back to their
    synthetic root paths.
    """
    _check_clean_path(str(path))
    resolved = resolved_path(path)
    for anc in (resolved, *resolved.parents):
        got = _f_getpath(anc)
        if got is not None:
            tail = resolved.relative_to(anc)
            combined = got if str(tail) == "." else f"{got.rstrip('/')}/{tail}"
            norm = _strip_firmlink(combined)
            return norm if norm == "/" else norm.rstrip("/")
    norm = _strip_firmlink(str(resolved))
    return norm if norm == "/" else norm.rstrip("/")


def _canonical_path(path: Path) -> Path:
    return Path(seatbelt_path(path))


def seatbelt_lexpath(path: Path) -> str:
    """Canonicalize `path`'s parent while preserving a symlink at `path.name`.

    Operations on a symlink vnode itself (`unlink`, `rename` over a link,
    `readlink`) check the symlink's own directory path rather than its target.
    """
    validate_destination(path)
    _check_clean_path(str(path))
    if path == Path("/"):
        return "/"
    parent = seatbelt_path(path.parent)
    return f"/{path.name}" if parent == "/" else f"{parent}/{path.name}"


def _reject_host_symlinks(path: Path, *, include_self: bool) -> None:
    """Reject symlinks in `path.parents` (and `path` when `include_self=True`)."""
    candidates = (path, *path.parents) if include_self else tuple(path.parents)
    for candidate in candidates:
        if candidate.is_symlink():
            try:
                target = f" -> {candidate.readlink()}"
            except OSError:
                target = ""
            raise SymlinkDestination(
                f"mount destination {path} passes through the symlink {candidate}"
                f"{target}; sandbox-exec requires the resolved path"
            )


def darwin_controlling_ttys() -> tuple[str, ...]:
    """Return `/dev/ttys*` paths attached to stdio fds (0, 1, 2), if any."""
    found: list[str] = []
    for fd in (0, 1, 2):
        try:
            name = os.ttyname(fd)
        except OSError:
            continue
        if _DEV_TTYS_RE.fullmatch(name) and name not in found:
            found.append(name)
    return tuple(found)


def darwin_selected_xcode_app() -> Path | None:
    """Return `/Applications/*.app` selected by `xcode-select`, if any."""
    if not _XCODE_SELECT_LINK.exists():
        return None
    try:
        target = Path(seatbelt_path(_XCODE_SELECT_LINK))
    except OSError:
        return None
    for candidate in (target, *target.parents):
        if (
            candidate.parent == Path("/Applications")
            and candidate.suffix == ".app"
            and candidate.is_dir()
        ):
            return candidate
    return None


def _active_system_subpaths() -> tuple[Path, ...]:
    present_dev = [p for p in _DARWIN_DEVELOPER_SUBPATHS if p.exists()]
    selected_app = darwin_selected_xcode_app()
    if selected_app is not None and selected_app not in present_dev:
        present_dev.append(selected_app)
    present_brew = tuple(p for p in _DARWIN_HOMEBREW_SUBPATHS if p.exists())
    return (*_DARWIN_SYSTEM_SUBPATHS, *present_dev, *present_brew)


def darwin_system_ro_roots() -> frozenset[Path]:
    """Return identity-mounted Darwin system roots that need no duplicate bind."""
    return frozenset(_canonical_path(p) for p in _active_system_subpaths())


def _surface_entries(norm: Callable[[Path], Path]) -> tuple[Entry, ...]:
    return tuple(
        Entry(norm(p), Op.RO, norm(p), fixed=True) for p in _active_system_subpaths()
    )


def _system_mounts() -> list[Mount]:
    sub = [
        Mount("ro", _canonical_path(p), _canonical_path(p))
        for p in _active_system_subpaths()
        if p.exists()
    ]
    lits = [
        Mount("ro", _canonical_path(p), _canonical_path(p))
        for p in _DARWIN_SYSTEM_LITERALS
        if p not in (Path("/"), Path("/dev")) and p.exists()
    ]
    return [*sub, *lits]


@dataclass(frozen=True)
class SeatbeltSandbox:
    """An immutable macOS Seatbelt confinement profile for one job."""

    root: Path
    binds: tuple[BindSpec, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    unshare_net: bool = False
    root_tmpfs: int = 0
    allow_local_loopback: bool = False
    mach_profiles: tuple[str, ...] = ()
    tcp_ports: tuple[int | str, ...] = ()
    unix_sockets: tuple[Path, ...] = ()
    controlling_ttys: tuple[str, ...] = ()
    profile_path: Path | None = None

    def __post_init__(self) -> None:
        validate_destination(self.root)
        if self.root_tmpfs < 0:
            raise ValueError(f"root_tmpfs must not be negative: {self.root_tmpfs}")
        if self.root_tmpfs > 0:
            raise ValueError(
                f"root_tmpfs ({self.root_tmpfs} bytes at {self.root}) is not"
                " supported on macOS Seatbelt"
            )
        _validate_mach_profiles(self.mach_profiles)
        if self.profile_path is not None:
            validate_destination(self.profile_path)
        for sock in self.unix_sockets:
            validate_destination(sock)
        for tty in self.controlling_ttys:
            if not _DEV_TTYS_RE.fullmatch(tty):
                raise ValueError(f"invalid controlling_tty path: {tty!r}")
        for port in self.tcp_ports:
            if (
                isinstance(port, int)
                and not isinstance(port, bool)
                and 1 <= port <= 65535
            ):
                continue
            if isinstance(port, str) and _PORT_PLACEHOLDER_RE.fullmatch(port):
                continue
            raise ValueError(f"invalid Seatbelt tcp_port: {port!r}")

    def _pinned_symlinks(self) -> tuple[PinnedSymlink, ...]:
        return tuple(b for b in self.binds if isinstance(b, PinnedSymlink))

    def _entries_with(self, norm: Callable[[Path], Path]) -> list[Entry]:
        out = list(_surface_entries(norm))
        root_dst = norm(self.root)
        out.append(Entry(root_dst, Op.RW, root_dst, fixed=True))
        for spec in self.binds:
            match spec:
                case BindOver(src=src, dst=dst):
                    raise ValueError(
                        f"BindOver ({src} -> {dst}) is not supported on macOS Seatbelt"
                    )
                case Overlay(path=p):
                    raise ValueError(
                        f"Overlay ({p}) is not supported on macOS Seatbelt"
                    )
                case WriteRegex(path=p, suffixes=suffixes, guard=guard):
                    dst = norm(p)
                    out.append(
                        Entry(
                            dst,
                            Op.WRITE_REGEX,
                            dst,
                            guard=guard,
                            suffixes=suffixes,
                        )
                    )
                case PinnedSymlink(src=src):
                    src_dst = norm(src)
                    out.append(Entry(src_dst, Op.RO, src_dst, guard=True))
                case Bind(path=p, mode=mode, optional=optional, guard=guard):
                    dst = norm(p)
                    out.append(
                        Entry(dst, Op(mode.value), dst, guard=guard, optional=optional)
                    )
                case Seal(path=p, allow_missing=allow_missing):
                    dst = norm(p)
                    out.append(
                        Entry(
                            dst,
                            Op.SEAL,
                            guard=True,
                            fixed=True,
                            allow_missing=allow_missing,
                        )
                    )
                case _:
                    out.append(entry_for(spec))
        return out

    def entries(self) -> list[Entry]:
        """Return every declared mount normalized to canonical vnode paths."""
        return self._entries_with(_canonical_path)

    def _lexical_tree(self) -> dict[Path, Entry]:
        """Build the pre-symlink lexical tree so `check_symlink_free` sees links."""
        lex_entries = self._entries_with(through_darwin_system_symlink)
        lex_surface = frozenset(
            e.dst for e in _surface_entries(through_darwin_system_symlink)
        )
        lex_tree = merge_entries(lex_entries)
        check_surface(lex_tree, lex_surface, tool="sandbox-exec")
        check_guards(lex_tree)
        return lex_tree

    def tree(self) -> dict[Path, Entry]:
        """Return the merged, guard-checked canonical destination tree."""
        self._lexical_tree()
        surface = frozenset(e.dst for e in _surface_entries(_canonical_path))
        tree = merge_entries(self.entries())
        check_surface(tree, surface, tool="sandbox-exec")
        check_guards(tree)

        seen_syms: dict[Path, Path] = {}
        for sym in self._pinned_symlinks():
            dst = Path(seatbelt_lexpath(sym.dst))
            src = _canonical_path(sym.src)
            prior_src = seen_syms.get(dst)
            if prior_src is not None and prior_src != src:
                raise MountConflict(
                    f"conflicting PinnedSymlink targets at {dst}: {prior_src} vs {src}"
                )
            for other_sym_dst in seen_syms:
                if other_sym_dst != dst and (
                    dst.is_relative_to(other_sym_dst)
                    or other_sym_dst.is_relative_to(dst)
                ):
                    raise MountConflict(
                        f"nested PinnedSymlink destinations: {dst} and {other_sym_dst}"
                    )
            seen_syms[dst] = src

            if dst in tree:
                raise MountConflict(
                    f"PinnedSymlink destination {sym.dst} collides with mount at {dst}"
                )
            for other_dst, other in tree.items():
                if other_dst.is_relative_to(dst) and (not other.guard or other.fixed):
                    raise GuardViolation(
                        f"mount at {other_dst} cannot sit below PinnedSymlink"
                        f" guard at {sym.dst}"
                    )
            for parent in dst.parents:
                above = tree.get(parent)
                if above is not None:
                    if above.op is Op.SEAL:
                        raise GuardViolation(
                            f"PinnedSymlink {sym.dst} lies directly inside sealed"
                            f" directory {parent}"
                        )
                    break
        return tree

    def resolve(self) -> list[Mount]:
        """Validate host paths and return resolved mounts in tree depth order."""
        lex_tree = self._lexical_tree()
        check_symlink_free(lex_tree, tool="sandbox-exec")
        for e in lex_tree.values():
            if e.op in (Op.RW, Op.SEAL, Op.WRITE_REGEX):
                _reject_host_symlinks(e.dst, include_self=True)
        for sym in self._pinned_symlinks():
            _reject_host_symlinks(
                through_darwin_system_symlink(sym.dst), include_self=False
            )

        tree = self.tree()
        surface = frozenset(e.dst for e in _surface_entries(_canonical_path))
        mounts: list[Mount] = []
        seal_ro: list[Mount] = []
        for e in sorted(tree.values(), key=lambda item: item.dst.parts):
            if e.dst in surface:
                continue
            m = mount_for(e)
            if m is None:
                continue
            mounts.append(m)
            if e.op is Op.SEAL:
                seal_ro.append(Mount("seal-ro", e.dst, guard=True))
        return [*mounts, *seal_ro]

    def _published(self) -> list[Mount]:
        return [*_system_mounts(), *self.resolve()]

    def exposed_path(
        self,
        paths: Iterable[Path],
        *,
        allowed_sources: Iterable[Path] = (),
    ) -> tuple[Path, Path] | None:
        """Return the first protected path left readable in the box, if any."""
        raw_paths = tuple(paths)
        if not raw_paths:
            return None
        mounts = self._published()
        canon_allowed = tuple(_canonical_path(p) for p in allowed_sources)
        for orig in raw_paths:
            hit = first_exposure(
                mounts, (_canonical_path(orig),), canon_allowed, casefold=True
            )
            if hit is not None:
                return hit[0], orig
        return None

    def sealed_exposure(self) -> tuple[Path, Path] | None:
        """Return the first mount source republishing a sealed directory."""
        seals = [b for b in self.binds if isinstance(b, Seal)]
        if not seals:
            return None
        mounts = self._published()
        for seal in seals:
            canon_allow = tuple(_canonical_path(p) for p in seal.allow)
            hit = first_exposure(
                mounts, (_canonical_path(seal.path),), canon_allow, casefold=True
            )
            if hit is not None:
                return hit[0], seal.path
        return None

    def assert_runtime_visible(self, runtime_dir: Path) -> None:
        """Verify that `runtime_dir` survived resolution as a read-only mount."""
        expected = _canonical_path(runtime_dir)
        if not any(m.dst == expected and m.op == "ro" for m in self.resolve()):
            raise ValueError(
                f"spec has egress backends but does not bind the runtime dir"
                f" {runtime_dir} ro -- the launcher would have no egress"
                " control data"
            )

    def render_sbpl(self) -> str:
        """Compile the validated policy into a complete `(version 1)` SBPL text."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root missing: {self.root}")
        hit = self.sealed_exposure()
        if hit is not None:
            src, path = hit
            raise ValueError(
                f"mount {src} would republish the sealed directory {path} under"
                " a second name, where the seal does not reach"
            )
        mounts = [m for m in self.resolve() if m.op != "seal-ro"]
        symlinks = self._pinned_symlinks()
        for sym in symlinks:
            if not sym.src.exists():
                raise FileNotFoundError(f"PinnedSymlink source missing: {sym.src}")
            if not sym.dst.is_symlink() or seatbelt_path(sym.dst) != seatbelt_path(
                sym.src
            ):
                raise ValueError(
                    f"PinnedSymlink destination {sym.dst} must be staged as a"
                    f" symlink to {sym.src} before rendering SBPL"
                )

        lines: list[str] = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow file-map-executable)",
            "(allow pseudo-tty)",
            "(deny process-info*)",
            "(allow process-info* (target same-sandbox))",
            "(allow signal (target same-sandbox))",
            "(allow ipc-posix-sem)",
            self._render_sysctls(),
            '(allow sysctl-write (sysctl-name "kern.grade_cputype"))',
            *self._render_mach(),
            *self._render_surface_and_ancestors(mounts, symlinks),
            *self._render_policy_mounts(mounts, symlinks),
            *self._render_anti_rename_pins(mounts, symlinks),
            *self._render_network(),
        ]
        return "\n".join(line for line in lines if line) + "\n"

    def _render_sysctls(self) -> str:
        names = " ".join(_sbpl_quote(n) for n in _SYSCTL_NAMES)
        return (
            f"(allow sysctl-read (sysctl-name {names})"
            ' (sysctl-name-prefix "hw.optional.")'
            ' (sysctl-name-prefix "hw.perflevel")'
            ' (sysctl-name-prefix "kern.proc.pid."))'
        )

    def _render_mach(self) -> list[str]:
        services = ["com.apple.system.opendirectoryd.libinfo"]
        if "codex-cfprefs" in self.mach_profiles:
            services.extend(("com.apple.cfprefsd.agent", "com.apple.cfprefsd.daemon"))
        if not self.unshare_net:
            services.extend(("com.apple.trustd.agent", "com.apple.dnssd.service"))
        quoted = " ".join(f"(global-name {_sbpl_quote(s)})" for s in services)
        out = [f"(allow mach-lookup {quoted})"]
        if "codex-cfprefs" in self.mach_profiles:
            out.append(
                '(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))'
            )
            out.append(
                '(allow user-preference-read (preference-domain "com.openai.codex"))'
            )
        return out

    def _render_surface_and_ancestors(
        self,
        mounts: list[Mount],
        symlinks: tuple[PinnedSymlink, ...],
    ) -> list[str]:
        metadata_literals: set[str] = {
            "/",
            "/etc",
            str(Path("/") / "tmp"),
            "/var",
            "/private",
            "/System",
            "/System/Volumes",
            "/System/Volumes/Data",
            "/System/Volumes/Data/private",
            "/System/Volumes/Data/Users",
            "/System/Volumes/Preboot",
        }
        raw_bind_paths = [
            b.path
            for b in self.binds
            if isinstance(b, Bind | WriteRegex)
            and (not getattr(b, "optional", False) or b.path.exists())
        ]
        all_paths: list[Path] = [
            *_active_system_subpaths(),
            *_DARWIN_SYSTEM_LITERALS,
            *(m.dst for m in mounts),
            *raw_bind_paths,
            *(sym.src for sym in symlinks),
            *(sym.dst for sym in symlinks),
            *self.unix_sockets,
        ]
        for p in all_paths:
            for anc in _canonical_path(p).parents:
                metadata_literals.add(str(anc))
            lex_p = through_darwin_system_symlink(p)
            for anc in lex_p.parents:
                metadata_literals.add(seatbelt_lexpath(anc))
            if lex_p.is_symlink():
                metadata_literals.add(seatbelt_lexpath(lex_p))
        for sym in symlinks:
            metadata_literals.add(seatbelt_lexpath(sym.dst))
        for sock in self.unix_sockets:
            metadata_literals.add(seatbelt_lexpath(sock))

        lit_paths: list[str] = []
        for p in _DARWIN_SYSTEM_LITERALS:
            for candidate in (_strip_firmlink(str(p)), seatbelt_path(p)):
                if candidate not in lit_paths:
                    lit_paths.append(candidate)
        if not self.unshare_net:
            mdns = "/private/var/run/mDNSResponder"
            if mdns not in lit_paths:
                lit_paths.append(mdns)
            for anc in Path(mdns).parents:
                metadata_literals.add(str(anc))

        meta_args = " ".join(
            f"(literal {_sbpl_quote(lit)})" for lit in sorted(metadata_literals)
        )
        sub_args = " ".join(
            f"(subpath {_sbpl_quote(seatbelt_path(p))})"
            for p in _active_system_subpaths()
        )
        lit_args = " ".join(f"(literal {_sbpl_quote(lit)})" for lit in lit_paths)
        tty_lits = " ".join(
            f"(literal {_sbpl_quote(t)})"
            for t in (
                "/dev/null",
                "/dev/zero",
                "/dev/tty",
                "/dev/ptmx",
                *self.controlling_ttys,
            )
        )
        return [
            f'(allow file-read-metadata file-test-existence {meta_args} (regex #"^/dev/ttys[0-9]+$"))',
            f"(allow file-read* file-test-existence {sub_args} {lit_args})",
            f"(allow file-read* file-write* file-ioctl {tty_lits})",
            r'(allow file-read* file-write* file-ioctl (require-all (regex #"^/dev/ttys[0-9]+$") (extension "com.apple.sandbox.pty")))',
        ]

    def _render_policy_mounts(
        self,
        mounts: list[Mount],
        symlinks: tuple[PinnedSymlink, ...],
    ) -> list[str]:
        ordered = sorted(
            mounts,
            key=lambda m: (len(_canonical_path(m.dst).parts), seatbelt_path(m.dst)),
        )
        deeper_paths = [
            *(_canonical_path(m.dst) for m in ordered),
            *(_canonical_path(sym.src) for sym in symlinks),
            *(Path(seatbelt_lexpath(sym.dst)) for sym in symlinks),
        ]
        out: list[str] = []
        for m in ordered:
            sb = seatbelt_path(m.dst)
            sb_path = Path(sb)
            quoted = _sbpl_quote(sb)
            match m.op:
                case "tmpfs":
                    # `mount_for(Op.SEAL)` produces `Mount("tmpfs", ..., guard=True)`.
                    # Regular `self.tmpfs` is not passed to `SeatbeltSandbox`.
                    seal_meta = {sb}
                    for target in deeper_paths:
                        if target != sb_path and target.is_relative_to(sb_path):
                            for anc in target.parents:
                                if anc.is_relative_to(sb_path):
                                    seal_meta.add(str(anc))
                    meta_lits = " ".join(
                        f"(literal {_sbpl_quote(p)})" for p in sorted(seal_meta)
                    )
                    out.append(
                        f"(deny file-read* file-test-existence file-write* (subpath {quoted}))"
                    )
                    out.append(
                        f"(allow file-read-metadata file-test-existence {meta_lits})"
                    )
                case "rw":
                    kind = (
                        "subpath"
                        if (m.src is None or not m.src.exists() or m.src.is_dir())
                        else "literal"
                    )
                    out.append(
                        f"(allow file-read* file-test-existence file-write* ({kind} {quoted}))"
                    )
                case "ro":
                    is_dir = m.src is not None and m.src.is_dir()
                    kind = "subpath" if is_dir else "literal"
                    out.append(
                        f"(allow file-read* file-test-existence ({kind} {quoted}))"
                    )
                    out.append(f"(deny file-write* ({kind} {quoted}))")
                case "write-regex":
                    out.append(
                        f"(allow file-read* file-test-existence (subpath {quoted}))"
                    )
                    out.append(f"(deny file-write* (subpath {quoted}))")
                    for suffix in m.suffixes:
                        pattern = _sbpl_regex(f"^{_ere_escape(sb)}/({suffix[1:]})$")
                        out.append(
                            f"(allow file-write* (require-all (vnode-type REGULAR-FILE) (regex {pattern})))"
                        )
                        out.append(f"(deny file-write-unlink (regex {pattern}))")
        for sym in symlinks:
            sym_quoted = _sbpl_quote(seatbelt_lexpath(sym.dst))
            out.append(f"(deny file-write* (literal {sym_quoted}))")
        return out

    def _render_anti_rename_pins(
        self,
        mounts: list[Mount],
        symlinks: tuple[PinnedSymlink, ...],
    ) -> list[str]:
        rw_roots = [
            _canonical_path(m.dst) for m in mounts if m.op in ("rw", "write-regex")
        ]
        pinned: set[str] = {seatbelt_path(self.root)}
        protected_targets: list[Path] = []
        for m in mounts:
            sb = _canonical_path(m.dst)
            if m.op in ("tmpfs", "rw", "write-regex"):
                pinned.add(str(sb))
                protected_targets.append(sb)
            elif m.op == "ro":
                protected_targets.append(sb)
        for sym in symlinks:
            sym_lex = Path(seatbelt_lexpath(sym.dst))
            pinned.add(str(sym_lex))
            protected_targets.append(sym_lex)

        for target in protected_targets:
            for rw in rw_roots:
                if target == rw or not target.is_relative_to(rw):
                    continue
                cursor = target.parent
                while cursor != rw and cursor.is_relative_to(rw):
                    pinned.add(str(cursor))
                    cursor = cursor.parent
                pinned.add(str(rw))

        literals = " ".join(f"(literal {_sbpl_quote(p)})" for p in sorted(pinned))
        return [f"(deny file-write-unlink file-write-create {literals})"]

    def _render_network(self) -> list[str]:
        if not self.unshare_net:
            return ["(allow network*)"]
        out = [
            "(deny network*)",
            "(allow system-socket (socket-domain AF_UNIX))",
        ]
        for port in self.tcp_ports:
            out.append(
                f"(allow network-outbound (remote tcp {_sbpl_quote(f'localhost:{port}')}))"
            )
        for sock in self.unix_sockets:
            sb_sock = seatbelt_lexpath(sock)
            out.append(
                f"(allow network-outbound (remote unix-socket (path-literal {_sbpl_quote(sb_sock)})))"
            )
        if self.allow_local_loopback:
            out.append('(allow network-bind (local tcp "localhost:*"))')
            out.append('(allow network-inbound (local tcp "localhost:*"))')
            out.append('(allow network-outbound (remote tcp "localhost:*"))')
        return out

    def wrapper(self) -> list[str]:
        """Return the `/usr/bin/env` and `/usr/bin/sandbox-exec` argv prefix."""
        sbpl = self.render_sbpl()
        # Deduplicate repeated environment keys keeping the final value, matching
        # bwrap's ordered `--setenv` semantics (`BoxSpec.with_env`).
        dedup_env: dict[str, str] = {}
        for k, v in self.env:
            if not k or k.startswith("-") or "=" in k or "\x00" in k or "\x00" in v:
                raise ValueError(f"invalid environment entry for key {k!r}")
            dedup_env[k] = v
        env_args = [f"{k}={v}" for k, v in dedup_env.items()]
        if self.profile_path is not None:
            fd = os.open(
                self.profile_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(sbpl.encode("utf-8"))
            sb_args = ["-f", str(self.profile_path)]
        else:
            sb_args = ["-p", sbpl]
        return [
            "/usr/bin/env",
            "-i",
            "-C",
            seatbelt_path(self.root),
            "--",
            *env_args,
            "/usr/bin/sandbox-exec",
            *sb_args,
        ]
