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
* Every path emitted into SBPL is canonicalized to XNU's `F_GETPATH` spelling
  (resolving `/tmp`, `/var`, `/etc` -> `/private/...` and mapping APFS
  `/System/Volumes/Data` firmlinks back to their synthetic root paths).
* Policy mounts are emitted in canonical path-depth order: `RO` mounts emit an
  explicit `(deny file-write*)` so pins inside writable trees cannot be edited,
  and `Seal` directories deny read, existence-testing, and write access while
  permitting `file-read-metadata` on the sealed root itself so `realpath(3)`
  succeeds inside allowed holes.
* Every writable directory above a pin, seal, or `PinnedSymlink` receives a
  final `(deny file-write-unlink file-write-create (literal ...))` rule, giving
  Seatbelt the rename-and-recreate protection that VFS mount points provide on
  Linux.
* Network isolation (`unshare_net=True`) denies `network*` and opens only the
  activated host-proxy TCP ports and Unix domain sockets (plus localhost TCP
  when `allow_local_loopback=True`). Neither `com.apple.SecurityServer` nor
  `com.apple.securityd.xpc` is ever permitted.
"""

from __future__ import annotations

import fcntl
import functools
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .policy_tree import (
    Bind,
    BindOver,
    BindSpec,
    Entry,
    Mount,
    Op,
    Overlay,
    PinnedSymlink,
    Seal,
    SymlinkDestination,
    WriteRegex,
    check_guards,
    check_surface,
    check_symlink_free,
    entry_for,
    first_exposure,
    merge_entries,
    mount_for,
    resolved_path,
)

__all__ = [
    "SeatbeltSandbox",
    "darwin_system_ro_roots",
    "seatbelt_lexpath",
    "seatbelt_path",
    "through_darwin_system_symlink",
]

_FIRMLINKS_FILE = Path("/usr/share/firmlinks")
_DEFAULT_FIRMLINKS: tuple[tuple[str, str], ...] = (
    ("/Users", "Users"),
    ("/Library", "Library"),
    ("/Applications", "Applications"),
    ("/System/Library/Caches", "System/Library/Caches"),
    ("/private", "private"),
    ("/opt", "opt"),
    ("/AppleInternal", "AppleInternal"),
)

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


@functools.lru_cache(maxsize=1)
def _firmlinks() -> tuple[tuple[str, str], ...]:
    """Return `(/System/Volumes/Data/<rel>, <synthetic_root>)` pairs, longest first."""
    raw = list(_DEFAULT_FIRMLINKS)
    try:
        lines = _FIRMLINKS_FILE.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0].startswith("/") and parts[1]:
            pair = (parts[0], parts[1])
            if pair not in raw:
                raw.append(pair)
    mapped = [(f"/System/Volumes/Data/{rel.lstrip('/')}", root) for root, rel in raw]
    return tuple(sorted(mapped, key=lambda item: -len(item[0])))


def _strip_firmlink(raw: str) -> str:
    for data_prefix, synthetic in _firmlinks():
        if raw == data_prefix:
            return synthetic
        if raw.startswith(data_prefix + "/"):
            suffix = raw[len(data_prefix) :]
            return suffix if synthetic == "/" else synthetic + suffix
    return raw


def _f_getpath(path: Path) -> str | None:
    getpath = getattr(fcntl, "F_GETPATH", None)
    if getpath is None:
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        buf = fcntl.fcntl(fd, getpath, bytes(1024))
    except OSError:
        return None
    finally:
        os.close(fd)
    return buf.split(b"\x00", 1)[0].decode("utf-8", "surrogateescape")


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
    on the deepest existing prefix when running on Darwin, and maps APFS
    `/System/Volumes/Data` firmlinks back to their synthetic root paths.
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


def seatbelt_lexpath(path: Path) -> str:
    """Canonicalize `path`'s parent while preserving a symlink at `path.name`.

    Operations on a symlink vnode itself (`unlink`, `rename` over a link,
    `readlink`) check the symlink's own directory path rather than its target.
    """
    _check_clean_path(str(path))
    if path == path.parent:
        return "/"
    parent = seatbelt_path(path.parent)
    return f"/{path.name}" if parent == "/" else f"{parent}/{path.name}"


def _active_system_subpaths() -> tuple[Path, ...]:
    present_dev = tuple(p for p in _DARWIN_DEVELOPER_SUBPATHS if p.exists())
    present_brew = tuple(p for p in _DARWIN_HOMEBREW_SUBPATHS if p.exists())
    return (*_DARWIN_SYSTEM_SUBPATHS, *present_dev, *present_brew)


def darwin_system_ro_roots() -> frozenset[Path]:
    """Return identity-mounted Darwin system roots that need no duplicate bind."""
    return frozenset(_active_system_subpaths())


def _surface_entries() -> tuple[Entry, ...]:
    return tuple(Entry(p, Op.RO, p, fixed=True) for p in _active_system_subpaths())


def _system_mounts() -> list[Mount]:
    return [Mount("ro", p, p) for p in _active_system_subpaths() if p.exists()]


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
    profile_path: Path | None = None

    def __post_init__(self) -> None:
        if self.root_tmpfs < 0:
            raise ValueError(f"root_tmpfs must not be negative: {self.root_tmpfs}")
        if self.root_tmpfs > 0:
            raise ValueError(
                f"root_tmpfs ({self.root_tmpfs} bytes at {self.root}) is not"
                " supported on macOS Seatbelt"
            )

    def _pinned_symlinks(self) -> tuple[PinnedSymlink, ...]:
        return tuple(b for b in self.binds if isinstance(b, PinnedSymlink))

    def _write_regexes(self) -> dict[Path, list[str]]:
        out: dict[Path, list[str]] = {}
        for b in self.binds:
            if isinstance(b, WriteRegex):
                dst = through_darwin_system_symlink(b.path)
                out.setdefault(dst, []).extend(b.suffixes)
        return out

    def entries(self) -> list[Entry]:
        """Return every declared mount as a normalized destination-tree entry."""
        out = list(_surface_entries())
        root_dst = through_darwin_system_symlink(self.root)
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
                case WriteRegex(path=p, guard=guard):
                    dst = through_darwin_system_symlink(p)
                    out.append(Entry(dst, Op.RO, dst, guard=guard))
                case PinnedSymlink(src=src):
                    src_dst = through_darwin_system_symlink(src)
                    out.append(Entry(src_dst, Op.RO, src_dst, guard=True))
                case Bind(path=p, mode=mode, optional=optional, guard=guard):
                    dst = through_darwin_system_symlink(p)
                    out.append(
                        Entry(dst, Op(mode.value), dst, guard=guard, optional=optional)
                    )
                case Seal(path=p, allow_missing=allow_missing):
                    dst = through_darwin_system_symlink(p)
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

    def tree(self) -> dict[Path, Entry]:
        """Return the merged, guard-checked destination tree."""
        surface = frozenset(e.dst for e in _surface_entries())
        tree = merge_entries(self.entries())
        check_surface(tree, surface, tool="sandbox-exec")
        check_guards(tree)
        for sym in self._pinned_symlinks():
            dst = through_darwin_system_symlink(sym.dst)
            if dst in tree:
                raise ValueError(
                    f"PinnedSymlink destination {sym.dst} collides with mount at {dst}"
                )
            for other_dst, other in tree.items():
                if other_dst.is_relative_to(dst) and not other.guard:
                    raise ValueError(
                        f"plain mount at {other_dst} lies below PinnedSymlink"
                        f" guard at {sym.dst}"
                    )
        return tree

    def resolve(self) -> list[Mount]:
        """Validate host paths and return resolved mounts in tree depth order."""
        tree = self.tree()
        check_symlink_free(tree, tool="sandbox-exec")
        for e in tree.values():
            if e.op in (Op.RW, Op.SEAL) and e.dst.is_symlink():
                raise SymlinkDestination(
                    f"mount destination {e.dst} is a symlink; sandbox-exec"
                    " requires the resolved directory path for writable and"
                    " sealed roots"
                )
        surface = frozenset(e.dst for e in _surface_entries())
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
        paths = tuple(paths)
        if not paths:
            return None
        return first_exposure(self._published(), paths, allowed_sources)

    def sealed_exposure(self) -> tuple[Path, Path] | None:
        """Return the first mount source republishing a sealed directory."""
        seals = [b for b in self.binds if isinstance(b, Seal)]
        if not seals:
            return None
        mounts = self._published()
        for seal in seals:
            hit = first_exposure(mounts, (seal.path,), seal.allow)
            if hit is not None:
                return hit
        return None

    def assert_runtime_visible(self, runtime_dir: Path) -> None:
        """Verify that `runtime_dir` survived resolution as a read-only mount."""
        expected = through_darwin_system_symlink(runtime_dir)
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

        lines: list[str] = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow file-map-executable)",
            "(allow process-info* (target same-sandbox))",
            "(allow signal (target same-sandbox))",
            "(allow ipc-posix-sem)",
            self._render_sysctls(),
            '(allow sysctl-write (sysctl-name "kern.grade_cputype"))',
            *self._render_mach(),
            *self._render_surface_and_ancestors(mounts, symlinks),
            *self._render_policy_mounts(mounts),
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
            services.append("com.apple.trustd.agent")
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
        all_paths: list[Path] = [
            *_active_system_subpaths(),
            *_DARWIN_SYSTEM_LITERALS,
            *(m.dst for m in mounts),
            *(sym.src for sym in symlinks),
            *(sym.dst for sym in symlinks),
        ]
        for p in all_paths:
            for anc in Path(seatbelt_path(p)).parents:
                metadata_literals.add(str(anc))
            for anc in p.parents:
                metadata_literals.add(_strip_firmlink(str(anc)))
            if p.is_symlink():
                metadata_literals.add(seatbelt_lexpath(p))

        meta_args = " ".join(
            f"(literal {_sbpl_quote(lit)})" for lit in sorted(metadata_literals)
        )
        sub_args = " ".join(
            f"(subpath {_sbpl_quote(seatbelt_path(p))})"
            for p in _active_system_subpaths()
        )
        lit_args = " ".join(
            f"(literal {_sbpl_quote(_strip_firmlink(str(p)))})"
            for p in _DARWIN_SYSTEM_LITERALS
        )
        return [
            f"(allow file-read-metadata file-test-existence {meta_args})",
            f"(allow file-read* file-test-existence {sub_args} {lit_args})",
            '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty"))',
            r'(allow file-read* file-write* file-ioctl (regex #"^/dev/ttys[0-9]+$"))',
        ]

    def _render_policy_mounts(self, mounts: list[Mount]) -> list[str]:
        write_regexes = self._write_regexes()
        ordered = sorted(
            mounts,
            key=lambda m: (len(Path(seatbelt_path(m.dst)).parts), seatbelt_path(m.dst)),
        )
        out: list[str] = []
        for m in ordered:
            sb = seatbelt_path(m.dst)
            quoted = _sbpl_quote(sb)
            match m.op:
                case "tmpfs":
                    # `mount_for(Op.SEAL)` produces `Mount("tmpfs", ..., guard=True)`.
                    # Regular `self.tmpfs` is not passed to `SeatbeltSandbox`.
                    out.append(
                        f"(deny file-read* file-test-existence file-write* (subpath {quoted}))"
                    )
                    out.append(
                        f"(allow file-read-metadata file-test-existence (literal {quoted}))"
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
                    for suffix in write_regexes.get(m.dst, ()):
                        pattern = f"^{_ere_escape(sb)}({suffix})$"
                        out.append(
                            f"(allow file-write* (regex {_sbpl_regex(pattern)}))"
                        )
        return out

    def _render_anti_rename_pins(
        self,
        mounts: list[Mount],
        symlinks: tuple[PinnedSymlink, ...],
    ) -> list[str]:
        rw_roots = [Path(seatbelt_path(m.dst)) for m in mounts if m.op == "rw"]
        pinned: set[str] = {seatbelt_path(self.root)}
        protected_targets: list[Path] = []
        for m in mounts:
            sb = Path(seatbelt_path(m.dst))
            if m.op == "tmpfs":  # Seal
                pinned.add(str(sb))
                protected_targets.append(sb)
            elif m.op == "ro":
                protected_targets.append(sb)
            elif m.op == "rw" and m.guard:
                # Guarded RW holes inside Seals (such as a linked worktree's
                # private gitdir) must not be renamed or deleted either.
                pinned.add(str(sb))
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
            out.append('(allow network-bind (local ip "localhost:*"))')
            out.append('(allow network-inbound (local ip "localhost:*"))')
            out.append('(allow network-outbound (remote ip "localhost:*"))')
        return out

    def wrapper(self) -> list[str]:
        """Return the `/usr/bin/env` and `/usr/bin/sandbox-exec` argv prefix."""
        sbpl = self.render_sbpl()
        # Deduplicate repeated environment keys keeping the final value, matching
        # bwrap's ordered `--setenv` semantics (`BoxSpec.with_env`).
        dedup_env: dict[str, str] = {}
        for k, v in self.env:
            if "=" in k or "\x00" in k or "\x00" in v:
                raise ValueError(f"invalid environment entry for key {k!r}")
            dedup_env[k] = v
        env_args = [f"{k}={v}" for k, v in dedup_env.items()]
        if self.profile_path is not None:
            fd = os.open(
                self.profile_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
            )
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, sbpl.encode("utf-8"))
            finally:
                os.close(fd)
            sb_args = ["-f", str(self.profile_path)]
        else:
            sb_args = ["-p", sbpl]
        return [
            "/usr/bin/env",
            "-i",
            "-C",
            seatbelt_path(self.root),
            *env_args,
            "/usr/bin/sandbox-exec",
            *sb_args,
        ]
