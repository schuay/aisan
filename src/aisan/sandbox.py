# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The bind model: an ordered mount policy, rendered as a bubblewrap argv.

The mechanism is per-call confinement. With an in-process agent harness the
model's whole capability surface is its tools, so wrapping tool execution
confines the agent without a process boundary -- and the same wrapper serves a
box whose payload is one command. Each call runs inside a fresh bubblewrap mount
namespace (the job worktree rw at its real absolute path, declared dependencies
ro, everything else -- $HOME, other worktrees, the main checkout -- absent, not
merely denied), under a systemd transient scope for cgroup limits.

What the box may touch is one ordered list of binds, and the rule is bwrap's
own: a later mount wins where two overlap. There is no second precedence rule
layered on top -- no read-vs-write asymmetry, no field whose name encodes when
it is emitted. Reading the list top to bottom is reading the policy, which is
the property that makes a profile auditable.

Known give-ups of this tier, documented rather than papered over:
- Network is the host's unless `unshare_net` is set. With it set the box has
  no route off the machine at all; host-side proxies then reach it only through
  a bind-mounted UNIX socket, which is what the Vertex and RBE proxies do.
  Interactive launchers default to that isolation and expose explicit `--net`
  opt-in backed by authenticated host-loopback proxies.
- No disk quota on the rw worktree (project quotas need root); only the tmpfs
  mounts are size-capped.
- /tmp is a fresh tmpfs per call: scratch there does not survive to the next
  shell call. Durable scratch needs an explicit persistent bind.
"""

from __future__ import annotations

import enum
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The fixed system surface every sandboxed command sees. /usr and /etc ro; the
# usual merged-usr symlinks recreated instead of bound so nothing else from /
# leaks in. No /opt, no /srv, no /home beyond the tmpfs + explicit binds.
_SYSTEM_ARGS = (
    "--ro-bind", "/usr", "/usr",
    "--symlink", "usr/bin", "/bin",
    "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib64", "/lib64",
    "--symlink", "usr/sbin", "/sbin",
    "--ro-bind", "/etc", "/etc",
    "--proc", "/proc",
    "--dev", "/dev",
)  # fmt: skip

# The host journal's stdout stream socket. `bash -lc` sources /etc/profile.d,
# and on a systemd host an /etc/profile.d snippet may pipe shell startup through
# `systemd-cat`, which connects here; /run is unbound, so absent this bind the
# connect fails and prefixes every command with two benign "Failed to create
# stream fd" lines that are then re-fed to the model on each shell result.
_JOURNAL_STDOUT_SOCK = Path("/run/systemd/journal/stdout")

_ISOLATION_ARGS = (
    "--unshare-pid",
    "--unshare-ipc",
    "--unshare-uts",
    "--cap-drop",
    "ALL",
    "--die-with-parent",
)


class Mode(enum.Enum):
    """Which way a bind is mounted. Nothing else distinguishes two binds: where
    they land relative to each other is their position in the list."""

    RO = "ro"
    RW = "rw"


RO = Mode.RO
RW = Mode.RW


@dataclass(frozen=True)
class Bind:
    """Mount `path` into the box at its own absolute path.

    A later Bind of an overlapping path wins, which is how a pin, a hole through
    it, and a pin back inside that hole are all written: as three Binds in that
    order. The mode alone carries no precedence -- an RO bind does not outrank an
    RW one, it just comes after it or does not.

    `optional` is the one property the ordering cannot express, and it is
    deliberately explicit rather than implied by what a bind is for. A dep
    symlink target that vanished is a bind to skip; a guard that vanished is a
    profile to refuse, because the path it was pinning stays writable under
    whatever it was pinned on top of. Default False: a bind whose source is
    missing fails the profile unless the caller has said it is safe to drop.
    """

    path: Path
    mode: Mode
    optional: bool = False


@dataclass(frozen=True)
class BindOver:
    """Mount `src` at `dst` -- the one bind that is not src -> the same path.

    Everything else answers "let the box see this"; this answers "let the box see
    THIS where it expects THAT", which is how a per-job /etc/hosts or a build
    config reaches a checkout the job does not own. Mounted ro: the freedom this
    adds is over the source, not over what the box may write.

    Never optional. Silently skipping it leaves the box reading the ORIGINAL
    file, which is the misconfiguration it exists to prevent.
    """

    src: Path
    dst: Path


@dataclass(frozen=True)
class Overlay:
    """Show `path` WARM but discard what the box writes to it: an overlayfs whose
    lower layer is the host directory and whose upper layer is a per-box tmpfs
    (bwrap --tmp-overlay). Reads come from the host copy, writes die with the box.

    The shape this exists for is a shared tool cache the box must be able to
    write to but must never actually change -- vpython's, which is 4 GB of
    interpreters and ~21k Python files that every `git cl` execution runs. It
    needs a lock file even to READ, so a ro bind fails ("failed to acquire read
    lock ... read-only file system") and an absent one makes vpython try to
    rebuild the venv, which with no network HANGS.

    Not a general substitute for RO: an overlay is writable from inside, so
    anything the box must not change EVEN LOCALLY (a pinned git config, a
    credential) is still a Bind(..., RO) later in the list. Never optional -- a
    skipped overlay is the hang, and a hang is the worst way to learn about a
    bad bind.
    """

    path: Path


@dataclass(frozen=True)
class Seal:
    """Replace `path` with an empty directory nothing can be created in.

    An RO bind of a directory makes its EXISTING entries read-only; it does not
    stop the box creating new ones... but it also cannot hide what is already
    there. A seal does both: an empty tmpfs over the directory, remounted ro once
    the rest of the list has been mounted.

    This is the .git/worktrees case, and it is why pins are not enough on their
    own. Pinning the sibling worktree configs that exist at assembly time is a
    SNAPSHOT: a profile outlives the scan, so a job could write a config.worktree
    into a sibling created after it, or into a directory of its own invention,
    and steer host-side git from there. Sealing removes the whole question --
    siblings are not read-only in the box, they are absent -- while a later
    Bind(<own dir>, RW) punches this job's own dir back through.

    The ro remount is deferred to the end of the list on purpose: bwrap cannot
    create a mountpoint inside an already-read-only tmpfs, so sealing atomically
    would make every hole through the seal fail with "Can't mkdir: Read-only file
    system" (verified against bwrap 0.11.2).
    """

    path: Path


BindSpec = Bind | BindOver | Overlay | Seal


@dataclass(frozen=True)
class EnsurePath:
    """A host path a profile needs present before its binds resolve, created
    empty if absent and removed again if the box created it.

    Not a mount. The .git guard pins bind REAL host files, so that an in-box
    agent cannot make host-side git read a config or hook it planted -- and on a
    fresh checkout some of those files are absent, while a missing bind source
    fails the profile. Creating them is a host mutation, and doing it inside the
    pure preset meant even `aisan explain`, a dry run, wrote into the shared
    .git. Declared here instead, the Box creates them on the run path and undoes
    them on the inspector's staged path, so a review leaves the host untouched.

    `is_dir` distinguishes a file (touched, unlinked) from a directory (mkdir'd,
    rmdir'd only if still empty); the producer knows which, and the two clean up
    differently.
    """

    path: Path
    is_dir: bool


@dataclass(frozen=True)
class Mount:
    """One resolved mount operation, in the order bwrap will apply it.

    The resolved list is the single representation of what a profile mounts: the
    leak check runs over it, the argv is a formatting of it, and the inspector
    displays it. A check against one representation and a display of another is
    how the two drift apart without anything failing.
    """

    op: str  # ro | rw | tmpfs | overlay | seal-ro
    dst: Path
    src: Path | None = None
    size: int = 0

    @property
    def covers(self) -> bool:
        """Whether this op mounts something over `dst`, as opposed to modifying a
        mount already there. Only a covering op can shadow what came before it."""
        return self.op != "seal-ro"


def _bindable(p: Path) -> bool:
    """Whether an OPTIONAL bind source can be mounted, i.e. whether we can stat
    it at all -- not merely whether it exists.

    `Path.exists()` is not enough: it swallows ENOENT/ENOTDIR/EBADF/ELOOP and
    re-raises everything else, so a path we cannot reach for any other reason
    propagates an OSError out of wrapper() instead of being skipped. The case
    that bit us is a credential-gated network mount (a /google path with an
    expired ticket) raising ENOKEY: the caller had already decided that bind was
    optional, but sandbox assembly blew up and took the whole job with it.
    Unreachable and absent are the same thing for an optional bind -- bwrap
    cannot mount either -- so treat any stat failure as "skip".

    Only for optional sources. A mandatory one must keep raising: skipping it
    silently drops a guard while whatever it was pinned over stays writable.
    """
    try:
        p.stat()
    except OSError:
        return False
    return True


# The ro roots _SYSTEM_ARGS already binds (/usr, /etc). An RO bind that resolves
# to one of these is already mounted (and mounted early, in the safe order), so
# emitting it again is pure waste -- and, when it lands after a tmpfs it covers
# (the motivating bug: a launcher-resolved interpreter root of /usr re-emitted after
# the $HOME tmpfs), a leak. Parsed from _SYSTEM_ARGS so it cannot drift.
_SYSTEM_RO_ROOTS = frozenset(
    Path(_SYSTEM_ARGS[i + 2])
    for i in range(len(_SYSTEM_ARGS) - 2)
    if _SYSTEM_ARGS[i] == "--ro-bind"
)


def _system_mounts() -> list[Mount]:
    """The fixed system surface as mounts, for checks that must reason about
    what the box gets rather than only about what the caller asked for.

    Read from `_SYSTEM_RO_ROOTS` (and so from `_SYSTEM_ARGS`) at call time, for
    the same reason that set is parsed rather than written out: two spellings of
    the system surface drift, and a credential check reading the stale one would
    pass a box that mounts the credential.
    """
    return [Mount("ro", p, p) for p in sorted(_SYSTEM_RO_ROOTS)]


def _resolved(p: Path) -> Path:
    """`p` canonicalised, or `p` itself when the host cannot say. Comparisons
    here are between real locations: a symlinked home and its target are one
    directory, and treating them as two is how a check misses."""
    try:
        return p.resolve()
    except OSError:
        return p


def paths_overlap(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are the same path or one is inside the other.

    The overlap decision, once, for every guard that asks whether a mount and a
    credential touch. EITHER direction: a mount above a credential store
    publishes the store whole, and a mount of one file inside it publishes that
    file -- which, for a store whose point is the token it holds, is the same
    answer. A one-way containment test reads as the stricter rule and is not.

    Resolved on both sides: these are host paths, where a symlinked home and its
    target are one directory.
    """
    ra, rb = _resolved(a), _resolved(b)
    return ra.is_relative_to(rb) or rb.is_relative_to(ra)


def _reachable_through(mounts: list[Mount], path: Path) -> Path | None:
    """The source of the mount that leaves `path` readable in the box after
    `mounts` have been applied in order, or None.

    Reachability, not containment, and the difference is the whole point. A
    mount whose SOURCE overlaps the path publishes it, at the mount's
    destination. Anything landing over that destination afterwards -- a tmpfs, a
    seal, a bind from somewhere else -- takes it away again. So the answer is a
    fact about the finished box, not about any one entry in the list, and it has
    to be computed the way bwrap computes it: in order, later winning.

    Tracked as box-side paths rather than a boolean because one host path can be
    published at several destinations (a bind-over mounts a source somewhere its
    own name does not appear), and a tmpfs over one of them says nothing about
    the others.

    Sources are resolved and destinations are not, because they are facts about
    two different filesystems. A source is a host path, where a symlinked home
    and its target are one directory. A destination is a location in the box,
    which bwrap creates as given -- and if the destination IS a symlink there,
    bwrap refuses the mount outright ("Can't mount on symlink destination", 0.11.2),
    so there is no case where following one host-side describes the box. Resolving
    them would let a symlink invent a mask that the box does not have, and a mask
    that is not there is a credential this returns None for.
    """
    target = _resolved(path)
    visible: dict[Path, Path] = {}
    for m in mounts:
        if not m.covers:
            continue
        for shadowed in [p for p in visible if p.is_relative_to(m.dst)]:
            del visible[shadowed]
        if m.src is None:
            continue
        src = _resolved(m.src)
        if not paths_overlap(target, src):
            continue
        # Where that overlap lands in the box: under the mount point when the
        # source holds the credential, AT it when the credential holds the
        # source. Geometry, not policy -- `paths_overlap` already decided.
        inside = target.relative_to(src) if target.is_relative_to(src) else Path()
        visible[m.dst / inside] = m.src
    # The LAST source to publish a given box path, because re-publishing the
    # same path overwrites the value while keeping its position: what the
    # operator has to remove is the mount that won, not the one it covered.
    return next(iter(visible.values()), None)


def _strict_ancestor(a: Path, b: Path) -> bool:
    """Is `a` a strict ancestor of `b` (a covers b, a != b)? Both resolved, so a
    symlinked home still compares against its real target."""
    try:
        ra, rb = a.resolve(), b.resolve()
    except OSError:
        return False
    return ra != rb and rb.is_relative_to(ra)


@dataclass(frozen=True)
class Sandbox:
    """One job's confinement profile; `wrapper()` yields the argv prefix that
    runs a command inside it. Immutable so a profile can be shared across a
    job's calls.
    """

    # The rw root (the job worktree), bound at its real absolute path because
    # remote-exec resolves build inputs by absolute path. Also the cwd, and one
    # of the two things the leak check protects. Mounted after the tmpfs and
    # before `binds`, so a bind written into the list can still pin a path
    # INSIDE the worktree read-only (the .git gitdir pointer is exactly that).
    root: Path
    # Everything else the box may touch, in mount order: later wins.
    #
    # There is one departure from pure list order, and it is an invariant rather
    # than a preference: an RO bind that is a strict ANCESTOR of the rw root or
    # of a tmpfs is hoisted to before the thing it would otherwise cover. The
    # ancestors are layout-dependent and mostly not chosen by the caller (an
    # operator's extra_ro, an editable source root, a resolved launcher chain),
    # so requiring them to be ordered by hand would turn a host's directory
    # layout into a profile bug. The motivating case: a launcher-resolved interpreter
    # root of /usr re-emitted after the $HOME tmpfs shadowed it read-only, and
    # $HOME/.cache is where vpython takes its lock. Nothing else moves -- the
    # relative order of the binds a caller wrote is preserved exactly.
    binds: tuple[BindSpec, ...] = ()
    # (mount point, size in bytes). Mounted before the root and the binds so a
    # bind under a tmpfs (the worktree under the blanked $HOME) lands on top.
    tmpfs: tuple[tuple[str, int], ...] = ()
    # The complete environment inside the sandbox (--clearenv first, so the
    # daemon's env -- credentials included -- never leaks through). Verbatim and
    # complete: nothing is added on the way to --setenv, so what a profile says
    # the box's environment is, is what it is. A caller wanting the
    # noninteractive defaults a build expects merges `spec.DEFANG_ENV` in itself.
    env: tuple[tuple[str, str], ...] = ()
    # cgroup limits, applied via `systemd-run --user --scope` when available.
    # Empty string / 0 skips that property.
    memory_max: str = ""
    cpu_quota: str = ""
    tasks_max: int = 0
    use_cgroup: bool = True
    # systemd slice to place the scope under (`--slice=`). A stable cgroup anchor
    # for host-side accounting; empty leaves systemd's default.
    slice_unit: str = ""
    # Give the box its OWN network namespace: no route off the machine, and a
    # fresh loopback of its own. This is the real isolation an L3 allowlist only
    # approximates, because RBE and the rest of Google share one frontend IP
    # range.
    #
    # It also redefines what "127.0.0.1" means in here. The box's loopback is
    # NOT the host's, so a host-side listener on a port is unreachable and any
    # such proxy needs its in-box half (a relay onto a bind-mounted UNIX socket,
    # which crosses the boundary because it is a filesystem object). Turning
    # this on without that half is what silently takes remote builds away.
    unshare_net: bool = False

    def resolve(self) -> list[Mount]:
        """The bind list as the ordered mount operations it compiles to.

        This is the representation everything else works from: `wrapper()`
        formats it, `_assert_no_leak` checks it, and the inspector prints it.
        Optional sources that cannot be stat'd are already dropped here and
        mandatory ones have already raised, so what comes back is exactly what
        the box will have.
        """
        tmpfs_mounts = [Path(m) for m, _ in self.tmpfs]
        mounts: list[Mount] = []
        # Deferred to the very end: a seal cannot be remounted ro until every
        # hole through it has been mounted. See Seal.
        seal_ro: list[Mount] = []
        # What mode each literal path is currently mounted with, so a bind that
        # is already in effect is not emitted twice. Keyed on the literal path,
        # never the resolved one: an interpreter root and the unversioned symlink
        # a venv names it by resolve to the same directory but are two distinct
        # destinations in the box, and binding only one leaves the other missing
        # so every exec fails ENOENT. Any covering mount invalidates the entries
        # it covers -- otherwise a pin re-stated after a hole was punched through
        # it would look redundant and be dropped, which is the guard the .git
        # dance depends on.
        in_effect: dict[Path, Mode] = {}

        def emit(m: Mount) -> None:
            mounts.append(m)
            if not m.covers:
                return
            for k in [k for k in in_effect if k == m.dst or k.is_relative_to(m.dst)]:
                del in_effect[k]

        early: list[Bind] = []  # RO ancestors of a tmpfs: before the tmpfs block
        mid: list[Bind] = []  # RO ancestors of the root: after it, before the root
        rest: list[BindSpec] = []
        for spec in self.binds:
            if isinstance(spec, Bind):
                # Every bind is validated here, once: an optional source we
                # cannot stat is dropped, a mandatory one fails the profile.
                # Splitting the check between this pass and the emit loop below
                # is how one of the two ends up unreachable.
                if spec.optional:
                    if not _bindable(spec.path):
                        continue
                elif not spec.path.exists():
                    raise FileNotFoundError(f"bind source missing: {spec.path}")
                if spec.mode is RO:
                    if spec.path.resolve() in _SYSTEM_RO_ROOTS:
                        continue
                    if any(_strict_ancestor(spec.path, t) for t in tmpfs_mounts):
                        early.append(spec)
                        continue
                    if _strict_ancestor(spec.path, self.root):
                        mid.append(spec)
                        continue
            rest.append(spec)

        def emit_bind(b: Bind) -> None:
            if in_effect.get(b.path) is b.mode:
                return
            emit(Mount(b.mode.value, b.path, b.path))
            in_effect[b.path] = b.mode

        for b in early:
            emit_bind(b)
        for mnt, size in self.tmpfs:
            emit(Mount("tmpfs", Path(mnt), size=size))
        for b in mid:
            emit_bind(b)
        emit(Mount("rw", self.root, self.root))
        in_effect[self.root] = RW
        for spec in rest:
            match spec:
                case Bind():
                    emit_bind(spec)
                case Overlay(path=p):
                    if not p.is_dir():
                        raise FileNotFoundError(f"tmp-overlay source missing: {p}")
                    emit(Mount("overlay", p, p))
                case Seal(path=p):
                    if not p.is_dir():
                        raise FileNotFoundError(f"seal source missing: {p}")
                    emit(Mount("tmpfs", p))
                    seal_ro.append(Mount("seal-ro", p))
                case BindOver(src=src, dst=dst):
                    if not src.exists():
                        raise FileNotFoundError(f"bind-over source missing: {src}")
                    emit(Mount("ro", dst, src))
        return [*mounts, *seal_ro]

    def exposed_credential(
        self, credentials: Iterable[Path]
    ) -> tuple[Path, Path] | None:
        """The first (mount source, credential) this profile leaves readable in
        the box, or None. The credential-absence invariant, asked of the box.

        Over the resolved mounts, with the fixed system surface in front of them,
        for the same reason `_assert_no_leak` runs over that list: it is what the
        box gets. The question a caller needs answered is "can the payload read
        this file", and no reading of the bind list alone answers it -- in either
        direction.

        Not, in particular, "does a bind name a path containing the credential".
        That proxy fails on a host whose home lives UNDER a system root. Some
        enterprise Linux images place home directories inside `/usr/local`, and
        there a credential store at `~/.config/<name>` is a path inside `/usr`:

        - every box already has the credential's directory in reach through the
          unconditional `--ro-bind /usr /usr`, which no bind list mentions and
          the proxy therefore never audits. What actually keeps the credential
          out is the tmpfs over $HOME landing on top of it, and that is a
          property of the mount ORDER;
        - meanwhile a spec that names `/usr` (an interpreter root under it, say
          -- see launch.interpreter_roots) reads as an exposure while compiling
          to no mount at all, because resolve() drops an RO bind of a system root
          as already in effect. The same layout is why that dedup and the
          RO-ancestor hoisting above exist; this is the third symptom of it.

        So a bind naming `/usr` here is ordinary rather than alarming, and the
        thing worth refusing is a box that can actually read the file.
        """
        mounts = [*_system_mounts(), *self.resolve()]
        for cred in credentials:
            source = _reachable_through(mounts, cred)
            if source is not None:
                return source, cred
        return None

    def wrapper(self) -> list[str]:
        """The argv prefix: [systemd-run ...] bwrap ... -- ready to have the
        actual command appended."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root missing: {self.root}")
        mounts = self.resolve()
        # Guarantee: no later mount may cover (be an ancestor-or-equal of) an
        # earlier tmpfs or the rw root. A violation is a silent sandbox leak --
        # the home-tmpfs shadow if the hoisting above ever regresses, or a
        # worktree turned read-only (or replaced with real disk) by a later
        # ancestor bind. Fail the profile at assembly rather than hand the agent
        # a box that is not what it claims to be.
        self._assert_no_leak(mounts)
        argv = list(self._cgroup_args())
        argv += ["bwrap", *_SYSTEM_ARGS]
        if self.unshare_net:
            argv += ["--unshare-net"]
        argv += self._journal_socket_args()
        argv += _mount_args(mounts)
        argv += _ISOLATION_ARGS
        argv += ["--clearenv"]
        for k, v in self.env:
            argv += ["--setenv", k, v]
        argv += ["--chdir", str(self.root)]
        return argv

    def _assert_no_leak(self, mounts: list[Mount]) -> None:
        """No later mount may cover (be an ancestor-or-equal of) an earlier
        tmpfs or the rw root, and raise naming the offending pair if one does.

        Runs over the resolved list rather than the emitted argv: one
        representation, checked and displayed identically. Parsing back the argv
        would mean a second reading of the same intent that can disagree with
        the first without anything failing.
        """
        root = self.root.resolve()
        protected = [
            (i, m.dst.resolve())
            for i, m in enumerate(mounts)
            if m.op == "tmpfs" or (m.op == "rw" and m.dst.resolve() == root)
        ]
        for pi, pp in protected:
            for mi, m in enumerate(mounts):
                if mi <= pi or not m.covers:
                    continue
                # is_relative_to is True for equality too: a later bind that
                # remounts the exact path is just as much a shadow.
                if pp.is_relative_to(m.dst.resolve()):
                    raise ValueError(
                        f"sandbox mount order leaks: {m.dst} (index {mi}) covers"
                        f" protected {pp} (index {pi}) -- a later bind shadows"
                        f" a tmpfs or the rw root"
                    )

    def _cgroup_args(self) -> list[str]:
        # systemd-run needs the user manager; when it is absent (a bare chroot,
        # a container) degrade to bwrap-only rather than failing every call.
        if not self.use_cgroup or shutil.which("systemd-run") is None:
            return []
        args = ["systemd-run", "--user", "--scope", "-q", "--collect"]
        if self.slice_unit:
            args += [f"--slice={self.slice_unit}"]
        if self.memory_max:
            args += ["-p", f"MemoryMax={self.memory_max}"]
        if self.cpu_quota:
            args += ["-p", f"CPUQuota={self.cpu_quota}"]
        if self.tasks_max:
            args += ["-p", f"TasksMax={self.tasks_max}"]
        return [*args, "--"]

    @staticmethod
    def _journal_socket_args() -> list[str]:
        # Bind the host journal socket (when present) so the login shell's
        # systemd-cat connect succeeds instead of spraying "Failed to create
        # stream fd" into every result. Keeping `bash -lc` preserves the
        # deployment-set build PATH; this just silences its one sandbox casualty.
        if _JOURNAL_STDOUT_SOCK.is_socket():
            return ["--ro-bind", str(_JOURNAL_STDOUT_SOCK), str(_JOURNAL_STDOUT_SOCK)]
        return []

    # There is deliberately no path-containment check here. Confinement is the
    # mount namespace: a caller runs the whole loop inside the wrapper, so a
    # second in-process check of the same policy would be a copy that can drift
    # from the argv without anything failing. A tool that needs containment
    # belongs inside the box, not behind a predicate.


def _mount_args(mounts: list[Mount]) -> list[str]:
    """Format resolved mounts as bwrap arguments, one op at a time and in
    order. The only place a Mount becomes a string."""
    argv: list[str] = []
    for m in mounts:
        match m.op:
            case "ro" | "rw":
                flag = "--ro-bind" if m.op == "ro" else "--bind"
                argv += [flag, str(m.src), str(m.dst)]
            case "tmpfs":
                if m.size:
                    argv += ["--size", str(m.size)]
                argv += ["--tmpfs", str(m.dst)]
            case "overlay":
                argv += ["--overlay-src", str(m.src), "--tmp-overlay", str(m.dst)]
            case "seal-ro":
                argv += ["--remount-ro", str(m.dst)]
    return argv
