# aisan

aisan is a set of Linux confinement and credential-aware egress components for
build and coding workloads. It compiles an explicit filesystem policy to
Bubblewrap, keeps network credentials on the host, and exposes only the
allowlisted operations a sandboxed process needs.

The model- and client-neutral core is roughly 3,100 lines of Python. That count
covers `BoxSpec`, the sandbox compiler, git bind policy, lifecycle and launcher,
inspection, the backend interface, relay, fail-closed policy, and the REAPI
transport. Provider/client adapters, presets, interactive session launchers,
and MCP importers are integrations outside that core count. The number is an
audit bound, not a comparison with another project's total source size.

The package also ships integrations for selected model APIs and interactive
Claude Code, Codex, and opencode sessions. Those are examples and conveniences;
the core does not depend on any one model or agent.

## What it provides

- **`BoxSpec`** is frozen, non-defaulting data. A reviewer can read a call site
  and see what is mounted without simulating default resolution. `Limits` is
  the exception: an unset resource cap is not an unstated mount.
- **One ordered bind list, later wins**, matching Bubblewrap's mount behavior.
  `Bind`, `Seal`, `Overlay`, and `BindOver` read top to bottom.
- **Credential-aware egress in both network modes.** Isolated boxes reach host
  proxies through Unix sockets and in-box loopback relays. Interactive boxes
  started with `--net` reach authenticated host-loopback TCP listeners directly;
  a private runtime file supplies the per-session proxy token without placing it
  in process arguments.
- **Fail-closed request policy.** A policy exception denies the request. Refusal
  messages name the policy reason rather than an internal callback.
- **Presets as pure `args -> BoxSpec` functions**, rather than project switches
  hidden inside the sandbox compiler.
- **`aisan explain`** renders resolved binds and the exact Bubblewrap argv from
  the same `Box` used to launch the process.
- **User bind specs** (`--binds FILE`, repeatable) name paths to mount `ro`,
  `rw`, or `overlay`, plus `path` entries prepended to the box PATH. The key
  grants nothing on its own: every entry must be covered by a mount the same
  file names. `examples/depot_tools.toml` is a worked example.

The REAPI transport is the largest specialized core component. Remote build
clients such as siso can speak plaintext HTTP/2 to a local endpoint while the
real bearer stays on the host. It checks `:authority` and `:path` together,
injects credentials per HTTP/2 stream, and refuses in gRPC's own terms so a
policy decision is not mistaken for a retryable network failure.

## Requirements

- Linux, Python 3.12 or newer, and `bubblewrap` (`bwrap`). User namespaces must
  be available to the invoking user.
- `systemd-run --user` is optional. Cgroup limits are skipped when the command
  is absent. On a host without a usable user manager, disable them explicitly
  with `Limits(use_cgroup=False)`.
- Interactive sessions require the corresponding host CLI (`claude`, `codex`,
  or `opencode`) to be installed and already logged in.
- RBE/V8 use additionally requires the relevant siso/depot_tools environment
  and `luci-auth`.
- Vertex credential minting requires the `google-auth` extra and Application
  Default Credentials.

Boxes have no general network access by default. Interactive `claude`, `codex`,
and `opencode` sessions accept `--net` before the literal `--` to share the
host network namespace. That exposes the internet, LAN/VPN routes, and
host-local services in both directions; configured model credential files stay
unmounted and model calls still pass through authenticated host proxies.

## Installation

From a checkout:

```sh
python -m pip install .
```

For Vertex credential minting:

```sh
python -m pip install '.[google-auth]'
```

This installs one human-facing command with inspection and interactive
subcommands:

```sh
aisan explain --help
aisan claude /path/to/repo
aisan codex /path/to/repo
aisan opencode /path/to/repo
aisan codex /path/to/repo --net
```

Launcher options come before a literal `--`; arguments after it are passed to
the underlying client unchanged.

Runtime dependencies are limited to `aiohttp` and `h2`. The Google credential
chain is optional. A boundary test walks the package AST and fails when a module
imports an undeclared third-party dependency.

### Local checks

Prepare the development environment while network access is available:

```sh
uv sync
```

Enable the repository's offline pre-commit checks with:

```sh
git config core.hooksPath .githooks
```

The hook runs staged-file checks with `uv run --offline --no-sync`: committing
does not resolve, install, update, or download dependencies. The checks do not
rewrite files; run Ruff or `scripts/add-license-headers.py` explicitly to apply
a reported fix.

## Relationship to sandbox-runtime

[Anthropic's sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)
(srt) is the broader choice for a general confined coding agent and supports
Linux, macOS, and Windows. aisan is Linux-only and concentrates on explicit
mount composition plus credential-aware transports, including plaintext HTTP/2
REAPI traffic and sealing an existing directory around a writable hole.

Reviewing srt informed three decisions here: policy exceptions fail closed,
refusals state an actionable policy reason, and launch commands remain argv so
payload bytes never pass through a host shell. These are general security and
interface rules, independently implemented in aisan; no srt source code was
copied or adapted. The last detailed comparison used srt revision `121c6ac`
(v0.0.71) on 2026-08-10, so current srt behavior should be checked before relying
on any feature difference.

## Status

Pre-1.0. Treat the API as unstable.

## License

MIT (see [LICENSE](LICENSE)).
