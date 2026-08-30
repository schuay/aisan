# aisan

aisan runs a coding agent with everything in the box: the harness, its
state, and your repo. Nothing else. The box has no network route and
holds no credential. Model calls still work: each harness gets a host-side
proxy that checks requests against an allowlist and attaches the real
credential to traffic the box never sees.

```sh
python -m pip install aisan   # or: uv tool install aisan
aisan claude /path/to/repo
```

That is a normal interactive Claude Code session (`aisan codex` and
`aisan opencode` work the same way), with three differences:

- **Zero credentials in the box.** `~/.claude/.credentials.json` is never
  mounted. The token the client sees is a per-box placeholder; the proxy
  drops it and attaches the host's real credential: the subscription login
  by default, or a static API key with `--api-key`. A test asserts from
  inside a real box that the credential file does not exist.
- **Zero network by default.** The box gets its own network namespace with no
  route off the machine. The one egress is a loopback relay to the model
  proxy over a Unix socket. `--net` opts back into host networking when a
  task needs it; credential files stay unmounted and model calls still pass
  through the authenticated proxy.
- **Selected filesystem slices.** The repo is bound rw at its real absolute
  path, system directories ro (`/usr`, `/etc`; fresh `/proc` and `/dev`), a
  tmpfs over `$HOME` and `/tmp`, and nothing else unless a bind spec names
  it. Local stdio MCP servers declared on the host are started inside the
  box, where they inherit its filesystem, cleared environment, and network
  namespace; remote MCP declarations and their authentication state stay on
  the host.

The same three commands, run on the host and then from inside the box:

<p align="center">
  <img src="https://raw.githubusercontent.com/schuay/aisan/main/docs/demo.gif" width="660"
       alt="On the host, reading the credential file, listing ~/.ssh, and fetching https://example.com all succeed. Inside an aisan box with permissions bypassed, the agent runs the same three and each one fails: no credential file, no key directory, no DNS.">
</p>

## Nothing on faith: `--explain`

Every launcher takes `--explain`: it prints the resolved profile and the
exact Bubblewrap argv from the same `Box` object used to launch, then exits.
Trimmed:

<p align="center">
  <img src="https://raw.githubusercontent.com/schuay/aisan/main/docs/explain.svg" width="660"
       alt="aisan claude /path/to/repo --explain: resolved binds, egress backends, and environment">
</p>

<details>
<summary>Text version</summary>

```
$ aisan claude /path/to/repo --explain

== inputs ==
  harness   claude-code
  repo      /path/to/repo
  network   own namespace (no route off the machine)

== egress backends (host half on a socket, in-box on loopback) ==
  anthropic 127.0.0.1:8713 -> /tmp/aisan-proxy-59d1d1bc/anthropic.sock

== tmpfs mounts (mounted before binds; intended writable scratch) ==
  [ 39] /tmp  (2147483648)
  [ 43] /home/user  (1073741824  <- $HOME)

== binds in argv order (later shadows earlier on overlap) ==
  system    /usr /bin /lib /lib64 /sbin /etc /proc /dev
  [ 45] rw-root   /path/to/repo
  [ 63] rw        /home/user/.cache/aisan-claude/aisan-4475d1c31168
  [ 66] ro        /home/user/.config/git/config
  [ 69] ro        /tmp/aisan-proxy-59d1d1bc

== environment (the box's complete environment; --clearenv first) ==
  CLAUDE_CONFIG_DIR=/path/to/repo/.aisan-claude-state
  GIT_PAGER=cat
  HOME=/home/user
  PATH=/usr/bin
  ...
```
</details>

## User bind specs

Presets cover the harness; `--binds FILE` (repeatable, TOML) covers your
project. The keys are `ro`, `rw`, `overlay`, and `path` entries prepended to
the box PATH:

```toml
ro      = ["~/depot_tools"]
overlay = ["~/.cache/vpython-root.1000"]
path    = ["~/depot_tools"]
```

The `path` key grants nothing on its own: every entry must be covered by a
mount the same file names. `include` pulls in other spec files, expanded in
place and before the including file's own keys, so a growing collection
composes in an order the files state rather than one the command line
implies. [`examples/depot_tools.toml`](examples/depot_tools.toml) is a worked
example with the reasoning written down.

## As a library: unattended API jobs

The same mechanism drives headless workloads. A preset is a pure
`args -> BoxSpec` function; `Box` compiles the spec, starts the backends, and
returns argv. The Vertex backend mints short-lived tokens host-side through
ADC impersonation, so a batch job's box carries no Google credential either.
This package was extracted from an autonomous patch pipeline that runs
model-driven build/test jobs against V8 worktrees; that pipeline remains its
first consumer.

The REAPI transport is the largest specialized core component. Remote build
clients such as siso can speak plaintext HTTP/2 to a local endpoint while the
real bearer stays on the host. It checks `:authority` and `:path` together,
injects credentials per HTTP/2 stream, and refuses in gRPC's own terms so a
policy decision is not mistaken for a retryable network failure.

## Design rules

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

The model- and client-neutral core is roughly 3,100 lines of Python. That count
covers `BoxSpec`, the sandbox compiler, git bind policy, lifecycle and launcher,
inspection, the backend interface, relay, fail-closed policy, and the REAPI
transport. Provider/client adapters, presets, interactive session launchers,
and MCP importers are integrations outside that core count. The number is an
audit bound, not a comparison with another project's total source size.

## Requirements

- Linux, Python 3.12 or newer, and `bubblewrap` (`bwrap`). User namespaces must
  be available to the invoking user; some distributions restrict unprivileged
  user namespaces by default.
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

```sh
python -m pip install aisan
```

For Vertex credential minting:

```sh
python -m pip install 'aisan[google-auth]'
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
is the broader cross-platform tool for a general confined coding agent; aisan
is Linux-only and concentrates on whole-harness confinement, explicit mount
composition, and credential-aware transports such as the plaintext HTTP/2
REAPI proxy.

## Status

Pre-1.0. Treat the API as unstable.

## License

MIT (see [LICENSE](LICENSE)).
