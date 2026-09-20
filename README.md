# aisan

aisan runs a coding agent, its state, and your repository inside a box. The box
has no network route or credentials. Model calls pass through a host-side proxy
that checks each request and adds the real credential.

```sh
python -m pip install aisan   # or: uv tool install aisan
aisan claude /path/to/repo
```

The command starts an interactive Claude Code session. `aisan codex` and
`aisan opencode` provide the same confinement:

- **No credentials in the box.** `~/.claude/.credentials.json` is never
  mounted. The client sends a per-box placeholder, and the proxy replaces it
  with the host subscription login or the static API key selected by
  `--api-key`.
- **No network by default.** The box gets its own network namespace with no
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
  the host. Only servers a bind spec names are started at all.

The same three commands, run on the host and then from inside the box:

<p align="center">
  <img src="https://raw.githubusercontent.com/schuay/aisan/main/docs/demo.gif" width="660"
       alt="On the host, reading the credential file, listing ~/.ssh, and fetching https://example.com all succeed. Inside an aisan box with permissions bypassed, the agent runs the same three and each one fails: no credential file, no key directory, no DNS.">
</p>

## Inspecting profiles with `--explain`

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
  anthropic 127.0.0.1:8713 -> /tmp/aisan-1000/proxy-59d1d1bc/anthropic.sock

== tmpfs mounts (intended writable scratch; binds below land on top) ==
  [ 39] /tmp  (2147483648)
  [ 43] /home/user  (1073741824  <- $HOME)
  [ 45] /tmp/aisan-1000  (no size)

== binds by destination (deeper wins; * marks a guard: nothing plain below) ==
  system    /usr /bin /lib /lib64 /sbin /etc /proc /dev
  [ 63] rw      * /home/user/.cache/aisan-claude/aisan-4475d1c31168
  [ 66] ro      * /home/user/.config/git/config
  [ 69] rw-root   /path/to/repo
  [ 72] ro      * /tmp/aisan-1000/proxy-59d1d1bc
  [ 75] seal-ro * /tmp/aisan-1000

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
project. The keys are `ro`, `rw`, `overlay`, `path` entries prepended to the
box PATH, and `mcp`:

```toml
ro      = ["~/depot_tools"]
overlay = ["~/.cache/vpython-root.1000"]
path    = ["~/depot_tools"]
mcp     = ["v8-mcp"]
```

The `path` key grants no filesystem access. Each entry must be covered by a
mount in the same file. `include` expands other spec files in place before the
including file's keys, which makes precedence explicit.
[`examples/depot_tools.toml`](examples/depot_tools.toml) shows a complete file.

`mcp` names the host MCP servers this box may start, by server name, by the
command, or by the command's full path; `["*"]` admits every local stdio
declaration. A command admits every declaration that runs it, so name a
server launched through `npx` or `uvx` by its server name. A box whose specs name none starts none, and the launcher says
which it withheld. A host client config is one list shared by every box: an
unattended box should not gain a channel to the outside because a server was
added for interactive work.
[`examples/mcp.toml`](examples/mcp.toml) shows the interactive-only split.

## As a library: unattended API jobs

The same mechanism supports headless workloads. A preset maps arguments to a
`BoxSpec`; `Box` compiles the spec, starts its backends, and returns the command
arguments. The Vertex backend uses ADC impersonation to mint short-lived tokens
on the host, so batch boxes don't receive Google credentials.

Remote build clients such as Siso use plaintext HTTP/2 to a local REAPI
endpoint while the bearer stays on the host. The proxy checks `:authority` and
`:path`, adds credentials per stream, and returns policy refusals as gRPC
errors.

## Design rules

- **`BoxSpec`** is frozen, non-defaulting data. Call sites state every mount.
  `Limits` may leave resource caps unset.
- **Mounts form a tree keyed by destination.** Order in a `BoxSpec` carries no
  meaning: an ancestor is mounted before its descendants, so the deeper entry
  wins. Two entries at one path must agree, or be identity binds, where the
  stricter mode wins. A *guard* (the default for `Bind` and `Overlay`; always
  for `Seal` and `BindOver`) admits only other guards below it, so a user bind
  cannot reopen part of a policy mount. User bind files and `extra_ro` are
  plain.
- **Credential-aware egress in both network modes.** Isolated boxes reach host
  proxies through Unix sockets and in-box loopback relays. Interactive boxes
  started with `--net` reach authenticated host-loopback TCP listeners directly;
  a private runtime file supplies the per-session proxy token without placing it
  in process arguments.
- **Fail-closed request policy.** A policy exception denies the request. Refusal
  messages state the policy reason and omit internal callback names.
- **Presets are pure functions** from arguments to `BoxSpec` values.

The model-neutral core covers `BoxSpec`, sandbox compilation, Git bind policy,
lifecycle, inspection, backend interfaces, relays, request policy, and REAPI.
Provider adapters, client presets, interactive launchers, and MCP importers are
separate integrations.

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

`--grant NAME` adds a named grant: the mounts, PATH entries and environment
some tree needs inside a box with no network route.

```sh
aisan claude /path/to/v8 --grant depot_tools
```

The `depot_tools` grant finds the checkout through `autoninja`, overlays the
vpython cache, and disables network-dependent updates and presubmit checks.
These environment settings require a grant because bind files contain paths
only. Grants apply before `--binds`, so user files retain precedence.

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

## Plugin commands

Out-of-tree commands register in the `aisan.commands` entry point group:

```toml
[project.entry-points."aisan.commands"]
jetski = "aisan_corp.cli.jetski:main"
```

Installed alongside aisan, they are dispatched by name and need no wrapper
binary of their own:

```sh
uv tool install aisan --with git+ssh://example.com/aisan-corp
aisan jetski /path/to/repo
```

The contract is the one the built-in launchers already follow: a callable
taking the tokens after the command name and returning an exit status, with
`LaunchRefused` handled by the dispatcher. Everything else a plugin imports
from `aisan` is internal and may change between versions.

The dispatcher enforces four rules:

- **Plugins can't override built-ins.** Claims on `claude`, `codex`, `opencode`,
  or `explain` are refused and reported.
- **Discovery costs nothing on the built-in path.** The group is read only when
  the first token names no built-in, and when help is printed. `aisan claude`
  scans no metadata and imports no plugin.
- **Plugin failures are isolated.** A failed import names the plugin and leaves
  other commands working.
- **Duplicate names resolve by provider name.** The losing plugin is reported.

Plugins that build boxes should support `--explain` and print the resolved
profile like built-in launchers. Aisan's read-only runtime binds also make
plugins installed in the same environment readable inside boxes with egress.

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
