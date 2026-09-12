# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Provide shared infrastructure for aiohttp-based host proxies.

Provider modules own request policy, credentials, upstream behavior, and error
envelopes. This module owns only the mechanics they share: request-size and rate
limits, unambiguous body parsing, constant-time token primitives, and UNIX/TCP
server lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

# Allow large prompts and histories while limiting bulk exfiltration.
MAX_BODY_BYTES = 32 << 20


class AmbiguousBody(ValueError):
    """A body that strict and lenient JSON parsers may interpret differently.

    Duplicate keys can resolve to different values, and Python accepts bare
    ``NaN`` and ``Infinity`` values that strict parsers reject. In either case,
    the proxy and upstream could apply policy to different data.
    """


def json_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object, refusing a key repeated at any depth.

    This function is an ``object_pairs_hook``, so it checks nested objects as
    well as the top level. Duplicate fields inside tool declarations matter as
    much as duplicate top-level fields.
    """
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise AmbiguousBody(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str) -> object:
    """Refuse the JSON constants only a lenient parser accepts.

    Python reads ``NaN``, ``Infinity``, and ``-Infinity`` as floats, while a
    strict parser rejects them. Treat the disagreement as ambiguity.
    """
    raise AmbiguousBody(f"non-JSON constant {value}")


def load_json_unambiguous(body: bytes) -> object:
    """Parse JSON while rejecting duplicate keys and nonstandard constants.

    Raise ``AmbiguousBody`` when strict and lenient parsers could disagree. Raise
    ``ValueError`` for malformed JSON. Each proxy decides how to handle them;
    Anthropic's hello routes also accept an empty body.
    """
    return json.loads(
        body, object_pairs_hook=json_unique_pairs, parse_constant=_reject_constant
    )


# Relay backoff, quota, and request identifiers. The caller handles Content-Type
# separately because it supplies a default.
_RELAYED_RESPONSE_HEADERS = frozenset({"retry-after", "x-request-id", "request-id"})
_RELAYED_RESPONSE_PREFIXES = ("anthropic-ratelimit-", "x-ratelimit-", "ratelimit-")


def quoted_names(names: set[str]) -> str:
    """Sort and quote box-controlled field names for a refusal message.

    ``repr`` prevents a field name containing a newline from forging host log
    lines.
    """
    return ", ".join(repr(name) for name in sorted(names))


def relayed_response_headers(
    up_headers, *, content_type_default: str = "application/json"
) -> dict[str, str]:
    """Return Content-Type and headers used for diagnostics and backoff.

    Drop other upstream headers because they describe a different connection
    from the one between the proxy and box.
    """
    out = {"Content-Type": up_headers.get("Content-Type", content_type_default)}
    for name in up_headers:
        low = name.lower()
        if low in _RELAYED_RESPONSE_HEADERS or low.startswith(
            _RELAYED_RESPONSE_PREFIXES
        ):
            out[name] = up_headers[name]
    return out


# Never follow upstream redirects. aiohttp drops Authorization across origins
# but retains x-api-key, which could send the key and request body to an
# unapproved host. The proxied APIs don't use redirects.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def is_redirect(status: int) -> bool:
    return status in _REDIRECT_STATUSES


def request_path(request) -> str:
    """Return the raw request path that will be forwarded, without its query.

    ``request.path`` decodes escapes before an allowlist can inspect them. For
    example, ``/v1%2Fmessages`` becomes the permitted ``/v1/messages`` even
    though forwarding preserves the encoded path. Check the forwarded form.
    """
    return request.rel_url.raw_path


@dataclass
class RateLimit:
    """Use a token bucket to bound requests from a runaway client.

    This bounds faulty loops; it doesn't allocate quota fairly. A monotonic
    clock prevents wall-clock changes from creating tokens.
    """

    per_minute: int = 120
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)

    def allow(self) -> bool:
        now = time.monotonic()
        if self._last == 0.0:
            self._tokens, self._last = float(self.per_minute), now
        self._tokens = min(
            float(self.per_minute),
            self._tokens + (now - self._last) * self.per_minute / 60.0,
        )
        self._last = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


class LogGate:
    """Limit refusal warnings from an in-box loop.

    Path refusals occur before request rate limiting, and 120 warnings per minute
    would still obscure useful logs. Allow a short burst, count suppressed
    warnings, and report the count when logging resumes.

    Policy exceptions and upstream outage warnings remain unlimited. The former
    signal boundary bugs, while the box doesn't control the latter.
    """

    def __init__(self, per_minute: int = 12) -> None:
        self._limit = RateLimit(per_minute=per_minute)
        self._suppressed = 0

    def warning(self, logger: logging.Logger, msg: str, *args: object) -> None:
        if not self._limit.allow():
            self._suppressed += 1
            return
        if self._suppressed:
            logger.warning(
                "(%d refusal warnings suppressed by the log rate limit)",
                self._suppressed,
            )
            self._suppressed = 0
        logger.warning(msg, *args)


async def serve(socket_path: Path, app: web.Application) -> web.AppRunner:
    """Serve `app` on `socket_path`; the caller owns the runner's lifetime.

    The socket lives in the box's mounted runtime directory. Remove stale sockets
    before binding. Per-box runtime paths prevent this from unlinking a peer's
    live socket.
    """
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    # Policy modules log decisions; aiohttp access logs would bury refusals.
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.UnixSite(runner, str(socket_path)).start()
    # The box runs as the same user, so exclude other users from the socket.
    socket_path.chmod(0o600)
    return runner


async def serve_tcp(app: web.Application) -> tuple[web.AppRunner, int]:
    """Serve `app` on one explicit IPv4 loopback socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    runner = web.AppRunner(app, access_log=None)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(socket.SOMAXCONN)
        sock.setblocking(False)
        assigned = int(sock.getsockname()[1])
        await runner.setup()
        await web.SockSite(runner, sock).start()
    except BaseException:
        await runner.cleanup()
        sock.close()
        raise
    else:
        return runner, assigned


def token_matches(presented: str | None, expected: str | None) -> bool:
    """Whether a client-presented proxy token is valid for this mode."""
    if expected is None:
        return True
    return presented is not None and secrets.compare_digest(presented, expected)


def bearer_token(request: web.Request) -> str | None:
    """Return one exact bearer credential, rejecting duplicates and malformed input."""
    values = request.headers.getall("Authorization", [])
    if len(values) != 1:
        return None
    scheme, separator, value = values[0].partition(" ")
    if not separator or scheme.lower() != "bearer" or not value or " " in value:
        return None
    return value


async def run_forever(socket_path: Path, app: web.Application) -> None:
    """Serve until cancelled, then clean up the runner and socket."""
    runner = await serve(socket_path, app)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        socket_path.unlink(missing_ok=True)
