# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Host-side credential minting for the egress proxies.

One provider per upstream identity. Each mints a short-lived credential in the
host process -- where the durable credential lives -- and RETURNS it, so
the proxy can attach it to a request the box never sees. That "returns rather
than writes" is the whole distinction from the token brokers this replaced: they
wrote a bearer into a directory bound into the box, which put the credential
back where it was not supposed to be.

Which principal to mint as is supplied by the deployment. The implementation
uses Application Default Credentials to impersonate a keyless service account;
the durable source credential remains host-side and only the short-lived access
token is returned to the proxy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

# Returns (access_token, aware-UTC expiry). Tests plug in here rather than
# reaching for a real cloud credential, which is exactly the dependency the
# proxies exist to keep out of the loop.
Fetch = Callable[[], Awaitable[tuple[str, datetime]]]

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"


def _read_adc_impersonate_token(target: str) -> tuple[str, datetime]:
    # Blocking; run via to_thread below. The source principal needs Token
    # Creator on the target.
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import Request

    source, _ = google.auth.default()
    creds = impersonated_credentials.Credentials(
        source_credentials=source,
        target_principal=target,
        target_scopes=[_CLOUD_PLATFORM],
    )
    creds.refresh(Request())
    # google.auth expiry is naive UTC by convention.
    expiry = creds.expiry or (datetime.now(UTC).replace(tzinfo=None))
    return creds.token, expiry.replace(tzinfo=UTC)


def vertex_fetch(impersonate: str) -> Fetch:
    """The Vertex mint provider: a cloud-platform token AS `impersonate`.

    An empty target is a config error raised here rather than surfacing later as
    an opaque IAM failure on the proxy's first mint. Config validates the same
    thing at load; this is the guard for a caller that did not come through it.
    """
    if not impersonate:
        raise ValueError(
            "the Vertex proxy needs a principal to mint as: pass the target SA email"
        )

    async def _fetch() -> tuple[str, datetime]:
        return await asyncio.to_thread(_read_adc_impersonate_token, impersonate)

    return _fetch


async def rbe_token(lifetime_s: int = 1800) -> str:
    """A cloud-scoped luci token, returned rather than written.

    Cloud scope, not gerrit: the RBE proxy needs nothing else. Note the bound is
    on LIFETIME and not on REACH -- chromium-review accepts a cloud-platform
    bearer and resolves it to the full user account (verified), so this token is
    Gerrit-capable while it lives. That is why it stays in this process.
    """
    proc = await asyncio.create_subprocess_exec(
        "luci-auth",
        "token",
        "-scopes-cloud",
        f"-lifetime={lifetime_s}s",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            f"luci-auth token failed (exit {proc.returncode}): "
            f"{err.decode(errors='replace').strip()[:300]}"
        )
    return out.decode().strip()


__all__ = ["Fetch", "rbe_token", "vertex_fetch"]
