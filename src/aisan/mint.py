# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Mint short-lived credentials for host-side egress proxies.

Providers return credentials in memory so proxies can attach them after requests
leave the box. Earlier brokers wrote bearers into directories mounted inside the
box.

The deployment selects the principal. Vertex uses Application Default
Credentials to impersonate a keyless service account while keeping the durable
source credential on the host.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from .hostproc import HostChild, neutral_child

# Return an access token and timezone-aware UTC expiry. Tests provide fakes.
Fetch = Callable[[], Awaitable[tuple[str, datetime]]]

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"

# Bound luci-auth on the request path. A valid login normally returns within a
# second; a longer run may be waiting for interaction.
_TOKEN_TIMEOUT_S = 30

# Give credentials without an expiry a short cache lifetime.
_NO_EXPIRY_TTL = timedelta(minutes=5)


def _read_adc_impersonate_token(target: str) -> tuple[str, datetime]:
    # This blocking call runs through ``to_thread`` below. The source principal
    # needs Token Creator on the target.
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
    # google-auth uses naive UTC. Give a missing expiry a short future value to
    # avoid a blocking IAM request on every call.
    expiry = creds.expiry or (datetime.now(UTC).replace(tzinfo=None) + _NO_EXPIRY_TTL)
    return creds.token, expiry.replace(tzinfo=UTC)


def vertex_fetch(impersonate: str) -> Fetch:
    """Create a provider for cloud-platform tokens as ``impersonate``.

    Reject an empty target before the first IAM request. This also covers callers
    that bypass configuration loading.
    """
    if not impersonate:
        raise ValueError(
            "the Vertex proxy needs a principal to mint as: pass the target SA email"
        )

    async def _fetch() -> tuple[str, datetime]:
        return await asyncio.to_thread(_read_adc_impersonate_token, impersonate)

    return _fetch


async def rbe_token(lifetime_s: int = 1800) -> str:
    """Return a cloud-scoped LUCI token without writing it to disk.

    The RBE proxy only requests cloud scope. Chromium Review also accepts this
    bearer and resolves it to the full user account, so keep it in the host
    process despite its short lifetime.
    """
    # Keep this operator process out of the repository edited by the box.
    with neutral_child() as child:
        return await _luci_token(lifetime_s, child)


async def _luci_token(lifetime_s: int, child: HostChild) -> str:
    proc = await asyncio.create_subprocess_exec(
        "luci-auth",
        "token",
        "-scopes-cloud",
        f"-lifetime={lifetime_s}s",
        # Prevent interactive reauthentication on the box's terminal. EOF makes
        # an expired login fail promptly, and the timeout covers other hangs.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child.env,
        cwd=child.cwd,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), _TOKEN_TIMEOUT_S)
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"luci-auth token timed out after {_TOKEN_TIMEOUT_S}s"
            " (interactive reauth needed?)"
        ) from e
    if proc.returncode:
        raise RuntimeError(
            f"luci-auth token failed (exit {proc.returncode}): "
            f"{err.decode(errors='replace').strip()[:300]}"
        )
    return out.decode(errors="replace").strip()


__all__ = ["Fetch", "rbe_token", "vertex_fetch"]
