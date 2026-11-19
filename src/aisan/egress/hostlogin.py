# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Keep a host client's own login fresh without becoming a second writer.

A subscription backend reads the credential the host client stores. When the
token nears expiry, the backend runs that client's refresh path and reads the
credential again; it never redeems the refresh token itself. The client knows
its token endpoint, OAuth client, and locking, and later host sessions keep
working with what it wrote.

``ensure_fresh`` decides from a fresh read after the attempt, not from the
attempt's outcome: a timed-out client may have refreshed, and a successful one
may have authenticated from another source and left the stored token alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class CredentialRefreshError(Exception):
    """The host client could not leave behind a sufficiently fresh credential."""


class SingleFlight:
    """Share one in-progress refresh among waiting requests.

    Sharing the task deduplicates failed and successful attempts. A later wave
    may start another attempt after the current task finishes.

    The task outlives a cancelled request handler so credential updates can
    finish. It returns exceptions to avoid leaving failures unobserved when all
    waiters disconnect.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[Exception | None] | None = None

    async def join(
        self, attempt: Callable[[], Awaitable[Exception | None]]
    ) -> Exception | None:
        """Join the current attempt or start a new one."""
        async with self._lock:
            task = self._task
            if task is None or task.done():
                task = asyncio.create_task(attempt())
                self._task = task
        return await task


async def ensure_fresh[C](
    read: Callable[[], C],
    remaining: Callable[[C], float],
    refresh: Callable[[], Awaitable[Exception | None]],
    *,
    margin: float,
    floor: float,
    refusal: Callable[[float, Exception | None], str],
) -> C:
    """Return a credential, refreshing it first if it is within ``margin``.

    ``remaining`` gives a credential's lifetime in seconds. ``floor`` is the
    least lifetime the caller accepts after a refresh: preflight asks for the
    full margin, while an active session accepts any unexpired token.
    ``refusal`` explains an unusable credential without its token value.
    """
    credential = read()
    if remaining(credential) >= margin:
        return credential

    failure = await refresh()

    credential = read()
    left = remaining(credential)
    if left >= floor:
        return credential
    raise CredentialRefreshError(refusal(left, failure))
