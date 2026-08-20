# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""The failure contract for a policy check: raising is a DENY.

A transport asks a policy a yes/no question -- does this allowlist permit this
path, does it permit this authority -- and the policy is caller-supplied. An
`Allowlist` built from a config with a bad regex, a consumer's own matcher with a
KeyError in it, a predicate that touches a file that is not there: every one of
those raises where the transport expected a bool.

Left to propagate, the exception unwinds the connection handler, and both proxies
treat a torn-down connection as transport trouble rather than as a decision. So
the client retries, and a policy BUG becomes a request the policy never actually
approved. Routed through here it becomes an outage instead: the request is
refused in the protocol's own terms, with the reason the client can read.

The fail-closed rule was informed by sandbox-runtime's request-filter contract.
This implementation is independent: it turns either predicate shape into an
explicit denial and records the exception.

Logged at exception level, always, and never rate-limited: a mint failure repeats
thousands of times in a build and is rightly summarised, but a policy that raises
is a bug in the boundary and there is no volume at which it should get quieter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)


def permits(check: Callable[[], bool], *, subject: str) -> bool:
    """`check()`, with any exception answered as False.

    `subject` names what was being decided, for the log line -- the request, not
    the predicate, since the predicate is in the traceback already.
    """
    try:
        return bool(check())
    except Exception:
        log.exception("policy check raised, denying: %s", subject)
        return False


def refusal(check: Callable[[], str | None], *, subject: str) -> str | None:
    """The same contract for a check that answers WHY, not just whether.

    `None` permits; a string is the reason, and reaches the client. A policy
    that needs to name what it refused cannot express that as a bool, and
    routing it through `permits` would either discard the reason or compute it
    outside the fail-closed wrapper -- so the contract is restated here rather
    than worked around at the call site.
    """
    try:
        return check()
    except Exception:
        log.exception("policy check raised, denying: %s", subject)
        return "the sandbox proxy's body policy failed to evaluate"
