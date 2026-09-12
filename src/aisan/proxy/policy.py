# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Treat an exception from a policy check as a denial.

A transport expects a policy predicate to return a decision. Caller-supplied
predicates can instead raise because of invalid configuration, bugs, or missing
files.

Letting the exception unwind the connection handler looks like a transport
failure, which clients retry. Converting the exception to a protocol refusal
keeps an unevaluated request from reaching the upstream service.

These helpers adapt each predicate shape to that fail-closed behavior and log
the exception.

Policy exceptions aren't rate-limited because each one signals a bug in the
security boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)


def permits(check: Callable[[], bool], *, subject: str) -> bool:
    """Run a boolean check, returning ``False`` if it raises.

    ``subject`` identifies the request in the log. The traceback identifies the
    predicate.
    """
    try:
        return bool(check())
    except Exception:
        log.exception("policy check raised, denying: %s", subject)
        return False


def refusal(check: Callable[[], str | None], *, subject: str) -> str | None:
    """Run a refusal check, returning a failure message if it raises.

    ``None`` permits the request. A string explains the denial to the client.
    """
    try:
        return check()
    except Exception:
        log.exception("policy check raised, denying: %s", subject)
        return "the sandbox proxy's body policy failed to evaluate"


def decision(check: Callable[[], str | None], *, subject: str) -> str | None:
    """Run a matching check, returning ``None`` if it raises.

    ``None`` denies the request. A string identifies the matching request shape
    and selects its policy. Evaluating once keeps the match and policy aligned.
    """
    try:
        return check()
    except Exception:
        log.exception("policy check raised, denying: %s", subject)
        return None
