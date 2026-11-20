# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Shared host-side reader helpers for the macOS Login Keychain."""

from __future__ import annotations

import subprocess

SECURITY_BIN = "/usr/bin/security"
SECURITY_EXIT_LOCKED = 36
SECURITY_EXIT_MISSING = 44


class KeychainLockedError(PermissionError):
    """The macOS Login Keychain is locked (`security` exited 36)."""


def decode_security_secret(raw: str) -> str:
    """Decode `security find-generic-password -w` plain or hex-encoded output."""
    stripped = raw.strip()
    if stripped and not stripped.startswith("{") and len(stripped) % 2 == 0:
        try:
            return bytes.fromhex(stripped).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass
    return stripped


def query_generic_password(
    *,
    service: str,
    account: str | None = None,
    security: str = SECURITY_BIN,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    """Run `security find-generic-password -w` without interactive prompts."""
    cmd = [security, "find-generic-password", "-s", service]
    if account is not None:
        cmd.extend(("-a", account))
    cmd.append("-w")
    return subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
