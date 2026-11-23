# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Grant Security.framework trust lookups to a shared-network Darwin box.

Some TLS clients enumerate the system trust store through the legacy Keychain
API, which fails without `securityd`. The grant permits that one Mach service
and no file access: the Keychain databases stay outside the box, so their items
remain unreadable. `Box` refuses any mount that would expose them.
"""

from __future__ import annotations

from ..sandbox import host_platform
from ..spec import SECURITY_SERVER, Grant


def security_server_grant(*, platform: str | None = None) -> Grant:
    """Permit `com.apple.SecurityServer` lookups; empty off Darwin."""
    if (host_platform() if platform is None else platform) != "darwin":
        return Grant()
    return Grant(mach_profiles=(SECURITY_SERVER,))
