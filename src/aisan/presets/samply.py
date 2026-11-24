# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Grant `samply record` its task-port handshake in a Darwin box.

samply launches the profiled process with a preloaded library that sends the
process's task port back over a one-shot `ipc-channel` server. The server is
registered with the bootstrap server under a random
`org.rust-lang.ipc-channel.` name, which the box can otherwise neither register
nor look up. The grant permits both for that prefix and adds no file access.

The prefix is shared by the whole login session. Two boxes holding the grant
can therefore agree on a name and exchange Mach ports, and a box can connect to
a host process's one-shot server if it learns the name.
"""

from __future__ import annotations

from ..sandbox import host_platform
from ..spec import SAMPLY, Grant


def samply_grant(*, platform: str | None = None) -> Grant:
    """Permit `ipc-channel` bootstrap names; empty off Darwin."""
    if (host_platform() if platform is None else platform) != "darwin":
        return Grant()
    return Grant(mach_profiles=(SAMPLY,))
