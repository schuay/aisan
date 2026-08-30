# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path


def known_credential_paths() -> tuple[Path, ...]:
    """Every default location a backend credential lives at on this host.

    For guards that must refuse a mount covering ANY backend's credential, not
    just the ones the current box's egress names: a bind computed for one
    launcher can expose another client's key store, and `credential_exposure`
    over this box's backends alone would never look at it. Recomputed per call
    because the defaults depend on the environment (HOME, XDG_*, CODEX_HOME).

    Only file-backed credentials appear here; the Vertex and RBE backends mint
    in memory and have nothing on disk to name.
    """
    from .anthropic import default_credentials as anthropic_credentials
    from .openai_compat import default_credentials as openai_compat_credentials
    from .openai_responses import default_credentials as openai_responses_credentials

    return (
        anthropic_credentials(),
        openai_compat_credentials(),
        openai_responses_credentials(),
    )
