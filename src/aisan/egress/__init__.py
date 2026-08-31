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

    Every backend with a file, which is every backend but one: the Vertex and
    RBE routes mint their tokens in memory, but each mints FROM a store on disk
    (ADC, luci-auth's), and a box able to read that store can mint the same
    token for itself. Only the Anthropic API-key kind has nothing to name, and
    it holds its key in the process.

    Importing the backends is what keeps this list from being a second spelling
    of paths their own modules define. It is not the import cost it looks like:
    they share the proxy stack, so after the first the rest are free.
    """
    from .anthropic import default_credentials as anthropic_credentials
    from .openai_compat import default_credentials as openai_compat_credentials
    from .openai_responses import default_credentials as openai_responses_credentials
    from .reapi import LUCI_STORE
    from .vertex import default_credentials as vertex_credentials

    return (
        anthropic_credentials(),
        openai_compat_credentials(),
        openai_responses_credentials(),
        *vertex_credentials(),
        LUCI_STORE,
    )
