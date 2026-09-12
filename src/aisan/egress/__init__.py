# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path


def known_credential_paths() -> tuple[Path, ...]:
    """Return every default backend credential path for the current environment.

    Mount guards check all clients because one launcher's bind can expose another
    client's store. Vertex and RBE mint transient tokens from on-disk stores, so
    those source stores count as credentials. Recompute paths because HOME, XDG,
    and client-specific environment variables may change.
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
