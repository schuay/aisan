# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT

"""Validate host routes before aiohttp normalizes them."""

from yarl import URL


def literal(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty literal string")
    if any(c in value for c in ("{env:", "{file:", "${")):
        raise ValueError(f"{field}: interpolation is unsupported")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field}: control characters are unsupported")
    return value


def validate_upstream(value: object, field: str = "upstream") -> str:
    value = literal(value, field)
    if any(c.isspace() for c in value) or any(c in value for c in "\\?#"):
        raise ValueError(
            f"{field}: whitespace, backslashes, queries and fragments are unsupported"
        )
    authority = value.partition("://")[2].split("/", 1)[0]
    if "@" in authority:
        raise ValueError(f"{field}: userinfo is unsupported")
    try:
        url = URL(value)
        valid = (
            url.scheme in {"http", "https"}
            and bool(url.host)
            and url.user is None
            and url.password is None
            and url.port is not None
            and 0 < url.port <= 65535
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError(
            f"{field}: expected an absolute HTTP(S) URL with host and valid port, without userinfo"
        )
    return str(url).rstrip("/")


def upstream_origin(value: str) -> str:
    return str(URL(value).origin())
