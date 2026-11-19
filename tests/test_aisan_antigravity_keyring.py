# Copyright 2026 The aisan developers
# SPDX-License-Identifier: MIT


from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from aisan.egress.antigravity_keyring import (
    KEYRING_ATTRIBUTES,
    KeyringCredential,
    KeyringError,
    parse_login,
    read_keyring,
)
from aisan.egress.base import PreflightError
from aisan.egress.hostlogin import CredentialRefreshError

OLD_TOKEN = "ya29.old-access-token-for-tests"
NEW_TOKEN = "ya29.new-access-token-for-tests"


def _item(access_token: str = OLD_TOKEN, *, lifetime: float = 3600.0, **extra) -> str:
    """A keyring item in the shape agy 1.5.11 stores."""
    expiry = datetime.now(UTC) + timedelta(seconds=lifetime)
    return json.dumps(
        {
            "token": {
                "access_token": access_token,
                "token_type": "Bearer",
                "refresh_token": "1//refresh-token-for-tests",
                "expiry": expiry.isoformat(),
                "expires_in": 3599,
            },
            "auth_method": "consumer",
            "id_token_email": "user@example.invalid",
            **extra,
        }
    )


class _Host:
    """A fake keyring item and a fake ``agy`` that may refresh it."""

    def __init__(self, tmp_path: Path, item: str, *, refreshed: str | None = None):
        self.item = tmp_path / "keyring-item"
        self.item.write_text(item)
        self.runs = tmp_path / "agy-runs"
        self.reads = 0
        self.agy = tmp_path / "agy"
        self.agy.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time, urllib.error, urllib.request\n"
            f"item, runs = {str(self.item)!r}, {str(self.runs)!r}\n"
            "token = json.load(open(item))['token']['access_token']\n"
            "request = urllib.request.Request(\n"
            "    os.environ['CLOUD_CODE_URL'] + '/v1internal:loadCodeAssist',\n"
            "    data=b'{}', headers={'Authorization': 'Bearer ' + token})\n"
            "try:\n"
            "    urllib.request.urlopen(request, timeout=10)\n"
            "    status = 200\n"
            "except urllib.error.HTTPError as e:\n"
            "    status = e.code\n"
            "time.sleep(0.2)\n"
            "with open(runs, 'a') as f:\n"
            "    f.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
            "        'status': status, 'env': dict(os.environ)}) + '\\n')\n"
            f"refreshed = {refreshed!r}\n"
            "if refreshed is not None:\n"
            "    open(item, 'w').write(refreshed)\n"
            "print('Error: Eligibility check failed', file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        self.agy.chmod(0o755)

    def read(self) -> str:
        self.reads += 1
        return self.item.read_text()

    def credential(self, **kw) -> KeyringCredential:
        kw.setdefault("agy_command", (str(self.agy),))
        return KeyringCredential(read=self.read, **kw)

    def agy_runs(self) -> list[dict]:
        if not self.runs.exists():
            return []
        return [json.loads(line) for line in self.runs.read_text().splitlines()]


def test_the_keyring_item_agy_stores_is_parsed():
    expiry = "2026-11-19T17:27:29.888166868+01:00"
    item = json.loads(_item())
    item["token"]["expiry"] = expiry
    login = parse_login(json.dumps(item))

    assert login.access_token == OLD_TOKEN
    cet = timezone(timedelta(hours=1))
    assert (
        login.expires_at == datetime(2026, 11, 19, 17, 27, 29, 888166, cet).timestamp()
    )


@pytest.mark.parametrize(
    ("secret", "why"),
    [
        ("not json", "not JSON"),
        ("[]", "no token"),
        (json.dumps({"auth_method": "consumer"}), "no token"),
        (_item(auth_method="gcp"), "not a consumer login"),
        (_item(""), "no access token"),
        (
            json.dumps(
                {"auth_method": "consumer", "token": {"access_token": OLD_TOKEN}}
            ),
            "no readable expiry",
        ),
        (
            json.dumps(
                {
                    "auth_method": "consumer",
                    "token": {"access_token": OLD_TOKEN, "expiry": "soon"},
                }
            ),
            "no readable expiry",
        ),
    ],
)
def test_an_unusable_keyring_item_is_refused_without_its_token(secret, why):
    with pytest.raises(KeyringError, match=why) as error:
        parse_login(secret)
    assert OLD_TOKEN not in str(error.value)


def _fake_python(tmp_path: Path, body: str) -> str:
    path = tmp_path / "python3"
    path.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    path.chmod(0o755)
    return str(path)


def test_the_reader_passes_the_attributes_and_returns_the_secret(tmp_path):
    python = _fake_python(
        tmp_path,
        "assert sys.argv[1] == '-I' and sys.argv[2] == '-c', sys.argv\n"
        "compile(sys.argv[3], 'reader', 'exec')\n"
        "sys.stdout.write('|'.join(sys.argv[4:]))",
    )
    assert read_keyring(python=python) == "service=gemini|username=antigravity"
    assert KEYRING_ATTRIBUTES == (("service", "gemini"), ("username", "antigravity"))


@pytest.mark.parametrize(
    ("body", "reason", "fix"),
    [
        ("sys.exit(3)", "no Antigravity login", "sign in"),
        ("sys.exit(4)", "keyring is locked", "unlock"),
        ("sys.exit(5)", "no libsecret bindings", "python-gobject"),
        ("sys.stderr.write('bus is gone'); sys.exit(1)", "bus is gone", "unlock"),
    ],
)
def test_a_failed_keyring_read_says_what_to_fix(tmp_path, body, reason, fix):
    with pytest.raises(KeyringError, match=reason) as error:
        read_keyring(python=_fake_python(tmp_path, body))
    assert fix in error.value.fix


def test_a_missing_system_interpreter_is_reported(tmp_path):
    with pytest.raises(KeyringError, match="cannot run") as error:
        read_keyring(python=str(tmp_path / "absent"))
    assert "python-gobject" in error.value.fix


async def test_a_fresh_token_is_used_without_running_agy_or_rereading(tmp_path):
    host = _Host(tmp_path, _item())
    credential = host.credential()

    assert [await credential.token() for _ in range(3)] == [OLD_TOKEN] * 3
    await credential.check("antigravity")
    assert host.reads == 1
    assert host.agy_runs() == []


async def test_a_token_in_the_clients_window_is_refreshed_by_host_agy(
    tmp_path, monkeypatch, caplog
):
    host = _Host(tmp_path, _item(lifetime=8), refreshed=_item(NEW_TOKEN))
    monkeypatch.setenv("GEMINI_API_KEY", "would-select-another-provider")
    monkeypatch.setenv("AGY_ADC_AUTH", "1")
    credential = host.credential()

    with caplog.at_level(logging.DEBUG):
        tokens = await asyncio.gather(*(credential.token() for _ in range(5)))

    assert tokens == [NEW_TOKEN] * 5
    [run] = host.agy_runs()
    assert run["argv"] == ["models"]
    # The sink stood in for Cloud Code and answered without forwarding.
    assert run["env"]["CLOUD_CODE_URL"].startswith("http://127.0.0.1:")
    assert run["status"] == 404
    assert "GEMINI_API_KEY" not in run["env"]
    assert "AGY_ADC_AUTH" not in run["env"]
    assert os.listdir(Path(run["cwd"]).parent) == []
    assert Path(run["cwd"]) != Path.cwd()
    # The sink received the real token and must not have logged it.
    assert OLD_TOKEN not in caplog.text
    assert NEW_TOKEN not in caplog.text

    assert await credential.token() == NEW_TOKEN
    assert len(host.agy_runs()) == 1


async def test_an_unexpired_token_agy_left_alone_is_still_used(tmp_path):
    host = _Host(tmp_path, _item(lifetime=8))
    credential = host.credential()

    assert await credential.token() == OLD_TOKEN
    assert len(host.agy_runs()) == 1


async def test_an_expired_token_agy_did_not_refresh_is_refused(tmp_path):
    host = _Host(tmp_path, _item(lifetime=-30))
    credential = host.credential()

    with pytest.raises(CredentialRefreshError, match="did not refresh") as error:
        await credential.token()
    assert OLD_TOKEN not in str(error.value)
    with pytest.raises(PreflightError, match=r"expired 3[0-9]s ago") as preflight:
        await credential.check("antigravity")
    assert "sign in" in preflight.value.fix
    assert len(host.agy_runs()) == 2


async def test_an_expired_token_is_refreshed_before_preflight_passes(tmp_path):
    host = _Host(tmp_path, _item(lifetime=-30), refreshed=_item(NEW_TOKEN))
    credential = host.credential()

    await credential.check("antigravity")
    assert await credential.token() == NEW_TOKEN
    assert len(host.agy_runs()) == 1


async def test_a_missing_agy_is_named_in_the_refusal(tmp_path):
    host = _Host(tmp_path, _item(lifetime=-30))
    credential = host.credential(agy_command=(str(tmp_path / "absent-agy"),))

    with pytest.raises(CredentialRefreshError, match="could not run host agy"):
        await credential.token()


async def test_a_keyring_failure_reaches_preflight_with_its_fix(tmp_path):
    def locked() -> str:
        raise KeyringError("the host keyring is locked", "unlock it")

    credential = KeyringCredential(read=locked)
    with pytest.raises(PreflightError, match="keyring is locked") as error:
        await credential.check("antigravity")
    assert error.value.fix == "unlock it"
    with pytest.raises(KeyringError):
        await credential.token()


async def test_a_host_relogin_is_read_when_the_cached_token_nears_expiry(
    tmp_path, monkeypatch
):
    host = _Host(tmp_path, _item(lifetime=60))
    credential = host.credential()
    assert await credential.token() == OLD_TOKEN

    host.item.write_text(_item(NEW_TOKEN))
    assert await credential.token() == OLD_TOKEN

    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 55)
    assert await credential.token() == NEW_TOKEN
    assert host.reads == 2
    assert host.agy_runs() == []
