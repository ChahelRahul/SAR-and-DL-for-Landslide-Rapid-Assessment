from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.acquisition.auth import (
    EarthEngineAuthenticationError,
    discover_authentication,
    initialize_earth_engine,
    redact_sensitive_text,
)


class FakeEE:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.auth_calls = 0
        self.initialize_calls = []

    def Authenticate(self):
        self.auth_calls += 1

    def Initialize(self, **kwargs):
        self.initialize_calls.append(kwargs)
        if self.error:
            raise self.error


def _service_account(path: Path) -> None:
    path.write_text(json.dumps({
        "type": "service_account",
        "client_email": "runner@example.invalid",
        "private_key": "-----BEGIN PRIVATE KEY-----\\nSECRET\\n-----END PRIVATE KEY-----\\n",
    }))


def test_mounted_service_account_is_discovered_without_interactive_auth(tmp_path):
    key = tmp_path / "gee.json"
    _service_account(key)
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(key)}
    ee = FakeEE()
    found = initialize_earth_engine(ee, project="my-project", env=env, home=tmp_path)
    assert found.method == "application-default-credentials"
    assert ee.auth_calls == 0
    assert ee.initialize_calls == [{"project": "my-project"}]
    assert found.to_safe_dict()["credential_path"] == "<redacted>"


def test_missing_mounted_credential_has_actionable_error(tmp_path):
    with pytest.raises(EarthEngineAuthenticationError, match="Mount the file read-only"):
        discover_authentication(
            env={"GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "missing.json")},
            home=tmp_path,
        )


def test_host_earth_engine_credentials_are_discovered(tmp_path):
    path = tmp_path / ".config" / "earthengine" / "credentials"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    found = discover_authentication(env={}, home=tmp_path)
    assert found.method == "earth-engine-user-credentials"


def test_workload_identity_style_environment_uses_ambient_credentials(tmp_path):
    found = discover_authentication(env={"K_SERVICE": "sar-lra"}, home=tmp_path)
    assert found.method == "ambient-google-credentials"


def test_interactive_auth_only_when_explicit(tmp_path):
    ee = FakeEE()
    initialize_earth_engine(ee, project=None, interactive_auth=True, env={}, home=tmp_path)
    assert ee.auth_calls == 1


def test_auth_error_redacts_secret_material_and_path(tmp_path, monkeypatch):
    key = tmp_path / "top-secret.json"
    _service_account(key)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    ee = FakeEE(RuntimeError(f'private_key=SUPERSECRET path={key}'))
    with pytest.raises(EarthEngineAuthenticationError) as caught:
        initialize_earth_engine(
            ee, project="p", env={"GOOGLE_APPLICATION_CREDENTIALS": str(key)}, home=tmp_path
        )
    message = str(caught.value)
    assert "SUPERSECRET" not in message
    assert str(key) not in message
    assert "<redacted>" in message


def test_redaction_handles_json_tokens(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    text = redact_sensitive_text('{"refresh_token":"abc123","client_secret":"def456"}')
    assert "abc123" not in text
    assert "def456" not in text
