from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

GOOGLE_CREDENTIAL_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
EARTHENGINE_CREDENTIAL_PATH = Path(".config/earthengine/credentials")


class EarthEngineAuthenticationError(RuntimeError):
    """Raised when Earth Engine credentials are missing or unusable."""


@dataclass(frozen=True, slots=True)
class AuthDiscovery:
    method: str
    credential_path: Path | None = None

    def to_safe_dict(self) -> dict[str, str | None]:
        return {"method": self.method, "credential_path": "<redacted>" if self.credential_path else None}


def redact_sensitive_text(value: object) -> str:
    """Remove likely credential material and credential file paths from diagnostics."""
    text = str(value)
    text = re.sub(
        r'("?(?:private_key|private_key_id|client_secret|refresh_token|access_token|token)"?\s*[:=]\s*)[^,}\n]+',
        r'\1<redacted>', text, flags=re.IGNORECASE,
    )
    env_path = os.environ.get(GOOGLE_CREDENTIAL_ENV)
    if env_path:
        text = text.replace(env_path, "<redacted-credential-path>")
    return text


def discover_authentication(env: Mapping[str, str] | None = None, home: Path | None = None) -> AuthDiscovery:
    env = os.environ if env is None else env
    adc_path = env.get(GOOGLE_CREDENTIAL_ENV)
    if adc_path:
        path = Path(adc_path).expanduser()
        if not path.is_file():
            raise EarthEngineAuthenticationError(
                f"{GOOGLE_CREDENTIAL_ENV} is set but the credential file does not exist. "
                "Mount the file read-only into the container and point the environment variable to it."
            )
        return AuthDiscovery("application-default-credentials", path)

    root = Path.home() if home is None else home
    ee_credentials = root / EARTHENGINE_CREDENTIAL_PATH
    if ee_credentials.is_file():
        return AuthDiscovery("earth-engine-user-credentials", ee_credentials)

    # On GCP, google.auth.default() can resolve attached service accounts or
    # Workload Identity without a local file. We intentionally do not probe the
    # metadata server here; ee.Initialize() / google-auth performs that lookup.
    if any(env.get(name) for name in ("K_SERVICE", "CLOUD_RUN_JOB", "GCE_METADATA_HOST", "GOOGLE_CLOUD_PROJECT")):
        return AuthDiscovery("ambient-google-credentials")

    return AuthDiscovery("automatic")


def validate_service_account_file(path: Path) -> None:
    """Fail early for malformed mounted service-account JSON without logging its contents."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EarthEngineAuthenticationError(
            "Mounted Google credential JSON cannot be read or parsed. Check the read-only secret mount."
        ) from exc
    if payload.get("type") == "service_account":
        required = ("client_email", "private_key")
        if any(not payload.get(name) for name in required):
            raise EarthEngineAuthenticationError(
                "Mounted service-account JSON is missing required fields."
            )


def initialize_earth_engine(
    ee: Any,
    *,
    project: str | None,
    interactive_auth: bool = False,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> AuthDiscovery:
    """Initialize Earth Engine from ambient credentials, never embedding secrets.

    `interactive_auth=True` is retained for local developer use only. Container
    deployments should mount credentials or use ADC / Workload Identity.
    """
    discovery = discover_authentication(env=env, home=home)
    if discovery.credential_path and discovery.method == "application-default-credentials":
        validate_service_account_file(discovery.credential_path)
    try:
        if interactive_auth:
            ee.Authenticate()
        ee.Initialize(**({"project": project} if project else {}))
    except Exception as exc:  # Earth Engine exposes several auth exception types.
        safe = redact_sensitive_text(exc)
        raise EarthEngineAuthenticationError(
            "Earth Engine authentication/initialization failed. "
            "Provide a registered Google Cloud project with --project and one of: "
            "(1) GOOGLE_APPLICATION_CREDENTIALS pointing to a read-only mounted service-account JSON, "
            "(2) Application Default Credentials, (3) a mounted ~/.config/earthengine/credentials file, "
            "or (4) cloud Workload Identity/attached service-account credentials. "
            f"Underlying error: {safe}"
        ) from exc
    return discovery
