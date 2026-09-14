from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

AcquisitionProvider = Literal[
    "auto", "planetary-computer", "planetary-computer-grd",
    "planetary-computer-rtc", "earth-engine"
]


def _earth_engine_credential_present() -> bool:
    explicit = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and Path(explicit).expanduser().is_file():
        return True
    return (Path.home() / ".config" / "earthengine" / "credentials").is_file() or (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    ).is_file()


def resolve_provider(requested: AcquisitionProvider | str | None) -> Literal[
    "planetary-computer-grd", "planetary-computer-rtc", "earth-engine"
]:
    """Resolve the remote acquisition backend.

    `planetary-computer` is a backwards-compatible alias for the keyless GRD path.
    `auto` defaults to Planetary Computer GRD because STAC discovery and anonymous
    SAS token issuance do not require a user subscription key. Deployments can
    override this with SAR_LRA_ACQUISITION_PROVIDER.
    """
    value = (requested or "auto").strip().lower()
    allowed = {"auto", "planetary-computer", "planetary-computer-grd", "planetary-computer-rtc", "earth-engine"}
    if value not in allowed:
        raise ValueError(f"unsupported acquisition provider: {value}")
    if value == "planetary-computer":
        return "planetary-computer-grd"
    if value != "auto":
        return value  # type: ignore[return-value]

    configured = os.getenv("SAR_LRA_ACQUISITION_PROVIDER", "auto").strip().lower()
    if configured == "planetary-computer":
        return "planetary-computer-grd"
    if configured in {"planetary-computer-grd", "planetary-computer-rtc", "earth-engine"}:
        return configured  # type: ignore[return-value]
    if configured not in {"", "auto"}:
        raise ValueError("SAR_LRA_ACQUISITION_PROVIDER must be auto, planetary-computer-grd, planetary-computer-rtc, or earth-engine")

    return "planetary-computer-grd"


def credential_summary() -> dict[str, bool | str]:
    return {
        "configured_provider": os.getenv("SAR_LRA_ACQUISITION_PROVIDER", "auto"),
        "planetary_computer_grd_requires_user_key": False,
        "planetary_computer_rtc_key_present": bool(os.getenv("PC_SDK_SUBSCRIPTION_KEY")),
        "earth_engine_file_credentials_detected": _earth_engine_credential_present(),
    }
