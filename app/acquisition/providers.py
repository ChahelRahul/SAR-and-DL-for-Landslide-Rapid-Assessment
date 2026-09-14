from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

AcquisitionProvider = Literal["auto", "planetary-computer", "earth-engine"]


def _earth_engine_credential_present() -> bool:
    explicit = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and Path(explicit).expanduser().is_file():
        return True
    ee_credentials = Path.home() / ".config" / "earthengine" / "credentials"
    adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return ee_credentials.is_file() or adc.is_file()


def resolve_provider(requested: AcquisitionProvider | str | None) -> Literal["planetary-computer", "earth-engine"]:
    """Resolve a remote imagery provider without accepting secrets in request bodies.

    Precedence for ``auto``:
      1. SAR_LRA_ACQUISITION_PROVIDER, when explicitly set to a concrete provider.
      2. Planetary Computer when PC_SDK_SUBSCRIPTION_KEY is present.
      3. Earth Engine when a conventional local/ADC credential file is detectable.

    Cloud-attached service accounts / workload identity cannot always be detected from
    the filesystem, so those deployments should set SAR_LRA_ACQUISITION_PROVIDER=earth-engine
    or send provider="earth-engine" explicitly.
    """
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "planetary-computer", "earth-engine"}:
        raise ValueError(f"unsupported acquisition provider: {value}")
    if value != "auto":
        return value  # type: ignore[return-value]

    configured = os.getenv("SAR_LRA_ACQUISITION_PROVIDER", "auto").strip().lower()
    if configured in {"planetary-computer", "earth-engine"}:
        return configured  # type: ignore[return-value]
    if configured not in {"", "auto"}:
        raise ValueError("SAR_LRA_ACQUISITION_PROVIDER must be auto, planetary-computer, or earth-engine")

    if os.getenv("PC_SDK_SUBSCRIPTION_KEY"):
        return "planetary-computer"
    if _earth_engine_credential_present():
        return "earth-engine"

    raise RuntimeError(
        "No remote acquisition credentials were detected. Provide PC_SDK_SUBSCRIPTION_KEY "
        "for Planetary Computer, configure Earth Engine credentials and select earth-engine, "
        "or use /v1/predict-raster / 'sar-lra predict-raster' for credential-free inference."
    )


def credential_summary() -> dict[str, bool | str]:
    """Return non-secret provider readiness information for diagnostics/readiness output."""
    return {
        "configured_provider": os.getenv("SAR_LRA_ACQUISITION_PROVIDER", "auto"),
        "planetary_computer_key_present": bool(os.getenv("PC_SDK_SUBSCRIPTION_KEY")),
        "earth_engine_file_credentials_detected": _earth_engine_credential_present(),
    }
