from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.config import AppConfig

ROI = {
    "type": "Polygon",
    "coordinates": [[[85.30, 27.65], [85.40, 27.65], [85.40, 27.75], [85.30, 27.75], [85.30, 27.65]]],
}


def test_planetary_provider_is_import_safe():
    import app.acquisition.planetary_computer as provider
    assert provider.COLLECTION == "sentinel-1-rtc"


def test_planetary_search_requires_subscription_key(monkeypatch):
    import app.acquisition.planetary_computer as provider
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    class DummyPC:
        class settings:
            @staticmethod
            def set_subscription_key(_): pass
    class DummyClient:
        pass
    monkeypatch.setattr(provider, "_deps", lambda: (DummyPC, DummyClient, None, None, None, None, None, None, None))
    with pytest.raises(RuntimeError, match="PC_SDK_SUBSCRIPTION_KEY"):
        provider.search_items(
            roi_geojson=ROI, event_date=date(2025, 8, 15), orbit="ASCENDING", config=AppConfig()
        )


def test_pipeline_mode_includes_planetary_computer(tmp_path, monkeypatch):
    from app.pipeline import run_planetary_computer
    from app.schemas import PlanetaryComputerRequest, PipelineResult

    monkeypatch.setattr(
        "app.acquisition.planetary_computer.acquire_intermediate_raster",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop-before-inference")),
    )
    request = PlanetaryComputerRequest(
        request_id="pc-test", orbit="ASCENDING", event_date=date(2025, 8, 15),
        weights_path=Path("weights.hdf5"), roi_geojson=ROI, cache_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="stop-before-inference"):
        run_planetary_computer(request, AppConfig(output_dir=tmp_path))


def test_auto_provider_prefers_planetary_computer_key(monkeypatch):
    from app.acquisition.providers import resolve_provider
    monkeypatch.setenv("PC_SDK_SUBSCRIPTION_KEY", "secret")
    monkeypatch.delenv("SAR_LRA_ACQUISITION_PROVIDER", raising=False)
    assert resolve_provider("auto") == "planetary-computer"


def test_explicit_earth_engine_does_not_require_pc_key(monkeypatch):
    from app.acquisition.providers import resolve_provider
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    assert resolve_provider("earth-engine") == "earth-engine"


def test_auto_provider_can_be_forced_by_environment(monkeypatch):
    from app.acquisition.providers import resolve_provider
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    monkeypatch.setenv("SAR_LRA_ACQUISITION_PROVIDER", "earth-engine")
    assert resolve_provider("auto") == "earth-engine"
