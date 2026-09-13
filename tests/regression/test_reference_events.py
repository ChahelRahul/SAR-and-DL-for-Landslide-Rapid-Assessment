from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.acquisition.local_raster import read_sentinel1_stack
from app.config import AppConfig, ModelConfig
from app.inference.predict import predict_stack
from app.postprocessing.vector import detection_geodataframe

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "reference"


class DeterministicReferenceModel:
    """Stable model surrogate for preprocessing/inference regression tests."""
    def predict(self, patches, batch_size=512, verbose=0):
        x = np.asarray(patches, dtype=np.float32)
        signal = x[..., 2:].mean(axis=(1, 2, 3))
        scores = 1.0 / (1.0 + np.exp(-signal * 4.0))
        return scores[:, None].astype(np.float32)


@pytest.mark.parametrize("case", ["haiti-2021-ascending", "sumatra-2022-descending"])
def test_reference_event_regression(case):
    expected = json.loads((FIX / f"{case}.expected.json").read_text())
    orbit = expected["orbit"]
    cfg = AppConfig(model=ModelConfig(orbit=orbit), output_dir=Path("unused"))
    roi = json.loads((FIX / f"{case}.roi.geojson").read_text())
    raster = read_sentinel1_stack(FIX / f"{case}.tif", config=cfg, roi_geojson=roi)
    pred = predict_stack(DeterministicReferenceModel(), raster.data, cfg.model, cfg.processing,
                         valid_mask=raster.valid_mask)

    values = pred.probability_surface[np.isfinite(pred.probability_surface)]
    observed = {
        "width": raster.metadata["width"],
        "height": raster.metadata["height"],
        "band_order": raster.metadata["band_order"],
        "patch_count": len(pred.scored_windows),
        "probability": {
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())
        },
        "candidate_count": len(pred.candidate_windows),
        "iou_retained_count": len(pred.iou_selected_windows),
        "positive_pixels": int(pred.mask.sum()),
    }

    assert observed["width"] == expected["input_metadata"]["width"]
    assert observed["height"] == expected["input_metadata"]["height"]
    assert observed["band_order"] == expected["input_metadata"]["band_order"]
    assert observed["patch_count"] == expected["patch_count"]
    assert observed["candidate_count"] == expected["candidate_count"]
    assert observed["iou_retained_count"] == expected["iou_retained_count"]
    assert observed["positive_pixels"] == expected["positive_pixels"]
    for key in ("min", "max", "mean"):
        assert observed["probability"][key] == pytest.approx(
            expected["probability_statistics"][key], abs=expected["tolerances"]["cpu_absolute"]
        )

    gdf = detection_geodataframe(
        pred.mask, pred.probability_surface,
        transform=raster.transform, crs=raster.crs, orbit=orbit, relative_orbit=None,
        weights_sha256="fixture-model", effective_config=cfg.to_dict(), roi_geojson=roi,
    )
    bounds = list(map(float, gdf.total_bounds)) if not gdf.empty else None
    assert len(gdf) == expected["vector_feature_count"]
    if bounds is not None:
        assert bounds == pytest.approx(expected["output_geometry_bounds"], abs=0.01)


def test_reference_fixture_manifest_declares_redistributable_data():
    manifest = json.loads((FIX / "source-manifest.json").read_text())
    assert {x["orbit"] for x in manifest["fixtures"]} == {"ASCENDING", "DESCENDING"}
    assert all(x["fixture_license"] == "MIT" for x in manifest["fixtures"])
