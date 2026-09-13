from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EXPECTED_BAND_ORDER, MODEL_NAME, MODEL_VERSION


def _base_tags(
    *,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
) -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "configured_model_version": str(effective_config["model"]["version"]),
        "orbit": orbit,
        "relative_orbit": "" if relative_orbit is None else str(relative_orbit),
        "weights_sha256": weights_sha256,
        "input_band_order": ",".join(EXPECTED_BAND_ORDER),
        "effective_configuration": json.dumps(effective_config, separators=(",", ":")),
    }


def write_probability_raster(
    path: str | Path,
    probabilities: np.ndarray,
    profile: dict[str, Any],
    *,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
) -> Path:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Raster output requires the 'geo' extra") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    nodata = np.float32(-9999.0)
    output_profile.update(dtype="float32", count=1, compress="lzw", nodata=float(nodata), tiled=True)
    rows_per_chunk = int(effective_config["processing"].get("output_rows_per_chunk", 1024))
    with rasterio.open(path, "w", **output_profile) as dst:
        for y0 in range(0, probabilities.shape[0], rows_per_chunk):
            y1 = min(y0 + rows_per_chunk, probabilities.shape[0])
            block = np.asarray(probabilities[y0:y1], dtype=np.float32)
            disk_block = np.where(np.isfinite(block), block, nodata).astype(np.float32, copy=False)
            dst.write(disk_block, 1, window=((y0, y1), (0, probabilities.shape[1])))
        dst.update_tags(
            **_base_tags(
                orbit=orbit, relative_orbit=relative_orbit, weights_sha256=weights_sha256,
                effective_config=effective_config,
            ),
            output_type="probability_surface",
            probability_semantics="maximum model-window probability covering each valid pixel",
            aggregation=str(effective_config["processing"]["probability_aggregation"]),
            threshold_applied="false",
            write_mode="row_chunked",
        )
    return path


def write_detection_mask(
    path: str | Path,
    mask: np.ndarray,
    profile: dict[str, Any],
    *,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
) -> Path:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Raster output requires the 'geo' extra") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(dtype="uint8", count=1, compress="lzw", nodata=0, tiled=True)
    rows_per_chunk = int(effective_config["processing"].get("output_rows_per_chunk", 1024))
    with rasterio.open(path, "w", **output_profile) as dst:
        for y0 in range(0, mask.shape[0], rows_per_chunk):
            y1 = min(y0 + rows_per_chunk, mask.shape[0])
            dst.write(np.asarray(mask[y0:y1], dtype=np.uint8), 1, window=((y0, y1), (0, mask.shape[1])))
        dst.update_tags(
            **_base_tags(
                orbit=orbit, relative_orbit=relative_orbit, weights_sha256=weights_sha256,
                effective_config=effective_config,
            ),
            output_type="binary_detection_mask",
            probability_threshold=str(effective_config["model"]["probability_threshold"]),
            derived_from="probability_surface",
            write_mode="row_chunked",
        )
    return path


def threshold_probability_raster(
    probability_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float,
) -> Path:
    """Create a binary mask from a saved probability surface, without inference."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Raster thresholding requires the 'geo' extra") from exc

    probability_path = Path(probability_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(probability_path) as src:
        probability = src.read(1, masked=True)
        mask = ((probability >= threshold) & ~np.ma.getmaskarray(probability)).astype(np.uint8)
        profile = src.profile.copy()
        tags = src.tags()
        profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)
            tags.update(
                output_type="binary_detection_mask",
                probability_threshold=str(threshold),
                derived_from=str(probability_path.name),
            )
            dst.update_tags(**tags)
    return output_path
