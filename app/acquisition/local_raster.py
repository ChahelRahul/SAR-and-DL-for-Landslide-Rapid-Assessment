from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EXPECTED_BAND_ORDER


@dataclass(slots=True)
class RasterInput:
    path: Path
    data: np.ndarray
    profile: dict[str, Any]
    transform: Any
    crs: Any
    metadata: dict[str, Any]


def read_sentinel1_stack(path: str | Path, *, require_contract_tags: bool = False) -> RasterInput:
    """Read and validate a prepared four-band Sentinel-1 raster.

    Importing this module does not require Earth Engine. Rasterio is loaded only
    when the reader is called.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Prepared raster inference requires the 'geo' extra") from exc

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input raster does not exist: {path}")
    with rasterio.open(path) as src:
        if src.count != len(EXPECTED_BAND_ORDER):
            raise ValueError(
                f"Expected four bands {EXPECTED_BAND_ORDER}; found {src.count} in {path}"
            )
        if src.crs is None:
            raise ValueError(f"Input raster has no CRS: {path}")
        if src.width < 1 or src.height < 1:
            raise ValueError(f"Input raster has invalid dimensions: {path}")
        tags = src.tags()
        descriptions = tuple(value or "" for value in src.descriptions)
        tagged_order = tuple(tags.get("input_band_order", "").split(","))
        if tagged_order == ("",):
            tagged_order = ()
        observed_order = descriptions if all(descriptions) else tagged_order
        if require_contract_tags and observed_order != EXPECTED_BAND_ORDER:
            raise ValueError(
                "Intermediate raster must declare the band order "
                f"{EXPECTED_BAND_ORDER}; found {observed_order or 'none'}"
            )
        if observed_order and observed_order != EXPECTED_BAND_ORDER:
            raise ValueError(
                f"Input band order must be {EXPECTED_BAND_ORDER}; found {observed_order}"
            )
        data = np.moveaxis(src.read(), 0, 2).astype(np.float32, copy=False)
        if not np.isfinite(data).all():
            raise ValueError(f"Input contains NaN or infinite values: {path}")
        metadata = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "band_order": list(observed_order or EXPECTED_BAND_ORDER),
            "source": tags.get("source", "prepared-raster"),
            "cache_key": tags.get("cache_key"),
        }
        return RasterInput(path, data, src.profile.copy(), src.transform, src.crs, metadata)
