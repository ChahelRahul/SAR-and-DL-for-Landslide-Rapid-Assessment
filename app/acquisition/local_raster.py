from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EXPECTED_BAND_ORDER, AppConfig


class RasterValidationError(ValueError):
    """Raised when a Sentinel-1 intermediate raster is incompatible."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__("Sentinel-1 raster validation failed:\n- " + "\n- ".join(errors))


@dataclass(slots=True)
class RasterInput:
    path: Path
    data: np.ndarray
    profile: dict[str, Any]
    transform: Any
    crs: Any
    metadata: dict[str, Any]


def read_sentinel1_stack(
    path: str | Path,
    *,
    config: AppConfig,
    roi_geojson: dict[str, Any] | None = None,
    require_contract_tags: bool = False,
) -> RasterInput:
    """Read, validate, and normalise a four-band Sentinel-1 intermediate raster."""
    try:
        import rasterio
        from rasterio.features import bounds as geometry_bounds
        from rasterio.warp import transform_geom
    except ImportError as exc:
        raise RuntimeError("Prepared raster inference requires the 'geo' extra") from exc

    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        raise RasterValidationError([f"Input raster does not exist: {path}"])

    with rasterio.open(path) as src:
        tags = src.tags()
        descriptions = tuple(value or "" for value in src.descriptions)
        tagged_order = tuple(x.strip() for x in tags.get("input_band_order", "").split(",") if x.strip())
        observed_order = descriptions if all(descriptions) else tagged_order

        if src.count != len(EXPECTED_BAND_ORDER):
            errors.append(f"expected 4 bands {EXPECTED_BAND_ORDER}, found {src.count}")
        if not observed_order:
            errors.append("band order is unknown; set band descriptions or input_band_order metadata")
        elif observed_order != EXPECTED_BAND_ORDER:
            errors.append(f"band order must be {EXPECTED_BAND_ORDER}, found {observed_order}")
        if require_contract_tags and not tagged_order:
            errors.append("Earth Engine intermediate raster must include input_band_order metadata")
        if src.crs is None:
            errors.append("CRS is missing")
        if src.transform is None or src.transform.is_identity:
            errors.append("geotransform is missing or identity")
        if src.width < config.processing.tile_size or src.height < config.processing.tile_size:
            errors.append(
                f"raster is {src.width}x{src.height}; minimum is "
                f"{config.processing.tile_size}x{config.processing.tile_size} for one model window"
            )

        resolution = (abs(src.transform.a), abs(src.transform.e))
        if src.crs is not None:
            if src.crs.is_projected:
                expected = float(config.imagery.scale_m)
                tolerance = expected * config.processing.resolution_tolerance
                if any(abs(value - expected) > tolerance for value in resolution):
                    errors.append(
                        f"pixel resolution {resolution} is outside {config.processing.resolution_tolerance:.0%} "
                        f"tolerance of {expected} m"
                    )
            else:
                warnings.append("geographic CRS: metre-scale resolution could not be verified exactly")

        raster_bounds = src.bounds
        roi_coverage = None
        if roi_geojson is not None and src.crs is not None:
            geometry = roi_geojson.get("geometry", roi_geojson)
            try:
                projected = transform_geom("EPSG:4326", src.crs, geometry, precision=8)
                left, bottom, right, top = geometry_bounds(projected)
                ix0, iy0 = max(left, raster_bounds.left), max(bottom, raster_bounds.bottom)
                ix1, iy1 = min(right, raster_bounds.right), min(top, raster_bounds.top)
                intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                roi_area = max(0.0, right - left) * max(0.0, top - bottom)
                roi_coverage = intersection / roi_area if roi_area else 0.0
                if intersection <= 0:
                    errors.append("raster does not intersect the supplied ROI")
                elif roi_coverage < 1.0:
                    warnings.append(
                        f"raster partially covers ROI ({roi_coverage:.1%}); inference is limited to the covered area"
                    )
            except Exception as exc:
                errors.append(f"ROI intersection check failed: {exc}")

        data_ma = src.read(masked=True).astype(np.float32)
        nodata_pixels = int(np.ma.getmaskarray(data_ma).sum())
        total_values = int(data_ma.size)
        nodata_fraction = nodata_pixels / total_values if total_values else 1.0
        if nodata_fraction >= 1.0:
            errors.append("all raster values are NoData")
        elif nodata_fraction > config.processing.max_nodata_fraction:
            errors.append(
                f"NoData fraction {nodata_fraction:.1%} exceeds allowed "
                f"{config.processing.max_nodata_fraction:.1%}"
            )

        if errors:
            raise RasterValidationError(errors, warnings)

        # Fill sparse NoData per band using the valid-band median, preserving a dense TensorFlow input.
        filled = np.empty(data_ma.shape, dtype=np.float32)
        for band in range(data_ma.shape[0]):
            values = data_ma[band]
            valid = values.compressed()
            fill = float(np.median(valid)) if valid.size else 0.0
            filled[band] = values.filled(fill)
        data = np.moveaxis(filled, 0, 2)

        band_ranges: dict[str, dict[str, float]] = {}
        for index, name in enumerate(EXPECTED_BAND_ORDER):
            values = data[..., index]
            low, high = float(np.percentile(values, 0.1)), float(np.percentile(values, 99.9))
            band_ranges[name] = {"p0_1": low, "p99_9": high}
            if low < config.processing.value_min_db or high > config.processing.value_max_db:
                errors.append(
                    f"{name} range [{low:.2f}, {high:.2f}] is outside expected training-preprocessing "
                    f"range [{config.processing.value_min_db}, {config.processing.value_max_db}] dB"
                )
        if errors:
            raise RasterValidationError(errors, warnings)

        validation = {
            "status": "passed_with_warnings" if warnings else "passed",
            "errors": [],
            "warnings": warnings,
            "checks": {
                "band_order": list(observed_order),
                "resolution": list(resolution),
                "nodata_fraction": nodata_fraction,
                "nodata_fill": "per-band median" if nodata_pixels else "none",
                "roi_coverage_fraction": roi_coverage,
                "minimum_window": config.processing.tile_size,
                "band_ranges_db": band_ranges,
                "orbit_tag": tags.get("orbit"),
            },
        }
        tagged_orbit = tags.get("orbit")
        if tagged_orbit and tagged_orbit != config.model.orbit:
            raise RasterValidationError([
                f"raster orbit metadata is {tagged_orbit}, but selected model orbit is {config.model.orbit}"
            ], warnings)
        if not tagged_orbit:
            validation["warnings"].append("orbit metadata missing; orbit compatibility inferred from command selection")
            validation["status"] = "passed_with_warnings"

        metadata = {
            "path": str(path), "width": src.width, "height": src.height,
            "crs": str(src.crs), "band_order": list(observed_order),
            "source": tags.get("source", "prepared-raster"), "cache_key": tags.get("cache_key"),
            "validation": validation,
        }
        profile = src.profile.copy()
        profile["nodata"] = src.nodata
        return RasterInput(path, data, profile, src.transform, src.crs, metadata)
