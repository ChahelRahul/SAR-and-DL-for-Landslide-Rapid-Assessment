from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.config import EXPECTED_BAND_ORDER, ImageryConfig, Orbit
from app.acquisition.local_raster import RasterInput, read_sentinel1_stack


def initialize(project: str | None = None, *, authenticate: bool = False) -> Any:
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Earth Engine acquisition requires the 'earth-engine' extra") from exc
    if authenticate:
        ee.Authenticate()
    ee.Initialize(**({"project": project} if project else {}))
    return ee


def build_composite_image(
    ee: Any,
    geometry: Any,
    *,
    orbit: Orbit,
    relative_orbit: int,
    event_date: date,
    config: ImageryConfig,
) -> Any:
    pre_start = event_date - timedelta(days=config.pre_days)
    post_end = event_date + timedelta(days=config.post_days)
    base = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
    )
    pre = base.filterDate(str(pre_start), str(event_date))
    post = base.filterDate(str(event_date), str(post_end))
    pre_vv = pre.select("VV").median()
    pre_vh = pre.select("VH").median()
    post_vv = post.select("VV").median().rename(EXPECTED_BAND_ORDER[0])
    post_vh = post.select("VH").median().rename(EXPECTED_BAND_ORDER[1])
    diff_vv = post_vv.subtract(pre_vv).rename(EXPECTED_BAND_ORDER[2])
    diff_vh = post_vh.subtract(pre_vh).rename(EXPECTED_BAND_ORDER[3])
    return post_vv.addBands(post_vh).addBands(diff_vv).addBands(diff_vh).clip(geometry)


def _cache_key(roi: dict[str, Any], event_date: date, orbit: Orbit, config: ImageryConfig) -> str:
    payload = {
        "roi": roi,
        "event_date": event_date.isoformat(),
        "orbit": orbit,
        "imagery": {"pre_days": config.pre_days, "post_days": config.post_days, "scale_m": config.scale_m},
        "bands": EXPECTED_BAND_ORDER,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def acquire_intermediate_raster(
    *,
    roi_geojson: dict[str, Any],
    event_date: date,
    orbit: Orbit,
    config: ImageryConfig,
    cache_dir: Path,
    project: str | None = None,
    authenticate: bool = False,
) -> tuple[RasterInput, bool]:
    """Acquire or reuse a validated four-band GeoTIFF.

    Earth Engine and geemap are imported only when a cache miss requires an
    export. The returned boolean is true when an existing cache entry was used.
    """
    key = _cache_key(roi_geojson, event_date, orbit, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"sentinel1-{orbit.lower()}-{event_date.isoformat()}-{key}.tif"
    if target.is_file():
        return read_sentinel1_stack(target, require_contract_tags=True), True

    ee = initialize(project, authenticate=authenticate)
    try:
        import geemap
    except ImportError as exc:
        raise RuntimeError("Earth Engine raster export requires geemap") from exc
    geometry_value = roi_geojson.get("geometry", roi_geojson)
    geometry = ee.Geometry(geometry_value)
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(str(event_date - timedelta(days=config.pre_days)), str(event_date + timedelta(days=config.post_days)))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
    )
    relative_orbits = collection.aggregate_array("relativeOrbitNumber_start").distinct().getInfo() or []
    if not relative_orbits:
        raise ValueError("No Sentinel-1 relative orbit covers the ROI and configured date range")
    relative_orbit = int(sorted(relative_orbits)[0])
    image = build_composite_image(
        ee, geometry, orbit=orbit, relative_orbit=relative_orbit,
        event_date=event_date, config=config,
    )
    geemap.ee_export_image(
        image,
        filename=str(target),
        scale=config.scale_m,
        region=geometry,
        file_per_band=False,
    )
    _write_contract_metadata(target, key, orbit, relative_orbit, event_date, config)
    return read_sentinel1_stack(target, require_contract_tags=True), False


def _write_contract_metadata(
    path: Path, key: str, orbit: Orbit, relative_orbit: int,
    event_date: date, config: ImageryConfig,
) -> None:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Earth Engine raster validation requires the 'geo' extra") from exc
    with rasterio.open(path, "r+") as dst:
        for index, band in enumerate(EXPECTED_BAND_ORDER, start=1):
            dst.set_band_description(index, band)
        dst.update_tags(
            source="earth-engine",
            cache_key=key,
            orbit=orbit,
            relative_orbit=str(relative_orbit),
            event_date=event_date.isoformat(),
            pre_days=str(config.pre_days),
            post_days=str(config.post_days),
            scale_m=str(config.scale_m),
            input_band_order=",".join(EXPECTED_BAND_ORDER),
        )
