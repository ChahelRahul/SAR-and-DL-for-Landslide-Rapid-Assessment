from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.acquisition.local_raster import RasterInput, read_sentinel1_stack
from app.config import AppConfig, EXPECTED_BAND_ORDER, Orbit

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"


def _deps():
    try:
        import planetary_computer as pc
        import pystac_client
        import rasterio
        from rasterio.features import bounds as geometry_bounds
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, transform_bounds, transform_geom
        from rasterio.vrt import WarpedVRT
    except ImportError as exc:
        raise RuntimeError(
            "Planetary Computer acquisition requires the 'planetary-computer' extra"
        ) from exc
    return pc, pystac_client, rasterio, geometry_bounds, from_origin, Resampling, transform_bounds, transform_geom, WarpedVRT


def _cache_key(roi: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig) -> str:
    payload = {
        "provider": "planetary-computer",
        "collection": COLLECTION,
        "roi": roi,
        "event_date": event_date.isoformat(),
        "orbit": orbit,
        "pre_days": config.imagery.pre_days,
        "post_days": config.imagery.post_days,
        "scale_m": config.imagery.scale_m,
        "bands": EXPECTED_BAND_ORDER,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _utc(day: date) -> str:
    return datetime.combine(day, time.min, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _orbit_state(item: Any) -> str | None:
    value = item.properties.get("sat:orbit_state") or item.properties.get("sat:orbit_state".replace("_", ""))
    return str(value).upper() if value else None


def _relative_orbit(item: Any) -> int | None:
    value = item.properties.get("sat:relative_orbit")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def search_items(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig) -> tuple[list[Any], list[Any], int]:
    pc, pystac_client, *_ = _deps()
    subscription_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
    if not subscription_key:
        raise RuntimeError(
            "Planetary Computer Sentinel-1 RTC asset access requires PC_SDK_SUBSCRIPTION_KEY. "
            "Create/use a Planetary Computer account key, or use /v1/predict-raster for provider-independent inference."
        )
    pc.settings.set_subscription_key(subscription_key)
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    geometry = roi_geojson.get("geometry", roi_geojson)
    start = event_date - timedelta(days=config.imagery.pre_days)
    end = event_date + timedelta(days=config.imagery.post_days)
    items = list(
        catalog.search(
            collections=[COLLECTION],
            intersects=geometry,
            datetime=f"{_utc(start)}/{_utc(end)}",
        ).items()
    )
    wanted = []
    for item in items:
        props = item.properties
        polarizations = {str(x).upper() for x in props.get("sar:polarizations", [])}
        mode = str(props.get("sar:instrument_mode", "")).upper()
        if mode and mode != "IW":
            continue
        if polarizations and not {"VV", "VH"}.issubset(polarizations):
            continue
        if _orbit_state(item) not in {None, orbit}:
            continue
        if "vv" not in item.assets or "vh" not in item.assets:
            continue
        wanted.append(item)
    pre = [i for i in wanted if i.datetime and start <= i.datetime.date() < event_date]
    post = [i for i in wanted if i.datetime and event_date <= i.datetime.date() <= end]
    common_orbits = Counter(_relative_orbit(i) for i in pre if _relative_orbit(i) is not None)
    post_orbits = Counter(_relative_orbit(i) for i in post if _relative_orbit(i) is not None)
    candidates = sorted(set(common_orbits) & set(post_orbits))
    if not candidates:
        raise ValueError("No common Sentinel-1 relative orbit has both pre- and post-event Planetary Computer RTC scenes")
    relative_orbit = max(candidates, key=lambda x: (min(common_orbits[x], post_orbits[x]), common_orbits[x] + post_orbits[x], -x))
    pre = [i for i in pre if _relative_orbit(i) == relative_orbit]
    post = [i for i in post if _relative_orbit(i) == relative_orbit]
    if not pre or not post:
        raise ValueError("Planetary Computer search returned an incomplete pre/post Sentinel-1 scene set")
    return pre, post, relative_orbit


def _utm_epsg(geometry: dict[str, Any]) -> str:
    coords = geometry.get("coordinates") or []
    flat: list[tuple[float, float]] = []
    def walk(node):
        if isinstance(node, (list, tuple)) and len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            flat.append((float(node[0]), float(node[1])))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
    walk(coords)
    if not flat:
        raise ValueError("ROI has no coordinates")
    lon = sum(p[0] for p in flat) / len(flat)
    lat = sum(p[1] for p in flat) / len(flat)
    zone = max(1, min(60, int(math.floor((lon + 180) / 6) + 1)))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _load_composite(items: list[Any], asset_name: str, *, dst_crs: str, transform: Any, width: int, height: int) -> np.ndarray:
    *_, rasterio, _geometry_bounds, _from_origin, Resampling, _transform_bounds, _transform_geom, WarpedVRT = _deps()
    stack: list[np.ndarray] = []
    for item in items:
        href = item.assets[asset_name].href
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                nodata=np.nan,
            ) as vrt:
                values = vrt.read(1, masked=True).astype("float32").filled(np.nan)
        values[values <= 0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 10.0 * np.log10(values)
        stack.append(values.astype("float32"))
    return np.nanmedian(np.stack(stack, axis=0), axis=0).astype("float32")


def acquire_intermediate_raster(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig, cache_dir: Path) -> tuple[RasterInput, bool]:
    pc, pystac_client, rasterio, geometry_bounds, from_origin, Resampling, transform_bounds, transform_geom, WarpedVRT = _deps()
    del pc, pystac_client, Resampling, transform_bounds, WarpedVRT
    key = _cache_key(roi_geojson, event_date, orbit, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"sentinel1-pc-rtc-{orbit.lower()}-{event_date.isoformat()}-{key}.tif"
    if target.is_file():
        return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), True

    pre, post, relative_orbit = search_items(roi_geojson=roi_geojson, event_date=event_date, orbit=orbit, config=config)
    geometry = roi_geojson.get("geometry", roi_geojson)
    dst_crs = _utm_epsg(geometry)
    projected = transform_geom("EPSG:4326", dst_crs, geometry, precision=8)
    left, bottom, right, top = geometry_bounds(projected)
    scale = float(config.imagery.scale_m)
    width = max(1, int(math.ceil((right - left) / scale)))
    height = max(1, int(math.ceil((top - bottom) / scale)))
    transform = from_origin(left, top, scale, scale)

    pre_vv = _load_composite(pre, "vv", dst_crs=dst_crs, transform=transform, width=width, height=height)
    pre_vh = _load_composite(pre, "vh", dst_crs=dst_crs, transform=transform, width=width, height=height)
    post_vv = _load_composite(post, "vv", dst_crs=dst_crs, transform=transform, width=width, height=height)
    post_vh = _load_composite(post, "vh", dst_crs=dst_crs, transform=transform, width=width, height=height)
    bands = np.stack([post_vv, post_vh, post_vv - pre_vv, post_vh - pre_vh]).astype("float32")
    nodata = -9999.0
    bands = np.where(np.isfinite(bands), bands, nodata).astype("float32")
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 4,
        "dtype": "float32", "crs": dst_crs, "transform": transform,
        "nodata": nodata, "compress": "deflate", "tiled": True,
    }
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(bands)
        for index, band in enumerate(EXPECTED_BAND_ORDER, start=1):
            dst.set_band_description(index, band)
        dst.update_tags(
            source="planetary-computer-sentinel-1-rtc",
            cache_key=key,
            orbit=orbit,
            relative_orbit=str(relative_orbit),
            event_date=event_date.isoformat(),
            pre_days=str(config.imagery.pre_days),
            post_days=str(config.imagery.post_days),
            scale_m=str(config.imagery.scale_m),
            input_band_order=",".join(EXPECTED_BAND_ORDER),
            source_collection=COLLECTION,
            source_scale="RTC linear intensity converted to dB before temporal median",
            pre_scene_count=str(len(pre)),
            post_scene_count=str(len(post)),
        )
    return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), False
