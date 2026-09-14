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
RTC_COLLECTION = "sentinel-1-rtc"
GRD_COLLECTION = "sentinel-1-grd"
DEM_COLLECTION = "cop-dem-glo-30"
COLLECTION = GRD_COLLECTION


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
        "provider": "planetary-computer-rtc",
        "collection": RTC_COLLECTION,
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


def search_rtc_items(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig) -> tuple[list[Any], list[Any], int]:
    pc, pystac_client, *_ = _deps()
    subscription_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
    if not subscription_key:
        raise RuntimeError(
            "Planetary Computer precomputed RTC access requires PC_SDK_SUBSCRIPTION_KEY. "
            "Use provider=planetary-computer-grd (or planetary-computer) for keyless GRD acquisition."
        )
    pc.settings.set_subscription_key(subscription_key)
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    geometry = roi_geojson.get("geometry", roi_geojson)
    start = event_date - timedelta(days=config.imagery.pre_days)
    end = event_date + timedelta(days=config.imagery.post_days)
    items = list(
        catalog.search(
            collections=[RTC_COLLECTION],
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


def acquire_rtc_intermediate_raster(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig, cache_dir: Path) -> tuple[RasterInput, bool]:
    pc, pystac_client, rasterio, geometry_bounds, from_origin, Resampling, transform_bounds, transform_geom, WarpedVRT = _deps()
    del pc, pystac_client, Resampling, transform_bounds, WarpedVRT
    key = _cache_key(roi_geojson, event_date, orbit, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"sentinel1-pc-rtc-{orbit.lower()}-{event_date.isoformat()}-{key}.tif"
    if target.is_file():
        return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), True

    pre, post, relative_orbit = search_rtc_items(roi_geojson=roi_geojson, event_date=event_date, orbit=orbit, config=config)
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
            source_collection=RTC_COLLECTION,
            source_scale="RTC linear intensity converted to dB before temporal median",
            pre_scene_count=str(len(pre)),
            post_scene_count=str(len(post)),
        )
    return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), False

# --- Keyless Sentinel-1 GRD path -------------------------------------------------
# Planetary Computer exposes public STAC metadata and issues read-only SAS tokens
# for the Sentinel-1 GRD Azure container without requiring a user subscription key.
# The raw SAFE product is terrain-corrected locally with sarsen.


def _grd_deps():
    try:
        import adlfs
        import planetary_computer as pc
        import pystac_client
        import rasterio
        import rioxarray  # noqa: F401 - registers the xarray .rio accessor
        import sarsen
        from rasterio.features import bounds as geometry_bounds
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject, transform_geom
    except ImportError as exc:
        raise RuntimeError(
            "Keyless Planetary Computer GRD acquisition requires the planetary-computer-grd dependencies "
            "(adlfs, sarsen, rioxarray, rasterio, pystac-client, planetary-computer)."
        ) from exc
    return adlfs, pc, pystac_client, rasterio, sarsen, geometry_bounds, from_origin, Resampling, reproject, transform_geom


def _search_grd_items(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig) -> tuple[list[Any], list[Any], int]:
    _adlfs, _pc, pystac_client, *_ = _grd_deps()
    catalog = pystac_client.Client.open(STAC_URL)
    geometry = roi_geojson.get("geometry", roi_geojson)
    start = event_date - timedelta(days=config.imagery.pre_days)
    end = event_date + timedelta(days=config.imagery.post_days)
    items = list(catalog.search(
        collections=[GRD_COLLECTION], intersects=geometry,
        datetime=f"{_utc(start)}/{_utc(end)}",
    ).items())
    wanted: list[Any] = []
    for item in items:
        props = item.properties
        pols = {str(x).upper() for x in props.get("sar:polarizations", [])}
        mode = str(props.get("sar:instrument_mode", "")).upper()
        if mode and mode != "IW":
            continue
        if pols and not {"VV", "VH"}.issubset(pols):
            continue
        if _orbit_state(item) not in {None, orbit}:
            continue
        if "safe-manifest" not in item.assets:
            continue
        wanted.append(item)
    pre = [i for i in wanted if i.datetime and start <= i.datetime.date() < event_date]
    post = [i for i in wanted if i.datetime and event_date <= i.datetime.date() <= end]
    pre_counts = Counter(_relative_orbit(i) for i in pre if _relative_orbit(i) is not None)
    post_counts = Counter(_relative_orbit(i) for i in post if _relative_orbit(i) is not None)
    candidates = sorted(set(pre_counts) & set(post_counts))
    if not candidates:
        raise ValueError("No common Sentinel-1 relative orbit has both pre- and post-event GRD scenes")
    relative_orbit = max(candidates, key=lambda x: (min(pre_counts[x], post_counts[x]), pre_counts[x] + post_counts[x], -x))
    pre = [i for i in pre if _relative_orbit(i) == relative_orbit]
    post = [i for i in post if _relative_orbit(i) == relative_orbit]
    return pre, post, relative_orbit


def _safe_product_folder(item: Any) -> tuple[str, str]:
    """Return Azure container-relative folder and local folder name from safe-manifest."""
    from urllib.parse import urlparse
    href = item.assets["safe-manifest"].href
    parsed = urlparse(href)
    parts = parsed.path.lstrip("/").split("/")
    if len(parts) < 3 or parts[0] != "s1-grd" or parts[-1] != "manifest.safe":
        raise ValueError(f"unexpected Sentinel-1 GRD safe-manifest URL: {href}")
    folder = "/".join(parts[:-1])  # includes s1-grd container prefix for adlfs
    return folder, parts[-2]


def _download_safe(item: Any, target_root: Path) -> Path:
    adlfs, pc, *_ = _grd_deps()
    folder, product_name = _safe_product_folder(item)
    local = target_root / product_name
    if (local / "manifest.safe").is_file():
        return local
    target_root.mkdir(parents=True, exist_ok=True)
    # Anonymous read-only SAS token; PC_SDK_SUBSCRIPTION_KEY is not required.
    token = pc.sas.get_token("sentinel1euwest", "s1-grd").token
    fs = adlfs.AzureBlobFileSystem(account_name="sentinel1euwest", credential=token)
    if not fs.exists(f"{folder}/manifest.safe"):
        raise FileNotFoundError(f"Planetary Computer GRD manifest not found: {folder}/manifest.safe")
    fs.get(folder, str(local), recursive=True)
    if not (local / "manifest.safe").is_file():
        raise RuntimeError(f"GRD download completed without manifest.safe: {local}")
    return local


def _build_dem(*, roi_geojson: dict[str, Any], config: AppConfig, cache_dir: Path) -> Path:
    """Create a projected DEM for sarsen, using a local override or PC Copernicus DEM."""
    override = os.getenv("SAR_LRA_PC_DEM_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAR_LRA_PC_DEM_PATH does not exist: {path}")
        return path

    _adlfs, pc, pystac_client, rasterio, _sarsen, geometry_bounds, from_origin, Resampling, reproject, transform_geom = _grd_deps()
    geometry = roi_geojson.get("geometry", roi_geojson)
    dst_crs = _utm_epsg(geometry)
    projected = transform_geom("EPSG:4326", dst_crs, geometry, precision=8)
    left, bottom, right, top = geometry_bounds(projected)
    scale = float(config.imagery.scale_m)
    width = max(1, int(math.ceil((right - left) / scale)))
    height = max(1, int(math.ceil((top - bottom) / scale)))
    dst_transform = from_origin(left, top, scale, scale)
    dem_path = cache_dir / f"dem-{hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:16]}.tif"
    if dem_path.is_file():
        return dem_path

    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(catalog.search(collections=[DEM_COLLECTION], intersects=geometry).items())
    if not items:
        raise ValueError("No Copernicus DEM GLO-30 tiles found for ROI")
    destination = np.full((height, width), np.nan, dtype="float32")
    for item in items:
        asset = item.assets.get("data")
        if asset is None:
            asset = next((a for a in item.assets.values() if "tiff" in str(a.media_type or "").lower() or str(a.href).lower().endswith((".tif", ".tiff"))), None)
        if asset is None:
            continue
        with rasterio.open(asset.href) as src:
            reproject(
                source=rasterio.band(src, 1), destination=destination,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                src_nodata=src.nodata, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    if not np.isfinite(destination).any():
        raise RuntimeError("Copernicus DEM tiles were found but no DEM pixels could be projected into the ROI")
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": dst_crs, "transform": dst_transform,
        "nodata": -9999.0, "compress": "deflate", "tiled": True,
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(destination), destination, -9999.0).astype("float32"), 1)
    return dem_path


def _rtc_scene(product_path: Path, polarization: str, dem_path: Path, output_path: Path) -> Path:
    _adlfs, _pc, _pystac, _rasterio, sarsen, *_ = _grd_deps()
    if output_path.is_file():
        return output_path
    product = sarsen.Sentinel1SarProduct(str(product_path), measurement_group=f"IW/{polarization}")
    rtc = sarsen.terrain_correction(
        product, dem_urlpath=str(dem_path), correct_radiometry="gamma_bilinear"
    )
    rtc = rtc.squeeze(drop=True)
    rtc.rio.to_raster(str(output_path), compress="DEFLATE")
    return output_path


def _composite_rtc_files(paths: list[Path]) -> np.ndarray:
    *_prefix, rasterio, _sarsen, _geometry_bounds, _from_origin, _Resampling, _reproject, _transform_geom = _grd_deps()
    arrays: list[np.ndarray] = []
    shape = None
    for path in paths:
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        if shape is None:
            shape = arr.shape
        if arr.shape != shape:
            raise ValueError("sarsen RTC outputs do not share a common DEM grid")
        arr[arr <= 0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            arr = 10.0 * np.log10(arr)
        arrays.append(arr.astype("float32"))
    return np.nanmedian(np.stack(arrays, axis=0), axis=0).astype("float32")


def acquire_grd_intermediate_raster(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig, cache_dir: Path) -> tuple[RasterInput, bool]:
    *_deps_prefix, rasterio, _sarsen, _geometry_bounds, _from_origin, _Resampling, _reproject, _transform_geom = _grd_deps()
    payload = {
        "provider": "planetary-computer-grd", "collection": GRD_COLLECTION,
        "roi": roi_geojson, "event_date": event_date.isoformat(), "orbit": orbit,
        "pre_days": config.imagery.pre_days, "post_days": config.imagery.post_days,
        "scale_m": config.imagery.scale_m,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"sentinel1-pc-grd-{orbit.lower()}-{event_date.isoformat()}-{key}.tif"
    if target.is_file():
        return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), True

    pre, post, relative_orbit = _search_grd_items(roi_geojson=roi_geojson, event_date=event_date, orbit=orbit, config=config)
    dem = _build_dem(roi_geojson=roi_geojson, config=config, cache_dir=cache_dir / "dem")
    scenes_dir = cache_dir / "pc-grd-scenes"
    rtc_dir = cache_dir / "pc-grd-rtc"
    rtc_dir.mkdir(parents=True, exist_ok=True)

    def process(items: list[Any], pol: str) -> list[Path]:
        outputs: list[Path] = []
        for item in items:
            safe = _download_safe(item, scenes_dir)
            out = rtc_dir / f"{item.id}-{pol.lower()}.tif"
            outputs.append(_rtc_scene(safe, pol, dem, out))
        return outputs

    pre_vv = _composite_rtc_files(process(pre, "VV"))
    pre_vh = _composite_rtc_files(process(pre, "VH"))
    post_vv = _composite_rtc_files(process(post, "VV"))
    post_vh = _composite_rtc_files(process(post, "VH"))
    bands = np.stack([post_vv, post_vh, post_vv - pre_vv, post_vh - pre_vh]).astype("float32")
    nodata = -9999.0
    bands = np.where(np.isfinite(bands), bands, nodata).astype("float32")
    with rasterio.open(dem) as dem_src:
        profile = dem_src.profile.copy()
        profile.update(count=4, dtype="float32", nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(bands)
        for index, band in enumerate(EXPECTED_BAND_ORDER, start=1):
            dst.set_band_description(index, band)
        dst.update_tags(
            source="planetary-computer-sentinel-1-grd-keyless",
            access="anonymous-sas-token", user_subscription_key_required="false",
            cache_key=key, orbit=orbit, relative_orbit=str(relative_orbit),
            event_date=event_date.isoformat(), pre_days=str(config.imagery.pre_days),
            post_days=str(config.imagery.post_days), scale_m=str(config.imagery.scale_m),
            input_band_order=",".join(EXPECTED_BAND_ORDER), source_collection=GRD_COLLECTION,
            terrain_correction="sarsen gamma_bilinear", dem=str(dem),
            pre_scene_count=str(len(pre)), post_scene_count=str(len(post)),
        )
    return read_sentinel1_stack(target, config=config, roi_geojson=roi_geojson, require_contract_tags=True), False


def acquire_intermediate_raster(*, roi_geojson: dict[str, Any], event_date: date, orbit: Orbit, config: AppConfig, cache_dir: Path, variant: str = "grd") -> tuple[RasterInput, bool]:
    if variant == "rtc":
        return acquire_rtc_intermediate_raster(
            roi_geojson=roi_geojson, event_date=event_date, orbit=orbit, config=config, cache_dir=cache_dir
        )
    if variant != "grd":
        raise ValueError(f"unsupported Planetary Computer variant: {variant}")
    return acquire_grd_intermediate_raster(
        roi_geojson=roi_geojson, event_date=event_date, orbit=orbit, config=config, cache_dir=cache_dir
    )
