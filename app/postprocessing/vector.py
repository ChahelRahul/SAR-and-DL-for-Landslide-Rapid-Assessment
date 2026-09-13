from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from app.config import MODEL_VERSION

GEOJSON_CRS = "EPSG:4326"
GEOMETRY_SEMANTICS = "thresholded_candidate_area"
BOUNDARY_SEMANTICS = "not_exact_landslide_boundary"


def _require_geo():
    try:
        import geopandas as gpd
        from rasterio.features import geometry_mask, shapes
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("Vector output requires the 'geo' extra") from exc
    return gpd, geometry_mask, shapes, shape


def _roi_geometry(roi_geojson: dict[str, Any] | None, crs):
    if roi_geojson is None:
        return None
    gpd, _, _, shape = _require_geo()
    payload = roi_geojson.get("geometry") if roi_geojson.get("type") == "Feature" else roi_geojson
    if not isinstance(payload, dict):
        raise ValueError("ROI GeoJSON Feature must contain a geometry")
    geom = shape(payload)
    if geom.is_empty:
        raise ValueError("ROI geometry is empty")
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("ROI must be a Polygon or MultiPolygon")
    return gpd.GeoSeries([geom], crs=GEOJSON_CRS).to_crs(crs).iloc[0]


def detection_geodataframe(
    mask: np.ndarray,
    probability_surface: np.ndarray,
    *,
    transform,
    crs,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
    roi_geojson: dict[str, Any] | None = None,
):
    """Create candidate-area features in the source raster CRS.

    Vector geometry is derived from the thresholded raster, not from model-window
    bounding boxes. When an ROI is supplied, features are intersected with it so
    no vector geometry extends outside the submitted ROI.
    """
    gpd, geometry_mask, shapes, shape = _require_geo()
    mask = np.asarray(mask, dtype=np.uint8)
    probability_surface = np.asarray(probability_surface, dtype=np.float32)
    if probability_surface.shape != mask.shape:
        raise ValueError("probability surface and binary mask must have identical dimensions")
    if crs is None:
        raise ValueError("Vector output requires a raster CRS")

    roi_geometry = _roi_geometry(roi_geojson, crs)
    config_json = json.dumps(effective_config, separators=(",", ":"))
    records: list[dict[str, Any]] = []
    for geometry, value in shapes(mask, mask=mask == 1, transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geometry)
        if roi_geometry is not None:
            geom = geom.intersection(roi_geometry)
            if geom.is_empty:
                continue
        inside = ~geometry_mask(
            [geom.__geo_interface__],
            out_shape=mask.shape,
            transform=transform,
            invert=False,
        )
        values = probability_surface[inside & np.isfinite(probability_surface)]
        probability = float(values.max()) if values.size else float("nan")
        records.append(
            {
                "geometry": geom,
                "probability": probability,
                "model_ver": MODEL_VERSION,
                "cfg_version": str(effective_config["model"]["version"]),
                "orbit": orbit,
                "rel_orbit": relative_orbit,
                "weights_sha256": weights_sha256,
                "threshold": float(effective_config["model"]["probability_threshold"]),
                "geometry_semantics": GEOMETRY_SEMANTICS,
                "boundary_semantics": BOUNDARY_SEMANTICS,
                "effective_config": config_json,
            }
        )

    columns = [
        "probability",
        "model_ver",
        "cfg_version",
        "orbit",
        "rel_orbit",
        "weights_sha256",
        "threshold",
        "geometry_semantics",
        "boundary_semantics",
        "effective_config",
        "geometry",
    ]
    if records:
        return gpd.GeoDataFrame(records, crs=crs)[columns]
    return gpd.GeoDataFrame({name: [] for name in columns if name != "geometry"}, geometry=[], crs=crs)


def write_detection_geojson(
    path: str | Path,
    mask: np.ndarray,
    probability_surface: np.ndarray,
    **kwargs,
) -> Path:
    """Write RFC-7946-style GeoJSON in EPSG:4326, including empty results."""
    from shapely.geometry import mapping

    gdf = detection_geodataframe(mask, probability_surface, **kwargs).to_crs(GEOJSON_CRS)
    features = []
    for _, row in gdf.iterrows():
        props = {k: row[k] for k in gdf.columns if k != "geometry"}
        # Normalise NumPy/pandas scalar values for the JSON encoder.
        for key, value in list(props.items()):
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            props[key] = value
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": props,
            }
        )
    payload = {
        "type": "FeatureCollection",
        "features": features,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _empty_gpkg(path: Path, crs) -> None:
    try:
        import fiona
        from pyproj import CRS
    except ImportError as exc:
        raise RuntimeError("GeoPackage output requires Fiona/pyproj from the 'geo' extra") from exc
    schema = {
        "geometry": "Unknown",
        "properties": {
            "probability": "float",
            "model_ver": "str",
            "cfg_version": "str",
            "orbit": "str",
            "rel_orbit": "int",
            "weights_sha256": "str",
            "threshold": "float",
            "geometry_semantics": "str",
            "boundary_semantics": "str",
            "effective_config": "str",
        },
    }
    with fiona.open(
        path,
        mode="w",
        driver="GPKG",
        layer="detections",
        schema=schema,
        crs_wkt=CRS.from_user_input(crs).to_wkt(),
    ):
        pass


def write_detection_geopackage(
    path: str | Path,
    mask: np.ndarray,
    probability_surface: np.ndarray,
    **kwargs,
) -> Path:
    """Write candidate features to a single-file GeoPackage in raster CRS."""
    gdf = detection_geodataframe(mask, probability_surface, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        _empty_gpkg(path, gdf.crs)
    else:
        gdf.to_file(path, layer="detections", driver="GPKG")
    return path


def write_detection_shapefile_zip(
    path: str | Path,
    mask: np.ndarray,
    probability_surface: np.ndarray,
    **kwargs,
) -> Path:
    """Optional compatibility export: a zipped Shapefile, never loose sidecars."""
    gdf = detection_geodataframe(mask, probability_surface, **kwargs)
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Shapefile compatibility export must use a .zip output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sar-lra-shp-") as tmp:
        tmpdir = Path(tmp)
        shp = tmpdir / "detections.shp"
        if gdf.empty:
            try:
                import fiona
                from pyproj import CRS
            except ImportError as exc:
                raise RuntimeError("Shapefile output requires Fiona/pyproj from the 'geo' extra") from exc
            schema = {
                "geometry": "Polygon",
                "properties": {
                    "prob": "float",
                    "model_ver": "str",
                    "orbit": "str",
                    "threshold": "float",
                },
            }
            with fiona.open(
                shp,
                mode="w",
                driver="ESRI Shapefile",
                schema=schema,
                crs_wkt=CRS.from_user_input(gdf.crs).to_wkt(),
            ):
                pass
        else:
            # Shapefile has 10-character field-name limits, so export a reduced
            # compatibility schema rather than silently truncating API fields.
            compat = gdf[["probability", "model_ver", "orbit", "threshold", "geometry"]].rename(
                columns={"probability": "prob"}
            )
            compat.to_file(shp, driver="ESRI Shapefile")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for component in sorted(tmpdir.glob("detections.*")):
                archive.write(component, arcname=component.name)
    return path


# Backward-compatible dispatcher retained for one release.
def write_detection_vectors(path: str | Path, mask: np.ndarray, probability_surface: np.ndarray, **kwargs) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return write_detection_geojson(path, mask, probability_surface, **kwargs)
    if suffix == ".gpkg":
        return write_detection_geopackage(path, mask, probability_surface, **kwargs)
    if suffix == ".zip":
        return write_detection_shapefile_zip(path, mask, probability_surface, **kwargs)
    raise ValueError("Vector output must be .geojson, .gpkg, or a zipped Shapefile (.zip)")
