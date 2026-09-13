from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from app.config import AppConfig, ProcessingConfig
from app.postprocessing.vector import (
    write_detection_geojson,
    write_detection_geopackage,
    write_detection_shapefile_zip,
)


@pytest.fixture
def vector_kwargs():
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    return {
        "transform": from_origin(0, 4, 1, 1),
        "crs": "EPSG:4326",
        "orbit": "ASCENDING",
        "relative_orbit": 42,
        "weights_sha256": "abc",
        "effective_config": {
            "model": {"version": "v2", "probability_threshold": 0.6},
            "processing": {"probability_aggregation": "maximum"},
        },
    }


def test_geojson_is_default_vector_format():
    config = AppConfig()
    assert config.processing.vector_format == "geojson"
    assert config.processing.write_shapefile_zip is False


def test_empty_prediction_writes_valid_empty_geojson(tmp_path, vector_kwargs):
    mask = np.zeros((4, 4), dtype=np.uint8)
    probability = np.zeros((4, 4), dtype=np.float32)
    path = write_detection_geojson(
        tmp_path / "detections.geojson",
        mask,
        probability,
        **vector_kwargs,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"type": "FeatureCollection", "features": []}


def test_geojson_is_clipped_to_roi(tmp_path, vector_kwargs):
    mask = np.ones((4, 4), dtype=np.uint8)
    probability = np.full((4, 4), 0.8, dtype=np.float32)
    roi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]],
    }
    path = write_detection_geojson(
        tmp_path / "detections.geojson",
        mask,
        probability,
        roi_geojson=roi,
        **vector_kwargs,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["features"]) == 1
    coords = payload["features"][0]["geometry"]["coordinates"]
    xs = [point[0] for ring in coords for point in ring]
    assert min(xs) >= 0
    assert max(xs) <= 2


def test_geopackage_is_single_file_and_readable(tmp_path, vector_kwargs):
    gpd = pytest.importorskip("geopandas")
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    probability = np.full((4, 4), 0.8, dtype=np.float32)
    path = write_detection_geopackage(
        tmp_path / "detections.gpkg",
        mask,
        probability,
        **vector_kwargs,
    )
    assert path.is_file()
    result = gpd.read_file(path, layer="detections")
    assert len(result) == 1
    assert result.crs.to_epsg() == 4326


def test_shapefile_is_only_exposed_as_zip(tmp_path, vector_kwargs):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    probability = np.full((4, 4), 0.8, dtype=np.float32)
    path = write_detection_shapefile_zip(
        tmp_path / "detections-shapefile.zip",
        mask,
        probability,
        **vector_kwargs,
    )
    assert path.is_file()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    assert "detections.shp" in names
    assert "detections.shx" in names
    assert "detections.dbf" in names
    assert not list(tmp_path.glob("*.shp"))


def test_invalid_vector_format_rejected():
    with pytest.raises(ValueError, match="vector_format"):
        ProcessingConfig(vector_format="shapefile")
