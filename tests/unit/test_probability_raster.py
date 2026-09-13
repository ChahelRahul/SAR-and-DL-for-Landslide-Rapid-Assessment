import numpy as np
import pytest


def test_saved_probability_raster_can_be_rethresholded_without_model(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from app.postprocessing.raster import threshold_probability_raster, write_probability_raster

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 20, 10, 10),
    }
    probability = np.array([[0.2, 0.7, np.nan], [0.9, 0.4, 0.8]], dtype=np.float32)
    effective = {
        "model": {"version": "v2", "probability_threshold": 0.6},
        "processing": {"probability_aggregation": "maximum"},
    }
    probability_path = write_probability_raster(
        tmp_path / "probability.tif",
        probability,
        profile,
        orbit="ASCENDING",
        relative_orbit=None,
        weights_sha256="abc",
        effective_config=effective,
    )
    binary_path = threshold_probability_raster(
        probability_path,
        tmp_path / "threshold-08.tif",
        threshold=0.8,
    )
    with rasterio.open(binary_path) as src:
        mask = src.read(1)
        assert src.tags()["derived_from"] == "probability.tif"
        assert src.tags()["probability_threshold"] == "0.8"
    np.testing.assert_array_equal(mask, [[0, 0, 0], [1, 0, 1]])
