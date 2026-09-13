import json

import numpy as np
import pytest

from app.config import ModelConfig, ProcessingConfig
from app.inference.predict import (
    ScoredWindow,
    aggregate_window_probabilities,
    predict_stack,
    threshold_probability_surface,
)
from app.inference.windows import Window


class SequenceModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict(self, patches, batch_size, verbose):
        assert len(patches) == len(self.scores)
        return self.scores.reshape(-1, 1)


def stack(height, width):
    data = np.zeros((height, width, 4), dtype=np.float32)
    data[..., 0] = -15
    data[..., 1] = -20
    data[..., 2] = 1
    data[..., 3] = -1
    return data


def test_maximum_window_aggregation_is_threshold_independent():
    scored = [
        ScoredWindow(Window(0, 0, 2, 2), 0.4),
        ScoredWindow(Window(1, 0, 3, 2), 0.8),
    ]
    surface = aggregate_window_probabilities((2, 3), scored)
    np.testing.assert_allclose(surface, [[0.4, 0.8, 0.8], [0.4, 0.8, 0.8]])

    low = threshold_probability_surface(surface, 0.5)
    high = threshold_probability_surface(surface, 0.9)
    assert low.sum() == 4
    assert high.sum() == 0
    # Re-thresholding works directly from the retained probabilities.
    np.testing.assert_allclose(surface, [[0.4, 0.8, 0.8], [0.4, 0.8, 0.8]])


def test_probability_surface_preserves_scores_below_current_binary_threshold():
    # 65x65 produces four windows with the Issue 8 edge-coverage policy.
    model = SequenceModel([0.2, 0.4, 0.7, 0.9])
    prediction = predict_stack(
        model,
        stack(65, 65),
        ModelConfig(probability_threshold=0.6),
        ProcessingConfig(),
    )
    assert prediction.probability_surface.shape == (65, 65)
    assert np.isclose(np.nanmin(prediction.probability_surface), 0.2)
    assert np.isclose(np.nanmax(prediction.probability_surface), 0.9)
    # Scores under 0.6 remain in the probability output even though they are
    # absent from the binary mask.
    assert np.any((prediction.probability_surface < 0.6) & np.isfinite(prediction.probability_surface))
    assert set(np.unique(prediction.mask)).issubset({0, 1})


def test_invalid_pixels_are_nodata_in_probability_and_zero_in_binary():
    valid = np.zeros((64, 64), dtype=bool)
    valid[10:20, 10:20] = True
    prediction = predict_stack(
        SequenceModel([0.9]),
        stack(64, 64),
        ModelConfig(),
        ProcessingConfig(),
        valid_mask=valid,
    )
    assert np.isnan(prediction.probability_surface[~valid]).all()
    assert (prediction.mask[~valid] == 0).all()
    assert np.isclose(prediction.probability_surface[valid], 0.9).all()


def test_processing_config_rejects_unknown_probability_aggregation():
    with pytest.raises(ValueError, match="maximum"):
        ProcessingConfig(probability_aggregation="mean")


def test_vector_features_include_probability(tmp_path):
    gpd = pytest.importorskip("geopandas")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from app.postprocessing.vector import write_detection_vectors

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    probability = np.zeros((4, 4), dtype=np.float32)
    probability[1:3, 1:3] = [[0.65, 0.8], [0.7, 0.75]]
    path = write_detection_vectors(
        tmp_path / "detections.geojson",
        mask,
        probability,
        transform=from_origin(0, 40, 10, 10),
        crs="EPSG:3857",
        orbit="ASCENDING",
        relative_orbit=42,
        weights_sha256="abc",
        effective_config={
            "model": {"version": "v2", "probability_threshold": 0.6},
            "processing": {"probability_aggregation": "maximum"},
        },
    )
    assert path is not None
    payload = json.loads(path.read_text())
    props = payload["features"][0]["properties"]
    assert props["probability"] == pytest.approx(0.8)
    assert props["orbit"] == "ASCENDING"
    assert props["rel_orbit"] == 42
    assert props["model_ver"]
