from __future__ import annotations

import math

import pytest


def _bce_probability(y: float, p: float) -> float:
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _bce_logit(y: float, z: float) -> float:
    # Stable enough for the small fixed values used here.
    return max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))


def _focal_from_bce(bce: float, alpha: float = 0.25, gamma: float = 2.0) -> float:
    p_t = math.exp(-bce)
    return alpha * (1.0 - p_t) ** gamma * bce


def test_known_probability_and_logit_reference_values_are_not_interchangeable():
    """Document the numeric distinction that motivated Issue 7."""
    y = 1.0
    probability = 0.8
    logit = math.log(probability / (1.0 - probability))

    probability_loss = _focal_from_bce(_bce_probability(y, probability))
    logit_loss = _focal_from_bce(_bce_logit(y, logit))
    wrong_logit_interpretation = _focal_from_bce(_bce_logit(y, probability))

    assert probability_loss == pytest.approx(logit_loss, rel=1e-12)
    assert wrong_logit_interpretation != pytest.approx(probability_loss, rel=1e-3)


def test_tensorflow_loss_functions_match_their_declared_semantics():
    tf = pytest.importorskip("tensorflow")
    from app.inference.model import probability_focal_loss, released_focal_loss

    y_true = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    probabilities = tf.constant([[0.8], [0.2]], dtype=tf.float32)

    corrected = probability_focal_loss()(y_true, probabilities).numpy()
    released = released_focal_loss()(y_true, probabilities).numpy()

    # Standalone tensors have no Keras sigmoid provenance, so the two declared
    # interpretations must be numerically different.
    assert corrected.shape == released.shape
    assert not (abs(corrected - released) < 1e-7).all()


def test_compilation_choice_does_not_change_forward_predictions():
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    from app.config import ModelLoadConfig
    from app.inference.model import build_model, compile_for_training

    config = ModelLoadConfig(
        weights_path="unused.h5",
        filters_first_layer=2,
        patch_size=64,
        channels=4,
        dropout=0.0,
        learning_rate=0.001,
    )
    released_model = build_model(config)
    probability_model = build_model(config)
    probability_model.set_weights(released_model.get_weights())

    compile_for_training(released_model, config, semantics="released")
    compile_for_training(probability_model, config, semantics="probability")

    sample = np.linspace(-2.0, 2.0, 64 * 64 * 4, dtype=np.float32).reshape(1, 64, 64, 4)
    released_pred = released_model(sample, training=False).numpy()
    probability_pred = probability_model(sample, training=False).numpy()

    np.testing.assert_array_equal(released_pred, probability_pred)
    assert 0.0 <= float(released_pred[0, 0]) <= 1.0


def test_released_weight_files_match_expected_architecture_shapes():
    h5py = pytest.importorskip("h5py")
    from pathlib import Path

    expected = {
        "ASCENDING": {"filters": 32, "dense": 256, "flat": 6144},
        "DESCENDING": {"filters": 64, "dense": 512, "flat": 12288},
    }
    for orbit, spec in expected.items():
        path = next(Path("model/weights").glob(f"*{orbit}*.hdf5"))
        shapes = []
        with h5py.File(path, "r") as handle:
            def collect(_name, obj):
                if isinstance(obj, h5py.Dataset):
                    shapes.append(tuple(obj.shape))
            handle.visititems(collect)

        f = spec["filters"]
        assert (3, 3, 4, f) in shapes
        assert (3, 3, f, f) in shapes
        assert (spec["flat"], spec["dense"]) in shapes
        assert (spec["dense"], 1) in shapes
