from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import ModelConfig, ProcessingConfig
from app.inference.predict import iter_window_batches, predict_stack


class MeanModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, patches, batch_size=None, verbose=0):
        patches = np.asarray(patches, dtype=np.float32)
        self.batch_sizes.append(len(patches))
        scores = patches.mean(axis=(1, 2, 3))
        # Map deterministic arbitrary means into [0, 1].
        scores = 1.0 / (1.0 + np.exp(-scores / 10.0))
        return scores.reshape(-1, 1).astype(np.float32)


def _stack(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    bands = [
        -18.0 + x / 100.0,
        -24.0 + y / 100.0,
        (x - y) / 100.0,
        (x + y) / 200.0,
    ]
    return np.stack(bands, axis=-1).astype(np.float32)


def test_window_batches_never_exceed_configured_batch_size():
    image = _stack(257, 319)
    valid = np.ones(image.shape[:2], dtype=bool)
    sizes = [
        len(windows)
        for windows, _ in iter_window_batches(
            image, valid_mask=valid, size=64, step=32, batch_size=7, workers=2
        )
    ]
    assert sizes
    assert max(sizes) <= 7
    assert sum(sizes) > 7


def test_streamed_prediction_matches_in_memory_prediction(tmp_path: Path):
    image = _stack(173, 211)
    valid = np.ones(image.shape[:2], dtype=bool)
    valid[:5, :] = False
    model_cfg = ModelConfig(batch_size=5, probability_threshold=0.55)
    processing = ProcessingConfig(inference_workers=2, output_rows_per_chunk=31)

    baseline_model = MeanModel()
    baseline = predict_stack(
        baseline_model, image, model_cfg, processing, valid_mask=valid
    )

    streamed_model = MeanModel()
    streamed = predict_stack(
        streamed_model,
        image,
        model_cfg,
        processing,
        valid_mask=valid,
        surface_path=tmp_path / "surface.dat",
        mask_path=tmp_path / "mask.dat",
        retain_window_details=False,
    )

    assert isinstance(streamed.probability_surface, np.memmap)
    assert isinstance(streamed.mask, np.memmap)
    assert streamed.scored_windows == []
    assert streamed.scores.size == 0
    np.testing.assert_allclose(
        streamed.probability_surface,
        baseline.probability_surface,
        rtol=0,
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_array_equal(streamed.mask, baseline.mask)
    assert streamed.window_count == baseline.window_count
    assert streamed.candidate_count == baseline.candidate_count
    assert max(streamed_model.batch_sizes) <= model_cfg.batch_size


def test_peak_patch_memory_is_bounded_by_batch_not_total_window_count(tmp_path: Path):
    image = _stack(512, 512)
    cfg = ModelConfig(batch_size=3)
    processing = ProcessingConfig(tile_size=64, overlap=0.5, inference_workers=1)
    pred = predict_stack(
        MeanModel(),
        image,
        cfg,
        processing,
        surface_path=tmp_path / "surface.dat",
        mask_path=tmp_path / "mask.dat",
        retain_window_details=False,
    )
    expected_max = cfg.batch_size * processing.tile_size * processing.tile_size * 4 * 4
    assert pred.window_count > cfg.batch_size * 10
    assert pred.peak_patch_batch_bytes <= expected_max


def test_nms_diagnostic_memory_is_bounded(tmp_path: Path):
    image = np.zeros((256, 256, 4), dtype=np.float32)
    cfg = ModelConfig(batch_size=4, probability_threshold=0.0)
    processing = ProcessingConfig(nms_diagnostic_limit=5)
    pred = predict_stack(
        MeanModel(), image, cfg, processing,
        surface_path=tmp_path / "surface.dat", mask_path=tmp_path / "mask.dat",
        retain_window_details=False,
    )
    assert pred.candidate_count > 5
    assert len(pred.candidate_windows) == 5
    assert pred.nms_diagnostics_truncated is True


def test_processing_streaming_limits_are_validated():
    for kwargs in (
        {"inference_workers": 0},
        {"nms_diagnostic_limit": 0},
        {"output_rows_per_chunk": 0},
    ):
        try:
            ProcessingConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected validation failure for {kwargs}")
