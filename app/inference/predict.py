from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from app.config import ModelConfig, ProcessingConfig
from app.inference.windows import Window, iou_nms, legacy_overlap_nms, sliding_windows
from app.preprocessing.sentinel1 import validate_stack


@dataclass(frozen=True, slots=True)
class ScoredWindow:
    window: Window
    probability: float


@dataclass(slots=True)
class Prediction:
    scores: np.ndarray
    scored_windows: list[ScoredWindow]
    candidate_windows: list[Window]
    selected_windows: list[Window]
    legacy_selected_windows: list[Window]
    iou_selected_windows: list[Window]
    probability_surface: np.ndarray
    mask: np.ndarray
    window_count: int = 0
    candidate_count: int = 0
    nms_diagnostics_truncated: bool = False
    peak_patch_batch_bytes: int = 0


def _prepare_patch(patch: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(patch, dtype=np.float32)


def iter_window_batches(
    stack: np.ndarray,
    *,
    valid_mask: np.ndarray,
    size: int,
    step: int,
    batch_size: int,
    workers: int = 1,
) -> Iterator[tuple[list[Window], np.ndarray]]:
    """Yield bounded batches of valid model windows.

    At most ``batch_size`` patches are materialized at once. ``workers`` only
    controls preparation of patches already in the current bounded batch, so it
    cannot increase the number of queued batches or make memory scale with ROI size.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    windows: list[Window] = []
    patches: list[np.ndarray] = []

    def emit() -> tuple[list[Window], np.ndarray]:
        if workers == 1:
            prepared = [_prepare_patch(p) for p in patches]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                prepared = list(executor.map(_prepare_patch, patches))
        return list(windows), np.stack(prepared, axis=0)

    for window, patch in sliding_windows(stack, size=size, step=step):
        if not valid_mask[window.y0:window.y1, window.x0:window.x1].any():
            continue
        windows.append(window)
        patches.append(patch)
        if len(patches) >= batch_size:
            yield emit()
            windows.clear()
            patches.clear()

    if patches:
        yield emit()


def _surface_array(shape: tuple[int, int], path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    surface = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
    surface[:] = 0.0
    return surface


def _mask_array(shape: tuple[int, int], path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    mask[:] = 0
    return mask


def aggregate_window_probabilities(
    shape: tuple[int, int],
    scored_windows: list[ScoredWindow],
    *,
    valid_mask: np.ndarray | None = None,
    method: str = "maximum",
) -> np.ndarray:
    """Aggregate window probabilities in memory (compatibility helper)."""
    if method != "maximum":
        raise ValueError("Only 'maximum' probability aggregation is currently supported")
    surface = np.zeros(shape, dtype=np.float32)
    for item in scored_windows:
        region = surface[item.window.y0:item.window.y1, item.window.x0:item.window.x1]
        np.maximum(region, np.float32(item.probability), out=region)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != surface.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match raster shape {surface.shape}")
        surface[~valid] = np.nan
    return surface


def threshold_probability_surface(
    probability_surface: np.ndarray,
    threshold: float,
    *,
    valid_mask: np.ndarray | None = None,
    output: np.ndarray | None = None,
    rows_per_chunk: int = 1024,
) -> np.ndarray:
    """Derive a binary mask without materializing another full-raster temporary."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability threshold must be between 0 and 1")
    probability_surface = np.asarray(probability_surface)
    if probability_surface.ndim != 2:
        raise ValueError("probability_surface must be a two-dimensional array")
    if output is None:
        output = np.zeros(probability_surface.shape, dtype=np.uint8)
    if output.shape != probability_surface.shape:
        raise ValueError("output shape must match probability_surface")
    valid = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid is not None and valid.shape != probability_surface.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match raster shape {probability_surface.shape}")
    for y0 in range(0, probability_surface.shape[0], rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, probability_surface.shape[0])
        chunk = np.asarray(probability_surface[y0:y1], dtype=np.float32)
        chunk_mask = np.isfinite(chunk) & (chunk >= threshold)
        if valid is not None:
            chunk_mask &= valid[y0:y1]
        output[y0:y1] = chunk_mask.astype(np.uint8, copy=False)
    if isinstance(output, np.memmap):
        output.flush()
    return output


def probability_surface_stats(surface: np.ndarray, *, rows_per_chunk: int = 1024) -> dict[str, float | None]:
    """Compute final surface statistics in bounded row chunks."""
    count = 0
    total = 0.0
    minimum = np.inf
    maximum = -np.inf
    for y0 in range(0, surface.shape[0], rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, surface.shape[0])
        chunk = np.asarray(surface[y0:y1], dtype=np.float32)
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        count += int(finite.size)
        total += float(finite.sum(dtype=np.float64))
        minimum = min(minimum, float(finite.min()))
        maximum = max(maximum, float(finite.max()))
    return {
        "minimum": None if count == 0 else minimum,
        "maximum": None if count == 0 else maximum,
        "mean": None if count == 0 else total / count,
    }


def predict_stack(
    model,
    stack: np.ndarray,
    model_config: ModelConfig,
    processing_config: ProcessingConfig,
    *,
    valid_mask: np.ndarray | None = None,
    surface_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    retain_window_details: bool = True,
) -> Prediction:
    """Run bounded-memory sliding-window inference.

    Patches are predicted incrementally by ``model.batch_size``. When paths are
    supplied, the probability surface and binary mask are disk-backed memmaps,
    so production inference does not require full output rasters in RAM.
    """
    stack = validate_stack(stack)
    height, width = stack.shape[:2]
    if valid_mask is None:
        valid_mask = np.ones((height, width), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (height, width):
            raise ValueError(f"valid_mask shape {valid_mask.shape} does not match raster shape {(height, width)}")
    if processing_config.probability_aggregation != "maximum":
        raise ValueError("Only 'maximum' probability aggregation is currently supported")

    surface = _surface_array((height, width), surface_path)
    all_scored: list[ScoredWindow] = []
    all_scores: list[np.ndarray] = []
    diagnostic_candidates: list[ScoredWindow] = []
    window_count = 0
    candidate_count = 0
    peak_patch_batch_bytes = 0
    nms_truncated = False

    for windows, patches in iter_window_batches(
        stack,
        valid_mask=valid_mask,
        size=processing_config.tile_size,
        step=processing_config.window_step,
        batch_size=model_config.batch_size,
        workers=processing_config.inference_workers,
    ):
        peak_patch_batch_bytes = max(peak_patch_batch_bytes, int(patches.nbytes))
        batch_scores = np.asarray(
            model.predict(patches, batch_size=len(patches), verbose=0), dtype=np.float32
        ).reshape(-1)
        if len(batch_scores) != len(windows):
            raise RuntimeError("model.predict returned a different number of scores than input windows")
        window_count += len(windows)
        if retain_window_details:
            all_scores.append(batch_scores.copy())

        for window, score in zip(windows, batch_scores):
            probability = float(score)
            region = surface[window.y0:window.y1, window.x0:window.x1]
            np.maximum(region, np.float32(probability), out=region)
            item = ScoredWindow(window=window, probability=probability)
            if retain_window_details:
                all_scored.append(item)
            if probability >= model_config.probability_threshold:
                candidate_count += 1
                if len(diagnostic_candidates) < processing_config.nms_diagnostic_limit:
                    diagnostic_candidates.append(item)
                else:
                    nms_truncated = True

    if window_count == 0:
        surface[:] = np.nan
    else:
        # Invalid/ROI-excluded pixels are represented as NaN on the probability surface.
        for y0 in range(0, height, processing_config.output_rows_per_chunk):
            y1 = min(y0 + processing_config.output_rows_per_chunk, height)
            block = surface[y0:y1]
            block[~valid_mask[y0:y1]] = np.nan
    if isinstance(surface, np.memmap):
        surface.flush()

    mask = _mask_array((height, width), mask_path)
    threshold_probability_surface(
        surface,
        model_config.probability_threshold,
        valid_mask=valid_mask,
        output=mask,
        rows_per_chunk=processing_config.output_rows_per_chunk,
    )

    candidates = [item.window for item in diagnostic_candidates]
    boxes = np.asarray([[w.x0, w.y0, w.x1, w.y1] for w in candidates], dtype=np.int32)
    candidate_scores = np.asarray([item.probability for item in diagnostic_candidates], dtype=np.float32)
    legacy_kept = legacy_overlap_nms(boxes, candidate_scores, model_config.nms_overlap)
    iou_kept = iou_nms(boxes, candidate_scores, model_config.nms_overlap)
    legacy_selected = [Window(*map(int, box)) for box in legacy_kept]
    iou_selected = [Window(*map(int, box)) for box in iou_kept]

    scores = np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)
    return Prediction(
        scores=scores,
        scored_windows=all_scored,
        candidate_windows=candidates,
        selected_windows=iou_selected,
        legacy_selected_windows=legacy_selected,
        iou_selected_windows=iou_selected,
        probability_surface=surface,
        mask=mask,
        window_count=window_count,
        candidate_count=candidate_count,
        nms_diagnostics_truncated=nms_truncated,
        peak_patch_batch_bytes=peak_patch_batch_bytes,
    )
