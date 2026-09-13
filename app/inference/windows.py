from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Window:
    """Pixel-space extent of the real raster covered by one model patch.

    x1/y1 are exclusive and are always clipped to the source raster. A model
    patch may be padded beyond this extent when the source raster is smaller
    than the configured model window.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def axis_starts(length: int, size: int, step: int) -> list[int]:
    """Return starts that cover every source pixel, including the final edge."""
    if length <= 0:
        raise ValueError("image dimensions must be positive")
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, step))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def sliding_windows(
    image: np.ndarray,
    size: int,
    step: int,
    *,
    pad_mode: str = "edge",
) -> Iterator[tuple[Window, np.ndarray]]:
    """Yield fixed-size model patches with complete source-pixel coverage."""
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = image.shape[:2]
    y_starts = axis_starts(height, size, step)
    x_starts = axis_starts(width, size, step)

    for y in y_starts:
        for x in x_starts:
            y1 = min(y + size, height)
            x1 = min(x + size, width)
            patch = image[y:y1, x:x1]
            pad_bottom = size - patch.shape[0]
            pad_right = size - patch.shape[1]
            if pad_bottom or pad_right:
                pad_width = [(0, pad_bottom), (0, pad_right)] + [(0, 0)] * (patch.ndim - 2)
                patch = np.pad(patch, pad_width, mode=pad_mode)
            yield Window(x, y, x1, y1), patch


def _validate_nms_inputs(
    boxes: np.ndarray,
    scores: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (n, 4)")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("boxes must use exclusive x1/y1 coordinates greater than x0/y0")
    if scores is None:
        # Deterministic fallback used only for legacy compatibility. Prefer real
        # model scores for any scientific comparison.
        scores = np.arange(len(boxes), dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(boxes):
            raise ValueError("scores length must match boxes length")
    return boxes, scores


def pairwise_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection-over-union for exclusive-coordinate boxes."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    xx0 = np.maximum(box[0], boxes[:, 0])
    yy0 = np.maximum(box[1], boxes[:, 1])
    xx1 = np.minimum(box[2], boxes[:, 2])
    yy1 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + other_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def pairwise_legacy_overlap(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Historical SAR-LRA overlap metric: intersection / comparison-box area.

    This asymmetric metric is retained for reproducibility only. It is not IoU.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    xx0 = np.maximum(box[0], boxes[:, 0])
    yy0 = np.maximum(box[1], boxes[:, 1])
    xx1 = np.minimum(box[2], boxes[:, 2])
    yy1 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
    other_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.divide(intersection, other_area, out=np.zeros_like(intersection), where=other_area > 0)


def _nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray | None,
    overlap_threshold: float,
    *,
    metric: str,
) -> np.ndarray:
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1")
    boxes, scores = _validate_nms_inputs(boxes, scores)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    # Stable descending score order. Lower original index wins exact ties.
    order = np.lexsort((np.arange(len(scores)), -scores))
    selected: list[int] = []
    metric_fn = pairwise_iou if metric == "iou" else pairwise_legacy_overlap
    while len(order):
        current = int(order[0])
        selected.append(current)
        remaining = order[1:]
        if not len(remaining):
            break
        overlap = metric_fn(boxes[current], boxes[remaining])
        order = remaining[overlap <= overlap_threshold]
    return np.asarray(selected, dtype=np.int64)


def iou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    overlap_threshold: float,
) -> np.ndarray:
    """Return boxes retained by conventional score-ordered IoU NMS."""
    boxes_arr, _ = _validate_nms_inputs(boxes, scores)
    indices = _nms_indices(boxes_arr, scores, overlap_threshold, metric="iou")
    return boxes_arr[indices].astype(np.int32)


def legacy_overlap_nms(
    boxes: np.ndarray,
    scores: np.ndarray | None,
    overlap_threshold: float,
) -> np.ndarray:
    """Return boxes retained by the historical asymmetric overlap rule."""
    boxes_arr, _ = _validate_nms_inputs(boxes, scores)
    indices = _nms_indices(boxes_arr, scores, overlap_threshold, metric="legacy")
    return boxes_arr[indices].astype(np.int32)


def non_max_suppression(boxes: np.ndarray, overlap_threshold: float) -> np.ndarray:
    """Deprecated compatibility alias for historical SAR-LRA NMS.

    New code should call :func:`legacy_overlap_nms` or :func:`iou_nms`
    explicitly so the overlap semantics cannot be confused.
    """
    return legacy_overlap_nms(boxes, None, overlap_threshold)
