import numpy as np

from app.inference.windows import (
    iou_nms,
    legacy_overlap_nms,
    pairwise_iou,
    pairwise_legacy_overlap,
)


def test_iou_metric_known_boxes():
    box = np.array([0, 0, 10, 10], dtype=float)
    others = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    values = pairwise_iou(box, others)
    # Intersection=50, union=150.
    np.testing.assert_allclose(values, [1 / 3, 0.0])


def test_legacy_metric_is_asymmetric_and_not_iou():
    large = np.array([0, 0, 10, 10], dtype=float)
    small = np.array([[0, 0, 5, 10]], dtype=float)
    legacy = pairwise_legacy_overlap(large, small)[0]
    iou = pairwise_iou(large, small)[0]
    assert legacy == 1.0
    assert iou == 0.5


def test_iou_nms_prefers_highest_score_deterministically():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]])
    scores = np.array([0.7, 0.9, 0.8])
    kept = iou_nms(boxes, scores, overlap_threshold=0.5)
    np.testing.assert_array_equal(kept, [[1, 1, 11, 11], [30, 30, 40, 40]])


def test_legacy_and_iou_suppression_can_differ():
    # First box fully contains the second. Legacy overlap relative to the smaller
    # comparison box is 1.0 while IoU is only 0.5.
    boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 10]])
    scores = np.array([0.9, 0.8])
    legacy = legacy_overlap_nms(boxes, scores, overlap_threshold=0.6)
    iou = iou_nms(boxes, scores, overlap_threshold=0.6)
    assert len(legacy) == 1
    assert len(iou) == 2
