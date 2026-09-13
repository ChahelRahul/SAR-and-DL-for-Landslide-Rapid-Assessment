# Issue 10 — NMS and detection geometry methodology

## Decision

SAR-LRA v2 produces one score per 64×64 model window. The historical notebook then applied an asymmetric overlap suppression rule and represented retained positive windows as rectangles. That operation is retained only for reproducibility diagnostics.

Operational outputs are now derived from the thresholded probability surface introduced in Issue 9. NMS does **not** define the GeoTIFF mask or vector geometries.

## Historical overlap metric

The released implementation suppresses a comparison box when:

`intersection(current, comparison) / area(comparison) > threshold`

This is asymmetric and is not intersection-over-union (IoU). A small box completely contained inside a larger box can have overlap 1.0 in one direction even though its IoU is substantially lower.

The package exposes this behavior as `legacy_overlap_nms()` so it cannot be confused with conventional NMS.

## Conventional comparator

`iou_nms()` uses score-ordered conventional IoU:

`intersection / union`

The pipeline records retained counts for both algorithms under `processing_metadata.nms_diagnostics`. These values are diagnostic only and do not alter operational outputs.

## Geometry semantics

Vector features generated from the thresholded probability surface represent **thresholded candidate areas**. They are not validated individual landslide polygons and must not be interpreted as exact landslide boundaries.

Every vector feature records:

- `geometry_semantics=thresholded_candidate_area`
- `boundary_semantics=not_exact_landslide_boundary`

The model card and processing metadata repeat this distinction.

## Why probability-surface aggregation is preferred

A rectangular positive window only states that the model classified that image patch positively. Converting the full window directly into a landslide polygon overstates the model's spatial precision. Aggregating scores spatially and thresholding the resulting surface preserves the model's window-level nature while avoiding the claim that the rectangle itself is an observed landslide boundary.

This still does not make the thresholded surface a segmentation model. Its polygons remain candidate areas requiring expert interpretation.
