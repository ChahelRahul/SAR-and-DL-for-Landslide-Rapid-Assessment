# Issue 9 — Preserve prediction probabilities through post-processing

## Purpose

SAR-LRA V2 produces one sigmoid score for each 64×64 model window. Previous operational code discarded most of that information by applying the configured threshold immediately and writing retained windows into a binary raster. Issue 9 keeps the model scores as a first-class output so thresholds can be changed without rerunning TensorFlow inference.

## Probability semantics

The model does **not** produce a per-pixel segmentation probability. Each score is a probability-like value associated with an input window. The probability raster is therefore a derived coverage surface, not a calibrated probability that each individual pixel is a landslide pixel.

For every valid source pixel, SAR-LRA now records the **maximum** score of all model windows covering that pixel:

```text
pixel probability = max(score of every model window covering that pixel)
```

Maximum aggregation was selected for this release because it:

- retains the strongest model response;
- is deterministic and simple to reproduce;
- does not dilute a positive response merely because several lower-scoring overlapping windows also cover the pixel;
- keeps the stored surface independent of the configured binary threshold.

The aggregation method is written into output and processing metadata as `maximum`. Other aggregation methods are intentionally rejected until they are scientifically evaluated.

## Outputs

Every successful inference writes:

```text
probability.tif
```

This is a float32 GeoTIFF containing the maximum window score for each valid pixel. Invalid, NoData and outside-ROI pixels are stored as raster NoData. No binary threshold is applied to this file.

By default inference also writes:

```text
detection-mask.tif
```

The mask is derived from `probability.tif` using `model.probability_threshold`. It can be disabled with:

```bash
sar-lra predict-raster ... --no-binary-mask
```

or in YAML:

```yaml
processing:
  write_binary_mask: false
```

The binary file is therefore a convenience output, not the canonical model-score output.

## Re-thresholding without inference

A saved probability raster can be thresholded later without loading TensorFlow or model weights:

```bash
sar-lra threshold-raster \
  --probability-raster results/<request-id>/probability.tif \
  --threshold 0.75 \
  --output results/<request-id>/detection-mask-075.tif
```

This command only reads the probability GeoTIFF and writes a derived binary mask.

## Vector confidence

Thresholded vector detections now include:

- `probability` — maximum probability surface value inside the vector feature;
- `orbit`;
- `rel_orbit` when available from the intermediate raster;
- `model_ver`;
- configured threshold;
- model-weight checksum.

The vector geometry still represents connected thresholded candidate areas, not verified landslide boundaries. Issue 10 will review NMS and detection geometry methodology explicitly.

## Relationship to NMS

Historical candidate-window NMS values are still calculated and retained in the `Prediction` object for regression comparison and Issue 10. NMS no longer defines the probability raster or binary raster. Binary pixels are derived directly from the stored probability surface so a new threshold can be applied without rerunning model inference.

## Metadata

`result.json` includes probability statistics under:

```json
{
  "processing_metadata": {
    "probability_surface": {
      "aggregation": "maximum",
      "threshold": 0.6,
      "window_count": 123,
      "minimum": 0.02,
      "maximum": 0.94,
      "mean": 0.31
    }
  }
}
```

The probability raster tags explicitly use `output_type=probability_surface` and `threshold_applied=false`. The binary raster uses `output_type=binary_detection_mask`, records the applied threshold, and records that it is derived from the probability surface.
