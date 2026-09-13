# Issue 8 — Sliding-window edge coverage

## Decision

Inference windows now cover every valid source pixel. For raster dimensions that are not aligned to the configured step, the final window is anchored to the final possible source position rather than stopping at the last regular step.

With the default 64-pixel model window and 32-pixel step, a 100-pixel axis therefore uses starts `0, 32, 36`. A 65-pixel axis uses `0, 1`.

## Small rasters

A raster smaller than the 64×64 model input is accepted after Issue 8. The observed pixels are edge-padded to 64×64 only for the model input tensor. The `Window` coordinates remain clipped to the true source-raster extent.

Edge padding is used instead of a constant zero because the inputs are Sentinel-1 dB values and a synthetic 0 dB border would be far outside the typical signal represented in most model patches.

The raster-validation metadata records `edge_padding_required: true` and a warning when this behavior is used.

## Valid-data mask

The reader retains a two-dimensional valid-data mask. It combines:

- valid pixels from all four source bands;
- the submitted ROI mask, when an ROI is supplied.

Sparse NoData values may still be filled for the dense TensorFlow input, but filled pixels are not treated as valid output pixels. Final binary detections are masked so NoData, outside-ROI, and purely padded pixels cannot become output detections.

## Geospatial behavior

`Window.x0/y0/x1/y1` always refer to real source-raster pixel coordinates, with `x1/y1` exclusive. Padding exists only in the model patch array and does not extend the raster transform or output geometry.

This preserves the original raster transform for GeoTIFF and vector post-processing.

## Regression cases

Unit tests cover:

- 64×64;
- 65×65;
- 100×100;
- non-square 80×130;
- sub-window 32×48;
- output masking outside the valid-data/ROI mask.
