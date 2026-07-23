# Issue 6 — Sentinel-1 intermediate raster validation

Validation runs before model or weight loading. Failures are aggregated under `RasterValidationError` so operators receive all actionable corrections in one response.

## Required contract

- Four bands in order: `postVV`, `postVH`, `diffVV`, `diffVH`.
- CRS and non-identity affine transform.
- Raster dimensions at least one configured model window (64 × 64 by default).
- Resolution within 20% of the configured 10 m acquisition scale for projected CRSs.
- Robust band ranges (0.1–99.9 percentiles) within the configured dB limits.
- Orbit metadata, when present, must match the selected model.
- The raster must intersect an optional supplied ROI.

## NoData behaviour

A raster containing only NoData is rejected. A raster exceeding the configured 25% NoData fraction is rejected. Sparse NoData is filled independently in each band using the median of valid values, and the fill method and fraction are recorded.

## Small and partially covered ROIs

A raster smaller than one model window is rejected; inference does not pad beyond observed imagery. A raster that does not intersect the supplied ROI is rejected. Partial coverage is accepted, inference is restricted to available raster pixels, and a coverage warning is written to processing metadata. No predictions are extrapolated outside the raster.

## Processing metadata

`result.json` contains:

```json
{
  "processing_metadata": {
    "raster_validation": {
      "status": "passed_with_warnings",
      "errors": [],
      "warnings": [],
      "checks": {}
    }
  }
}
```

The same validation object remains available under `input_metadata.validation` for input provenance.
