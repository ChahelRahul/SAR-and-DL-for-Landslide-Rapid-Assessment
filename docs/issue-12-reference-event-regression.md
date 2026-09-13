# Issue 12 — Reference-event regression tests

## Purpose

Regression tests now detect unintended drift in raster validation, NoData handling, window enumeration, probability aggregation, thresholding, NMS diagnostics, and vector geometry creation without requiring Earth Engine.

## Reference cases

Two compact fixtures are stored under `tests/fixtures/reference/`:

| Fixture | Public event anchor | Orbit contract | Raster |
| --- | --- | --- | --- |
| Haiti 2021 | USGS M7.2 Nippes, 14 Aug 2021 | ASCENDING | 96×80, 10 m |
| Sumatra 2022 | USGS M6.1 near Bukittinggi, 25 Feb 2022 | DESCENDING | 96×80, 10 m |

The scientific paper reports Haiti (2021) and Sumatra (2022) as unseen-event evaluations. The fixture pixels themselves are deterministic synthetic values, not Sentinel-1 observations. This keeps the test assets compact and unambiguously redistributable.

## Stored expectations

Each case commits expected:

- raster width, height and band order;
- model-window count;
- minimum, maximum and mean aggregated probability;
- thresholded candidate-window count;
- IoU-NMS diagnostic retained count;
- positive pixel count;
- vector feature count;
- projected output geometry bounds.

The always-on regression suite uses a deterministic NumPy model surrogate. This deliberately isolates pipeline/preprocessing drift from TensorFlow kernel/version variation and means the tests run on ordinary CI runners without Earth Engine or TensorFlow.

## Numerical tolerances

Committed baseline files define:

- CPU absolute tolerance: `1e-6`;
- GPU absolute tolerance: `2e-5`;
- GPU relative tolerance: `2e-5`.

The looser GPU tolerance is reserved for released-model TensorFlow regression jobs because device kernels can differ slightly in floating-point reduction order. Discrete counts and geometry bounds must remain exact within the documented geospatial tolerance unless an intentional algorithm change is reviewed.

## Updating baselines

Do not regenerate expected files merely to make CI green. A baseline change must accompany a reviewed scientific/software change explaining why patch counts, probabilities, detections, or bounds changed.

Regenerate project-authored input fixtures with:

```bash
python tests/fixtures/reference/generate_fixtures.py
```

The normal test suite must not call Earth Engine and must not download imagery.

## Scope

These fixtures are regression assets, not accuracy benchmarks. They do not establish model performance on Haiti or Sumatra. Accuracy claims must continue to refer to the study and independently validated observational datasets.
