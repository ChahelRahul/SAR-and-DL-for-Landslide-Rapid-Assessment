# CLI reference

The installed executable is `sar-lra`.

## `validate-roi`

Validate GeoJSON and configured ROI limits before acquisition.

```bash
sar-lra validate-roi --roi /input/roi.geojson
```

## `acquire`

Acquire and build the four-band Sentinel-1 intermediate raster without running the model.

```bash
sar-lra acquire \
  --provider auto \
  --roi /input/roi.geojson \
  --event-date 2025-08-15 \
  --orbit ASCENDING \
  --output-dir /output
```

Providers: `auto`, `planetary-computer`, `earth-engine`. `--project` and `--authenticate` apply to Earth Engine; interactive authentication is intended for local development, not containers.

## `predict`

Acquire Sentinel-1 and run the complete inference/post-processing pipeline.

```bash
sar-lra predict \
  --provider planetary-computer \
  --roi /input/roi.geojson \
  --event-date 2025-08-15 \
  --orbit ASCENDING \
  --output-dir /output
```

Use `--orbits ASCENDING,DESCENDING` for both passes and supply orbit-specific weights when not using bundled weights.

## `predict-raster`

Provider-independent inference from a validated four-band GeoTIFF.

```bash
sar-lra predict-raster \
  --input /input/ascending-4band.tif \
  --orbit ASCENDING \
  --output-dir /output
```

Optional `--roi` performs coverage validation against a GeoJSON ROI.

## `threshold-raster`

Create a binary mask from an existing probability raster without running the model again.

```bash
sar-lra threshold-raster \
  --probability-raster /output/probability.tif \
  --threshold 0.5 \
  --output /output/mask.tif
```

## Common model/processing options

Depending on the command, common options include `--config`, `--weights`, `--request-id`, `--orbit`, `--probability-threshold`, `--batch-size`, `--inference-workers`, `--pre-days`, `--post-days`, `--scale-m`, `--tile-size`, `--overlap`, ROI limits, vector-output controls, and chunk sizes.

Run `sar-lra <command> --help` for the authoritative option list bundled with that release.
