# Issue 13 — `sar-lra` command-line interface

The CLI is the supported operator-facing entry point for validation, acquisition, prepared-raster inference, end-to-end Earth Engine inference, and probability re-thresholding.

## Commands

### Validate an ROI

```bash
sar-lra validate-roi --roi roi.geojson
```

This command performs lightweight GeoJSON structure/type validation and deliberately does not import TensorFlow or Earth Engine. More extensive area, topology, antimeridian, vertex-count, date, and coverage limits belong to Issue 14.

### Acquire Sentinel-1 input only

```bash
sar-lra acquire \
  --roi roi.geojson \
  --event-date 2024-04-03 \
  --orbit ASCENDING \
  --output-dir imagery
```

The stable user-facing file is `imagery/ascending.tif` (or `descending.tif`). Acquisition cache files are kept under `imagery/.cache/`, so generated files do not escape the requested output directory.

### Predict from a prepared raster

```bash
sar-lra predict-raster \
  --input imagery/ascending.tif \
  --orbit ASCENDING \
  --weights model/weights/<ascending-weight>.hdf5 \
  --output-dir results
```

The historical `--ascending FILE` and `--descending FILE` switches remain accepted for one compatibility release. New integrations should use `--input` plus `--orbit`.

### Acquire and predict one or both passes

```bash
sar-lra predict \
  --roi roi.geojson \
  --event-date 2024-04-03 \
  --orbits ASCENDING,DESCENDING \
  --ascending-weights model/weights/<ascending-weight>.hdf5 \
  --descending-weights model/weights/<descending-weight>.hdf5 \
  --output-dir results
```

A single-orbit run may use `--weights`. When running from a source checkout, the CLI can also discover the repository's bundled orbit-specific weight when exactly one matching file is present. Installed wheels should pass weight paths explicitly because trained weights are not part of the Python package contract.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 2 | CLI syntax/argument parsing error (`argparse`) |
| 3 | Input/configuration/validation error |
| 4 | Earth Engine/acquisition/dependency/authentication failure |
| 5 | Processing/inference failure |
| 130 | Interrupted by Ctrl+C |

Errors are emitted to stderr. Successful machine-readable command results are emitted as JSON to stdout.

## Structured diagnostic logs

Use `--log-format json` after the command name:

```bash
sar-lra validate-roi --roi roi.geojson --log-format json
```

Diagnostics are JSON Lines on stderr, leaving stdout available for the final JSON result. The text mode is the default for interactive use.

## Resource requirements

`sar-lra --help` and subcommand help state the runtime requirements. Prepared-raster inference requires TensorFlow and geospatial libraries. Earth Engine acquisition additionally requires the Earth Engine/geemap dependencies, credentials, network access, and sufficient remote/export quota. Memory use scales with batch size and the number of 64×64×4 windows.

## Output containment

Commands with `--output-dir` place generated run artifacts and Earth Engine cache entries below that directory. After a pipeline run, the CLI verifies every declared generated artifact remains under the requested output root and fails if an artifact escapes it. Input rasters, ROI files, configuration files, and weight files may of course reside elsewhere because they are inputs rather than generated outputs.
