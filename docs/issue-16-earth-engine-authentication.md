# Issue 16 — secure Earth Engine authentication for containers

SAR-LRA does not copy Google or Earth Engine credentials into the image. Earth Engine acquisition uses credentials supplied at runtime.

## Supported authentication patterns

### 1. Read-only mounted service-account JSON

```bash
docker run --rm \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gee.json \
  -v "$PWD/gee.json:/run/secrets/gee.json:ro" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/results:/output" \
  sar-lra:cpu acquire \
  --project YOUR_GOOGLE_CLOUD_PROJECT \
  --roi /input/roi.geojson \
  --event-date 2024-04-03 \
  --orbit ASCENDING \
  --output-dir /output
```

The service account must have access to Earth Engine and the specified Cloud project must be registered/usable for Earth Engine.

### 2. Application Default Credentials (ADC)

`ee.Initialize()` supports ambient Google credentials. Set `GOOGLE_APPLICATION_CREDENTIALS` to a mounted ADC/service-account file, or provide ADC through the hosting environment. Do not bake ADC files into an image layer.

### 3. Host-side Earth Engine credentials for local use

For local development, mount the existing Earth Engine credential directory into the container user's home:

```bash
docker run --rm \
  -v "$HOME/.config/earthengine:/home/sarlra/.config/earthengine:ro" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/results:/output" \
  sar-lra:cpu acquire --project YOUR_PROJECT --roi /input/roi.geojson \
  --event-date 2024-04-03 --orbit ASCENDING --output-dir /output
```

`--authenticate` remains available only for explicit interactive local use. It should not be used as a production container authentication mechanism.

### 4. Workload Identity / attached service account

Cloud deployments should prefer short-lived ambient credentials such as Workload Identity or an attached runtime service account. No credential file is required in the image. The Earth Engine client obtains credentials through Google Auth when `ee.Initialize()` runs.

## Secret handling

- `.dockerignore` excludes common credential/key locations and filenames.
- Credential JSON is never copied by the Dockerfile.
- Mounted service-account JSON is read only to fail early on malformed files; its content is never logged.
- CLI diagnostics redact common token/private-key fields and the path referenced by `GOOGLE_APPLICATION_CREDENTIALS`.
- Authentication failures explain the supported remediation paths without echoing credential contents.

## Prepared-raster isolation

`predict-raster`, `threshold-raster`, and `validate-roi` do not require Google credentials. Earth Engine and its authentication helper are only used by Earth Engine acquisition paths.

## Cloud project requirement

A Cloud project should be supplied with `--project`. Modern Earth Engine initialization requires an appropriate registered Cloud project for normal API use. The credential identity must have permission to use it.
