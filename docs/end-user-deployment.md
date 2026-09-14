# SAR-LRA end-user deployment guide

This guide is the primary notebook-free deployment path for SAR-LRA. It covers the CLI, prepared Sentinel-1 rasters, Earth Engine acquisition, containers, the HTTP API, the async Compose stack, output interpretation, troubleshooting, scientific limitations, and citation.

> **Operational warning:** SAR-LRA is a rapid-assessment window classifier. Its outputs are candidate areas for expert review, not authoritative landslide inventories or exact landslide boundaries.

## 1. Fastest path: pull the CPU image and run the bundled sample

The published CPU image is `ghcr.io/chahelrahul/sar-lra`. For a versioned production deployment, replace `latest` with an immutable release tag once available, for example `2.0.0`.

```bash
mkdir -p results

docker pull ghcr.io/chahelrahul/sar-lra:latest

docker run --rm \
  --mount type=bind,src="$PWD/results",dst=/output \
  ghcr.io/chahelrahul/sar-lra:latest \
  predict-raster \
  --input /opt/sar-lra/examples/quickstart/ascending-4band.tif \
  --roi /opt/sar-lra/examples/quickstart/roi.geojson \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

The bundled raster is a small project-authored synthetic fixture. It exists only to prove that the installed model/pipeline can complete a prepared-raster prediction. It is not real Haiti Sentinel-1 imagery and must not be used as an accuracy benchmark.

After the command completes, inspect the request directory under `results/`. A normal run contains at least `probability.tif`, `detections.geojson`, and `result.json`; `detection-mask.tif` is present unless binary-mask output was disabled.

### Podman equivalent

Podman uses the same image and command contract:

```bash
mkdir -p results

podman pull ghcr.io/chahelrahul/sar-lra:latest

podman run --rm \
  --mount type=bind,src="$PWD/results",dst=/output \
  ghcr.io/chahelrahul/sar-lra:latest \
  predict-raster \
  --input /opt/sar-lra/examples/quickstart/ascending-4band.tif \
  --roi /opt/sar-lra/examples/quickstart/roi.geojson \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

On SELinux-enforcing hosts, Podman users may need an appropriate `:z`/`:Z` label on bind-mounted host directories. Do not relabel credential files more broadly than necessary.

## 2. CLI-only installation

A local Python installation is useful for development and environments where containers are unavailable.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[geo,inference]'
sar-lra --help
```

For Earth Engine acquisition, install the Earth Engine extras as well:

```bash
python -m pip install '.[geo,inference,earth-engine]'
```

For the API/worker stack:

```bash
python -m pip install '.[geo,inference,earth-engine,api]'
```

The notebook is not part of the production execution path. Reusable code is under `app/` and all supported commands are exposed through `sar-lra`.

## 3. Prepared-raster mode

Prepared-raster mode is the simplest production path when Sentinel-1 preprocessing is handled upstream. It requires no Google/Earth Engine credentials.

The input raster must satisfy the validator, including:

- exactly four expected bands in known order: `postVV`, `postVH`, `diffVV`, `diffVH`;
- CRS and affine transform;
- expected approximate resolution;
- finite/acceptable data values;
- orbit metadata matching the selected model;
- overlap with the optional ROI.

Host files should be mounted read-only and results mounted read/write:

```bash
mkdir -p results

docker run --rm \
  --mount type=bind,src="$PWD/input",dst=/input,readonly \
  --mount type=bind,src="$PWD/results",dst=/output \
  ghcr.io/chahelrahul/sar-lra:latest \
  predict-raster \
  --input /input/ascending-4band.tif \
  --roi /input/roi.geojson \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

Use the DESCENDING weight file and `--orbit DESCENDING` for descending acquisitions. Do not use an ascending model with descending data or vice versa.

### Re-threshold without rerunning inference

`probability.tif` is retained so a reviewer can change the operational threshold without another TensorFlow forward pass:

```bash
sar-lra threshold-raster \
  --probability-raster results/<request-id>/probability.tif \
  --threshold 0.75 \
  --output results/<request-id>/detection-mask-075.tif
```

## 4. Remote acquisition providers

SAR-LRA supports Microsoft Planetary Computer and Google Earth Engine. Use `--provider auto` to select from runtime credentials, or choose a provider explicitly. Provider secrets are runtime configuration and are never accepted in API request bodies.

### Planetary Computer

```bash
sar-lra predict \
  --provider planetary-computer-grd \
  --roi roi.geojson \
  --event-date 2024-04-03 \
  --orbit ASCENDING \
  --output-dir results
```

The default Planetary Computer path is keyless: it downloads Sentinel-1 GRD SAFE products using an anonymously issued SAS token, terrain-corrects them locally with `sarsen`, converts to dB, and forms the four-band stack. Use `planetary-computer-rtc` with `PC_SDK_SUBSCRIPTION_KEY` for the precomputed RTC collection. See `ACQUISITION_PROVIDERS.md` for its scientific compatibility caveat.

### Earth Engine mode

#### Earth Engine credentials


```bash
sar-lra predict \
  --provider earth-engine \
  --roi roi.geojson \
  --event-date 2024-04-03 \
  --orbit ASCENDING \
  --project YOUR_GOOGLE_CLOUD_PROJECT \
  --output-dir results
```

Production deployments should use Application Default Credentials, a workload identity/attached service account, a read-only mounted Earth Engine credential, or a read-only service-account JSON. Interactive authentication is intended for local development.

Example mounted service-account credential:

```bash
docker run --rm \
  -e SAR_LRA_ACQUISITION_PROVIDER=earth-engine \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gee.json \
  --mount type=bind,src="$PWD/gee.json",dst=/run/secrets/gee.json,readonly \
  --mount type=bind,src="$PWD/input",dst=/input,readonly \
  --mount type=bind,src="$PWD/results",dst=/output \
  ghcr.io/chahelrahul/sar-lra:latest \
  predict --provider earth-engine --roi /input/roi.geojson --event-date 2024-04-03 \
  --orbit ASCENDING --project YOUR_GOOGLE_CLOUD_PROJECT --output-dir /output
```

Dates must not be in the future and must fall within the supported Sentinel-1 archive period. Local ROI/date validation runs before remote acquisition.

## 5. ROI and date examples

Example ROI (`EPSG:4326` GeoJSON):

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-72.45, 18.35], [-72.35, 18.35], [-72.35, 18.45], [-72.45, 18.45], [-72.45, 18.35]]]
      }
    }
  ]
}
```

Validate it before acquisition:

```bash
sar-lra validate-roi --roi roi.geojson
```

The default deployment rejects invalid/self-intersecting polygons, coordinates outside valid longitude/latitude ranges, antimeridian-crossing requests, excessive area/dimensions/vertex counts, future dates, and dates before the supported Sentinel-1 collection.

The event date is the triggering-event date, not the acquisition date. Default imagery windows are 60 days pre-event and 12 days post-event, matching the released model workflow.

## 6. CPU and GPU expectations

### CPU

Use the CPU image by default:

```text
ghcr.io/chahelrahul/sar-lra:<tag>
```

It is the reference deployment artifact, runs without CUDA, and is appropriate for correctness checks and moderate jobs. Sliding-window patches are streamed in bounded batches, but the prepared input raster and valid-data mask still occupy memory proportional to raster size.

### NVIDIA GPU

The separate GPU image is:

```text
ghcr.io/chahelrahul/sar-lra-gpu:<tag>
```

The host must provide a compatible NVIDIA driver plus NVIDIA Container Toolkit. Example:

```bash
docker run --rm --gpus all \
  --entrypoint python \
  ghcr.io/chahelrahul/sar-lra-gpu:latest \
  /opt/sar-lra/scripts/container_gpu_smoke_test.py
```

A missing GPU is a deployment error for the GPU image smoke test; it is not treated as a successful CPU fallback. Use the CPU image when GPU infrastructure is unavailable.

## 7. HTTP API deployment

The same scientific pipeline is available through FastAPI. For a single synchronous service:

```bash
docker run --rm \
  -p 8000:8000 \
  --mount type=bind,src="$PWD/input",dst=/input,readonly \
  --mount type=bind,src="$PWD/output",dst=/output \
  --entrypoint sar-lra-api \
  ghcr.io/chahelrahul/sar-lra:latest \
  --host 0.0.0.0 --port 8000
```

Useful endpoints:

```text
GET  /healthz
GET  /readyz
POST /v1/predict-raster
POST /v1/predict
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
```

Interactive OpenAPI/Swagger documentation is served at `/docs`.

For long-running/public workflows, prefer the asynchronous job API rather than holding a synchronous HTTP request open.

## 8. Complete Compose deployment

The repository includes a reference stack with API, worker, Redis, MinIO, health checks, persistent volumes, bounded worker resources, and optional result mirroring to S3-compatible storage.

```bash
cp .env.example .env
# Set a strong MINIO_ROOT_PASSWORD and deployment-specific values.
docker compose up --build -d
```

Reference ports:

```text
API            http://localhost:8000
MinIO S3       http://localhost:9000
MinIO console  http://localhost:9001
```

CLI-only use does not require Redis or MinIO:

```bash
docker compose --profile cli run --rm cli --help
```

The `cli` profile has no `depends_on`, so this command does not intentionally start the queue/object-store services.

## 9. Output interpretation

A normal request directory can contain:

| Artifact | Meaning |
| --- | --- |
| `probability.tif` | Float32 maximum-overlapping-window **model score surface**. Not a calibrated pixel probability. |
| `detection-mask.tif` | Binary thresholded derivative of the probability surface. |
| `detections.geojson` | EPSG:4326 candidate-area polygons; valid empty FeatureCollection if no detections. |
| `detections.gpkg` | Optional candidate-area vectors preserving raster CRS. |
| `result.json` | Machine-readable provenance, configuration, validation, model checksum, and processing metadata. |
| intermediate Sentinel-1 GeoTIFF | Remote-acquisition modes; cached/acquired four-band input. |

Important interpretation rules:

1. The classifier scores `64 × 64` windows. It does not segment exact landslide boundaries.
2. The `probability.tif` value is the maximum score of windows covering a valid pixel; it is not a calibrated per-pixel landslide probability.
3. Vector polygons represent thresholded candidate areas, not mapped landslide outlines.
4. A low detection count does not prove that no landslides occurred.
5. A high score does not prove that a landslide is present.
6. All outputs require expert review against source imagery, terrain/acquisition geometry, and independent information.

## 10. Troubleshooting

### `input/configuration validation error`

Run:

```bash
sar-lra validate-roi --roi roi.geojson
```

For prepared rasters, check band count/order, CRS, transform, resolution, NoData fraction, value ranges, orbit tag, and ROI intersection. Validation details are written to processing metadata when a run proceeds.

### Earth Engine authentication fails

Check that:

- a Google Cloud project is supplied;
- Earth Engine access is enabled for the identity/project;
- `GOOGLE_APPLICATION_CREDENTIALS` points to a file visible **inside** the container, not only on the host;
- the credential mount is read-only;
- interactive authentication is not being expected in a headless worker.

Do not paste credential JSON or private keys into logs/issues.

### Earth Engine reports no scenes for the requested orbit/window

The local validator cannot guarantee archive coverage. Try the scientifically appropriate orbit only after checking Sentinel-1 availability for the ROI and pre/post windows. Do not silently substitute an opposite-orbit model.

### Container cannot write results

The image runs as non-root UID/GID `10001`. Ensure the host output directory is writable by the container user. Docker/Podman bind-mount semantics and host security policy still apply.

### Podman bind mount is denied on SELinux

Use an appropriate SELinux volume label (`:z` or `:Z`) where required by the host policy. Credential files should remain as narrowly accessible as possible.

### GPU image sees no GPU

Verify the NVIDIA host driver, NVIDIA Container Toolkit, `--gpus all`, and the GPU smoke test. If accelerator infrastructure is not available, use the CPU image instead of treating the missing GPU as success.

### Job API returns HTTP 429

The synchronous inference semaphore or asynchronous queue is saturated. Respect `Retry-After`. For async deployments, review `SAR_LRA_MAX_QUEUED_JOBS`, worker capacity, job timeout, and memory limits before increasing concurrency.

### Worker job fails with `job_timeout`

The configured execution deadline was exceeded. Inspect workload size, acquisition availability, machine resources, and `SAR_LRA_JOB_TIMEOUT_SECONDS`. Avoid increasing the timeout without also checking memory/disk limits.

### Disk admission check fails

Free space is below `SAR_LRA_MIN_FREE_DISK_MB`. Remove stale deployment data or provision more storage. Do not disable the floor on a shared/public deployment without an equivalent storage safeguard.

### No detections

An empty `detections.geojson` is a valid result. It does not establish landslide absence. Review the score raster and source data and consider the documented threshold/model limitations.

## 11. Scientific limitations

The released model was developed for co-seismic landslide rapid assessment from Sentinel-1 SAR. Known or plausible limitations include:

- riverbeds and other earthquake-related surface changes can produce false positives;
- SAR shadow, layover, foreshortening, and incidence-angle effects are important in steep terrain;
- orbit/relative-orbit inconsistency can change geometry and model response;
- small landslides may be missed because the classifier operates on fixed windows;
- seasonal vegetation, flooding, snowmelt, agriculture, construction, and other non-landslide changes can alter SAR backscatter;
- transferability to geographic/geologic/land-cover conditions absent from training is uncertain;
- rainfall-triggered landslides are not established as a validated use case by the released evidence;
- operational Earth Engine preprocessing does not currently reproduce every shadow/layover treatment discussed in the study;
- the output score has not been demonstrated to be a calibrated landslide probability.

For scientific provenance, preprocessing requirements, held-out evaluation context, and the complete warning set, read `MODEL_CARD.md` before operational use.

## 12. Citation and acknowledgement

If SAR-LRA or the released model is used in scientific work, cite the associated publication:

> Nava, L., Mondini, A., Bhuyan, K., Fang, C., Monserrat, O., Novellino, A., and Catani, F. (2026). *Sentinel-1 SAR-based globally distributed co-seismic landslide detection by deep neural networks*. Geoscientific Model Development, 19, 167–185. https://doi.org/10.5194/gmd-19-167-2026

Also record the model version and weight SHA-256 from `result.json` so the exact inference artifact can be traced.

The repository currently does **not** contain an explicit rights-holder statement confirming that the trained weight files are covered by the repository MIT license. Do not represent the weight-license status more strongly than `MODEL_CARD.md` supports.

## 13. More detailed operator documentation

This guide is intentionally task-oriented. Detailed issue-specific documentation remains available for operators and maintainers:

- `docs/configuration.md` — configuration schema and precedence;
- `docs/container-contract.md` — filesystem/input/output contract;
- `docs/issue-14-roi-validation-processing-limits.md` — ROI safety limits;
- `docs/issue-16-earth-engine-authentication.md` — credential patterns;
- `docs/issue-17-gpu-docker.md` — GPU deployment;
- `docs/issue-18-streamed-inference.md` — memory behavior;
- `docs/issue-19-fastapi-service.md` — FastAPI service;
- `docs/issue-20-asynchronous-jobs.md` — async job API;
- `docs/issue-21-resource-timeout-cancellation.md` — worker safety controls;
- `docs/issue-22-ghcr-release.md` — image publication/signatures;
- `docs/issue-23-compose-deployment.md` — full local/reference stack.
