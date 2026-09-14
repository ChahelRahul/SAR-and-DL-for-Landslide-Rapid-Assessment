# HTTP API reference

The HTTP service is implemented with FastAPI. Default base URL in the reference Compose deployment is `http://localhost:8000`.

Interactive documentation:

- `GET /docs` — Swagger UI
- `GET /redoc` — ReDoc
- `GET /openapi.json` — generated OpenAPI schema

## Health

### `GET /healthz`

Liveness endpoint. Returns service name and package version.

### `GET /readyz`

Readiness endpoint. Verifies writable output storage and bundled model weights. A `503` means the service should not receive inference traffic.

## Synchronous inference

### `POST /v1/predict`

Acquires Sentinel-1 imagery from a remote provider and runs the complete SAR-LRA pipeline.

Request:

```json
{
  "provider": "auto",
  "roi": {
    "type": "Polygon",
    "coordinates": [[[85.30,27.65],[85.40,27.65],[85.40,27.75],[85.30,27.75],[85.30,27.65]]]
  },
  "event_date": "2025-08-15",
  "orbit": "ASCENDING",
  "request_id": "nepal-001",
  "project": null,
  "weights_path": null
}
```

Fields:

| Field | Required | Description |
|---|---:|---|
| `provider` | no | `auto`, `planetary-computer-grd` (alias `planetary-computer`), `planetary-computer-rtc`, or `earth-engine`; default `auto`. |
| `roi` | yes | GeoJSON geometry or Feature representing the area of interest. |
| `event_date` | yes | Event date in ISO `YYYY-MM-DD` form. |
| `orbit` | no | `ASCENDING` or `DESCENDING`; default `ASCENDING`. |
| `request_id` | no | Caller-supplied safe ID; otherwise generated. |
| `weights_path` | no | Optional weights under the configured input root. Bundled orbit weights are used when omitted. |
| `project` | only for EE when needed | Google Cloud/Earth Engine project. Ignored by Planetary Computer. |

Provider credentials come from the runtime environment, not this JSON body. See `ACQUISITION_PROVIDERS.md`.

### `POST /v1/predict-raster`

Runs inference on an already prepared four-band raster. This is the provider-independent/offline API.

```json
{
  "input_raster": "ascending-4band.tif",
  "orbit": "ASCENDING",
  "request_id": "offline-001",
  "weights_path": null,
  "roi": null,
  "probability_threshold": 0.5
}
```

`input_raster` and optional `weights_path` must resolve beneath `SAR_LRA_API_INPUT_ROOT` (default `/input`). The raster must satisfy SAR-LRA's validated four-band contract.

## Asynchronous jobs

Use async jobs for larger ROIs or deployments where HTTP connections should not remain open during acquisition/inference.

### `POST /v1/jobs`

Queues work in Redis and returns `202 Accepted`.

Remote provider example:

```json
{
  "source": "auto",
  "roi": {
    "type": "Polygon",
    "coordinates": [[[85.30,27.65],[85.40,27.65],[85.40,27.75],[85.30,27.75],[85.30,27.65]]]
  },
  "event_date": "2025-08-15",
  "orbit": "ASCENDING",
  "request_id": "job-nepal-001",
  "project": null
}
```

Prepared raster example:

```json
{
  "source": "prepared-raster",
  "input_raster": "ascending-4band.tif",
  "orbit": "ASCENDING",
  "request_id": "job-offline-001",
  "probability_threshold": 0.5
}
```

`source` accepts `prepared-raster`, `auto`, `planetary-computer-grd` (alias `planetary-computer`), `planetary-computer-rtc`, or `earth-engine`.

Typical acceptance response:

```json
{
  "job_id": "job-nepal-001",
  "state": "queued",
  "status_url": "/v1/jobs/job-nepal-001",
  "result_url": "/v1/jobs/job-nepal-001/result"
}
```

### `GET /v1/jobs/{job_id}`

Returns job metadata/state. States can include queued, acquiring, preprocessing, inferencing, postprocessing, completed, failed, and cancelled.

### `GET /v1/jobs/{job_id}/result`

Returns the completed pipeline result. Before completion the endpoint returns `409`. Failed jobs return the safe stored error record.

### `DELETE /v1/jobs/{job_id}`

Requests cancellation. Returns `202`. Output cleanup is performed for cancelled jobs.

### `POST /v1/jobs/{job_id}/retry`

Requeues a failed or cancelled job if the configured attempt limit has not been reached. Completed jobs are not duplicated.

## Concurrency and limits

- `SAR_LRA_API_MAX_CONCURRENT` bounds synchronous inference concurrency.
- `SAR_LRA_MAX_QUEUED_JOBS` bounds the async queue.
- `SAR_LRA_JOB_TIMEOUT_SECONDS` bounds worker execution time.
- `SAR_LRA_JOB_MAX_ATTEMPTS` bounds retries.
- `SAR_LRA_MIN_FREE_DISK_MB` prevents starting jobs when output disk is below the configured floor.
- ROI area/width/height/vertex limits come from YAML configuration and CLI/API validation.

## Error model

Validation errors are returned as structured JSON:

```json
{
  "error": {
    "code": "validation_error",
    "message": "..."
  }
}
```

Unexpected internal failures return a generic `internal_error`; arbitrary exception details and credentials are not reflected back to HTTP clients.
