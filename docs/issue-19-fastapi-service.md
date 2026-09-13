# Issue 19 — FastAPI inference service

SAR-LRA exposes an optional HTTP service that wraps the same validated pipeline used by the CLI. It does not implement a second inference path.

## Install and run

```bash
pip install '.[geo,inference,api]'
sar-lra-api --host 0.0.0.0 --port 8000 --workers 1
```

Container deployments can use the existing CPU or GPU image and override the entrypoint:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  --entrypoint sar-lra-api sar-lra:cpu
```

## Endpoints

- `GET /healthz` — process liveness; does not load TensorFlow or Earth Engine.
- `GET /readyz` — checks output writability and released-weight availability.
- `POST /v1/predict-raster` — prepared four-band raster inference.
- `POST /v1/predict` — Earth Engine acquisition plus inference using ambient/mounted credentials.
- `/docs` — generated OpenAPI/Swagger interface.

Example prepared-raster body:

```json
{
  "input_raster": "ascending-4band.tif",
  "orbit": "ASCENDING",
  "request_id": "event-001",
  "probability_threshold": 0.6
}
```

`input_raster` is resolved under `SAR_LRA_API_INPUT_ROOT` (`/input` by default). Outputs are always written under `SAR_LRA_API_OUTPUT_ROOT` (`/output` by default). Arbitrary host paths and path-traversal request IDs are rejected.

## Concurrency

TensorFlow inference is resource-intensive. `SAR_LRA_API_MAX_CONCURRENT` defaults to `1`. Requests above this in-process limit receive HTTP `429` with `Retry-After: 5`; they are not queued indefinitely. For production, keep Uvicorn worker count aligned with available memory/GPU capacity. Each worker has its own concurrency gate and TensorFlow process state.

## Configuration

- `SAR_LRA_API_INPUT_ROOT=/input`
- `SAR_LRA_API_OUTPUT_ROOT=/output`
- `SAR_LRA_API_CONFIG=/path/to/config.yaml`
- `SAR_LRA_API_MAX_CONCURRENT=1`
- `SAR_LRA_API_HOST=0.0.0.0`
- `SAR_LRA_API_PORT=8000`
- `SAR_LRA_API_WORKERS=1`
- `SAR_LRA_MODEL_DIR=/opt/sar-lra/model/weights` (optional override)

Earth Engine authentication follows Issue 16: mounted service-account/ADC credentials or ambient workload identity; the HTTP service never invokes interactive authentication.

## Error contract

API-originated validation and capacity errors use a stable envelope:

```json
{"error": {"code": "validation_error", "message": "..."}}
```

Pydantic body validation uses `request_validation_error`; saturation uses `busy`. Pipeline output remains the standard `PipelineResult` JSON contract used by the CLI metadata.

## Operational scope

This is a synchronous inference API. It deliberately does not claim to be a durable job queue. Long-running, multi-tenant deployments should put a queue/orchestrator in front of the service rather than increasing worker/concurrency counts without memory limits.
