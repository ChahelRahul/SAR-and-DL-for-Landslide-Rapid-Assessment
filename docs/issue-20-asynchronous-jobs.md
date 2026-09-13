# Issue 20 — Asynchronous job execution

SAR-LRA now supports a Redis-backed asynchronous API path so Earth Engine acquisition and TensorFlow inference do not hold an HTTP request open.

## Architecture

```text
FastAPI
  │ POST /v1/jobs
  ▼
Redis list + expiring job record
  │
  ▼
sar-lra-worker
  │
  ▼
existing run_raster()/run_earth_engine() pipeline
  │
  ▼
shared filesystem result store
```

The initial result store is the shared filesystem mounted at `SAR_LRA_API_OUTPUT_ROOT`. API and worker must mount the same location. S3-compatible storage is not implemented in Issue 20.

## API

Submit a prepared-raster job:

```http
POST /v1/jobs
Content-Type: application/json

{
  "source": "prepared-raster",
  "input_raster": "ascending-4band.tif",
  "orbit": "ASCENDING",
  "request_id": "event-001"
}
```

The API validates paths and returns immediately:

```json
{
  "job_id": "event-001",
  "state": "queued",
  "status_url": "/v1/jobs/event-001",
  "result_url": "/v1/jobs/event-001/result"
}
```

Other operations:

```text
GET    /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/result
DELETE /v1/jobs/{job_id}
POST   /v1/jobs/{job_id}/retry
```

The synchronous Issue 19 endpoints are retained for compatibility, but `/v1/jobs` is the recommended deployment interface for long-running work.

## States

Job states are:

```text
queued
acquiring
preprocessing
inferencing
postprocessing
completed
failed
cancelled
```

Prepared-raster jobs normally skip `acquiring`. Earth Engine jobs enter `acquiring` before export/download, then `preprocessing` before inference.

## Idempotency

Redis queue delivery is intentionally treated as at-least-once. Before executing a job, the worker reloads its record. A job already in `completed` or `cancelled` is not executed again. `POST /retry` also refuses to requeue completed work. The output directory is stable (`<output-root>/<job_id>`), so a retry of a failed job reuses the same job identity rather than producing duplicate completed outputs.

## Safe failures

Job records never store arbitrary exception text for unexpected failures. Public failure summaries are limited to stable codes/messages such as:

```json
{"code": "processing_failed", "message": "job processing failed"}
```

Detailed stack traces belong in worker logs, not API responses.

## Retention

`SAR_LRA_JOB_RETENTION_SECONDS` controls Redis job-record TTL and defaults to 86400 seconds (24 hours). Every state update refreshes that TTL, so retention is measured from the latest job activity/completion. Filesystem result deletion is intentionally left to deployment lifecycle policy / Issue 21 cleanup controls.

## Deployment

A reference deployment is included:

```bash
docker compose up --build
```

It starts Redis, one API process, and one SAR-LRA worker sharing the result volume.

## Cancellation boundary

Issue 20 supports cancellation state and prevents queued/cancelled jobs from starting. Cooperative interruption and cleanup of a pipeline already executing are part of Issue 21; if a running job is marked cancelled in Issue 20, the worker will not replace that terminal state with `completed`.
