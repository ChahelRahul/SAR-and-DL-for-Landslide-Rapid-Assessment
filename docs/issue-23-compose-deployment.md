# Issue 23 — Complete Docker Compose deployment

The reference deployment now includes the HTTP API, one inference worker, Redis,
MinIO object storage, persistent volumes, health checks, and an opt-in CLI-only
service profile.

## Full async stack

```bash
cp .env.example .env
# Replace MINIO_ROOT_PASSWORD before any shared deployment.
docker compose up --build -d
```

Services:

- `api`: FastAPI on port 8000 by default.
- `worker`: one resource-bounded SAR-LRA worker.
- `redis`: durable queue/job metadata with AOF enabled.
- `minio`: S3-compatible artifact mirror; API 9000 and console 9001 by default.
- `minio-init`: creates the private result bucket once MinIO is healthy.

Completed jobs keep their local `/output/<job-id>` files and, when object storage
is configured, mirror all files to `s3://<bucket>/jobs/<job-id>/...`. The result
JSON contains an `object_store` manifest with object keys, URIs, and sizes. MinIO
credentials are never returned in job metadata.

The filesystem remains the worker's active scratch/result area because GDAL and
TensorFlow operate on local paths. MinIO is the durable object-copy layer in this
reference architecture; it is not mounted as a POSIX filesystem.

## CLI-only profile

The CLI service has no Redis/MinIO dependency:

```bash
docker compose --profile cli run --rm cli \
  predict-raster \
  --input /input/sentinel1-4band.tif \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/<ascending-weight>.hdf5 \
  --output-dir /output
```

This mounts `${SAR_LRA_INPUT_DIR:-./input}` read-only at `/input` and
`${SAR_LRA_CLI_OUTPUT_DIR:-./results}` at `/output`.

## Persistence

Named volumes:

- `redis-data` — Redis AOF data.
- `minio-data` — S3 object data.
- `sar-lra-results` — API/worker shared local output.

Back up MinIO and any Redis metadata required by your retention policy. The job
TTL from Issue 20 still controls Redis job-record expiry; it does not delete
MinIO objects automatically.

## Security

The `.env.example` password is a placeholder. Change it before exposing MinIO or
the API beyond localhost. The result bucket is initialized as private. Do not
commit `.env`, service-account JSON, Google credentials, AWS keys, or MinIO
secrets. For cloud deployment, prefer secret managers/workload identities over
literal environment secrets.

## Object-store configuration

The worker recognizes:

- `SAR_LRA_S3_ENDPOINT_URL` — MinIO or another S3-compatible endpoint.
- `SAR_LRA_S3_BUCKET` — enables mirroring when non-empty.
- `SAR_LRA_S3_PREFIX` — defaults to `jobs`.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` — standard
  boto3 credentials/settings.

If `SAR_LRA_S3_BUCKET` is unset, object mirroring is disabled and the worker
behaves exactly as in Issue 21.
