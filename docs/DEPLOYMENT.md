# Deployment guide

## CPU container

The standard image is Linux/amd64 and runs as a non-root user. For a published image:

```bash
docker pull ghcr.io/chahelrahul/sar-lra:latest
```

For immutable production deployments, prefer a verified runnable digest rather than `latest`.

## Compose stack

The reference stack contains:

- `api` — FastAPI HTTP service
- `worker` — asynchronous job executor
- `redis` — queue/job metadata
- `minio` — S3-compatible result object store
- `minio-init` — creates a private result bucket
- `cli` — optional profile for command-line-only operation

```bash
cp .env.example .env
docker compose up -d
```

### Planetary Computer deployment

Set in `.env`:

```dotenv
SAR_LRA_ACQUISITION_PROVIDER=planetary-computer
PC_SDK_SUBSCRIPTION_KEY=your-key
```

### Earth Engine deployment

Do not put the service-account JSON inside the image. Mount it read-only and set the path inside the container. A Compose override can be used:

```yaml
services:
  api:
    environment:
      SAR_LRA_ACQUISITION_PROVIDER: earth-engine
      GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/gee.json
    volumes:
      - ./gee.json:/run/secrets/gee.json:ro
  worker:
    environment:
      SAR_LRA_ACQUISITION_PROVIDER: earth-engine
      GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/gee.json
    volumes:
      - ./gee.json:/run/secrets/gee.json:ro
```

The worker needs the same provider credentials as the API for asynchronous remote-acquisition jobs.

### Offline/prepared-raster deployment

No provider credential is needed. Mount the prepared raster beneath `/input` and call `/v1/predict-raster`, or run the CLI `predict-raster` command.

## Ports

Default reference ports are API `8000`, MinIO API `9000`, and MinIO console `9001`. Host ports are configurable in `.env`. Internal service URLs should continue to use Compose service names and container ports (for example `http://minio:9000`).

## GPU

`Dockerfile.gpu` builds a separate NVIDIA image. The host must supply a compatible driver and NVIDIA Container Toolkit. CPU remains the default supported deployment.
