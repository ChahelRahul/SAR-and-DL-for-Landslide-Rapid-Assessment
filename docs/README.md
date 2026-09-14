# SAR-LRA documentation

This directory is the operational reference for SAR-LRA 2.0.0.

## Start here

- [API reference](API.md) — all HTTP routes, request bodies, job lifecycle, responses, and examples.
- [Acquisition providers and credentials](ACQUISITION_PROVIDERS.md) — Planetary Computer, Earth Engine, automatic provider selection, and credential-free prepared-raster mode.
- [CLI reference](CLI.md) — every supported command and the main options.
- [Configuration and environment](configuration.md) — YAML settings and environment variables.
- [Deployment](DEPLOYMENT.md) — Docker, Compose, API/worker, MinIO, Redis, CPU/GPU, and credential injection.
- [Outputs](OUTPUTS.md) — probability rasters, masks, vectors, metadata, and object-store results.
- [Architecture](ARCHITECTURE.md) — acquisition → preprocessing → inference → post-processing flow.
- [Security](SECURITY.md) — secret handling, filesystem boundaries, container posture, and network considerations.
- [Troubleshooting](TROUBLESHOOTING.md) — common authentication, Docker, provider, and inference errors.

FastAPI also exposes live interactive documentation at `/docs`, ReDoc at `/redoc`, and the generated OpenAPI document at `/openapi.json` when the API service is running.
