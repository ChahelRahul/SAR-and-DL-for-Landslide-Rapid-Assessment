# Configuration and environment

## YAML

The default configuration is `config/default.yaml`. Main groups are model, imagery, and processing settings. Runtime overrides are available from the CLI for frequently changed parameters.

Important imagery settings include the pre-event window, post-event window, scale, and Sentinel-1 orbit. Important processing settings include tile/window size, overlap, probability threshold/aggregation, batching, worker count, vector output, and output chunking.

## Environment variables

### Provider selection and credentials

| Variable | Purpose |
|---|---|
| `SAR_LRA_ACQUISITION_PROVIDER` | `auto`, `planetary-computer-grd` (alias `planetary-computer`), `planetary-computer-rtc`, or `earth-engine`. |
| `PC_SDK_SUBSCRIPTION_KEY` | Optional; required only for `planetary-computer-rtc`, not for keyless GRD. |
| `SAR_LRA_PC_DEM_PATH` | Optional local DEM override for the keyless GRD path. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a Google service-account/ADC JSON visible inside the process/container. |

### API/filesystem

| Variable | Default | Purpose |
|---|---|---|
| `SAR_LRA_API_INPUT_ROOT` | `/input` | Allowed input filesystem root. |
| `SAR_LRA_API_OUTPUT_ROOT` | `/output` | Output/cache root. |
| `SAR_LRA_API_CONFIG` | unset | Optional YAML config path. |
| `SAR_LRA_API_MAX_CONCURRENT` | `1` | Synchronous inference slots. |
| `SAR_LRA_MODEL_DIR` | bundled model directory | Override model-weight directory. |

### Redis/jobs

| Variable | Default |
|---|---:|
| `SAR_LRA_REDIS_URL` | `redis://redis:6379/0` |
| `SAR_LRA_JOB_RETENTION_SECONDS` | `86400` |
| `SAR_LRA_JOB_TIMEOUT_SECONDS` | `3600` |
| `SAR_LRA_JOB_MAX_ATTEMPTS` | `3` |
| `SAR_LRA_MAX_QUEUED_JOBS` | `100` |
| `SAR_LRA_MIN_FREE_DISK_MB` | `1024` |

### S3/MinIO result mirroring

`SAR_LRA_S3_ENDPOINT_URL`, `SAR_LRA_S3_BUCKET`, `SAR_LRA_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` configure optional S3-compatible result mirroring used by async workers.

Never bake provider or object-store secrets into an image or repository.
