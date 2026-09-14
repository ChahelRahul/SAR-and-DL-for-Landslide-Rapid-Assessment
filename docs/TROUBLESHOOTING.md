# Troubleshooting

## `Earth Engine authentication/initialization failed`

The request selected Earth Engine without usable credentials. Configure a service account/ADC/Earth Engine credential, provide the appropriate project, or choose Planetary Computer/prepared-raster mode.

## `Planetary Computer ... requires PC_SDK_SUBSCRIPTION_KEY`

The request selected Planetary Computer but no key is available. Export the key into both API and worker containers for async jobs.

## `No remote acquisition credentials were detected`

`provider=auto` could not detect a supported credential. Set `SAR_LRA_ACQUISITION_PROVIDER` explicitly and provide its credential, or use `/v1/predict-raster`.

## `Bind for 0.0.0.0:9000 failed: port is already allocated`

Another host process/container owns MinIO's published port. Change `SAR_LRA_MINIO_API_PORT`/`SAR_LRA_MINIO_CONSOLE_PORT`. Do not change the internal `http://minio:9000` endpoint used by Compose services.

## `no matching manifest for linux/amd64`

Check the image/tag with `docker buildx imagetools inspect`. Use the runnable image/tag/digest that contains `Platform: linux/amd64`, not a standalone attestation index/tag.

## Redis `vm.overcommit_memory` warning

Redis can run but recommends enabling memory overcommit on the host. On Linux/WSL with appropriate privileges: `sudo sysctl -w vm.overcommit_memory=1`.

## Job remains queued

Verify the worker is running, can reach Redis, has the same provider credentials as the API, and has sufficient output disk.

## `429 busy` or `queue_full`

Synchronous concurrency or async queue limits have been reached. Retry later or raise the corresponding configured limit only if compute/memory capacity supports it.
