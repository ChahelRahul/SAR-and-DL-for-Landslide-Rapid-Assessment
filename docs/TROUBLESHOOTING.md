# Troubleshooting

## Planetary Computer GRD fails during download

`planetary-computer-grd` does not require `PC_SDK_SUBSCRIPTION_KEY`, but it does require outbound access to the Planetary Computer STAC/SAS endpoints and Azure Blob Storage. Check DNS, HTTPS egress, and system time. Anonymous SAS tokens are short-lived and are requested internally.

## `Planetary Computer precomputed RTC access requires PC_SDK_SUBSCRIPTION_KEY`

You selected `planetary-computer-rtc`. Either provide `PC_SDK_SUBSCRIPTION_KEY` or switch to the default keyless `planetary-computer-grd` provider.

## GRD terrain correction is slow

This mode downloads complete SAFE products and runs local radiometric terrain correction. Reuse the output cache, keep the ROI small, provide fast local storage, and optionally mount a prebuilt DEM using `SAR_LRA_PC_DEM_PATH`.

## DEM acquisition fails

Mount a suitable DEM and set `SAR_LRA_PC_DEM_PATH=/input/dem.tif`. The DEM must cover the ROI. Using a fixed DEM is also recommended for scientific reproducibility.

## Earth Engine authentication fails

Use `provider: "earth-engine"` only after configuring Google/Earth Engine credentials. Containers do not perform interactive authentication automatically.

## API returns 429

The synchronous concurrency gate or asynchronous queue is full. Retry after the `Retry-After` interval or adjust the corresponding deployment limits.

## Input path rejected

Prepared rasters must resolve under `SAR_LRA_API_INPUT_ROOT` (default `/input`). This is an intentional path traversal control.
