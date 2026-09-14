# Microsoft Planetary Computer provider

SAR-LRA exposes two Planetary Computer modes.

## Keyless GRD mode (recommended)

```bash
sar-lra predict \
  --provider planetary-computer-grd \
  --roi roi.geojson \
  --event-date 2025-08-15 \
  --orbit ASCENDING \
  --output-dir results
```

No `PC_SDK_SUBSCRIPTION_KEY` is required. SAR-LRA searches the public `sentinel-1-grd` STAC collection, obtains an anonymous read-only SAS token for the Azure GRD container, downloads required SAFE products, creates/uses a DEM, and performs local terrain correction with `sarsen`.

`--provider planetary-computer` is an alias for this mode.

## Precomputed RTC mode

```bash
export PC_SDK_SUBSCRIPTION_KEY='...'
sar-lra predict --provider planetary-computer-rtc --roi roi.geojson --event-date 2025-08-15 --orbit ASCENDING --output-dir results
```

This uses `sentinel-1-rtc`. It is faster operationally but the collection requires a Planetary Computer account key.

## DEM override

The keyless GRD path normally builds a DEM from `cop-dem-glo-30`. To use a controlled DEM:

```bash
export SAR_LRA_PC_DEM_PATH=/input/dem.tif
```

This is useful for reproducible scientific validation.

## Processing contract

Both modes produce `postVV`, `postVH`, `diffVV`, `diffVH` in dB. They are not assumed to be numerically identical to Earth Engine preprocessing; validate against reference events before changing a production provider.
