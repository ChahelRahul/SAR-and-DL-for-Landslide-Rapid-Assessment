# Issue 5: Acquisition and inference separation

SAR-LRA exposes two operational modes that converge on one validated four-band
GeoTIFF contract and one result schema.

## Mode A: Earth Engine acquisition and inference

```bash
sar-lra predict \
  --roi roi.geojson \
  --event-date 2024-04-03 \
  --source earth-engine \
  --weights model/weights/<orbit-weight>.hdf5 \
  --orbit ASCENDING \
  --output-dir results
```

Earth Engine is initialized only after this command is selected and only when a
matching intermediate raster is absent from the cache. Interactive
Earth Engine authentication occurs only with `--authenticate`.

Cache entries are content-addressed from the ROI geometry, event date, orbit,
imagery configuration and required band order. A cache hit is validated before
reuse.

## Mode B: Prepared raster inference

```bash
sar-lra predict-raster \
  --ascending ascending-4band.tif \
  --weights model/weights/<ascending-weight>.hdf5 \
  --output-dir results
```

Use `--descending` for the descending model. This mode does not import,
initialize or authenticate Earth Engine.

## Intermediate raster contract

The GeoTIFF must contain exactly four finite bands in this order:

1. `postVV`
2. `postVH`
3. `diffVV`
4. `diffVH`

It must have a CRS, valid dimensions and a geotransform. Earth Engine-generated
intermediates additionally carry source, cache key, orbit, relative orbit,
event date, temporal-window and scale metadata.

## Unified result

Both commands return `PipelineResult` with the same fields. The `mode` field is
`earth-engine` or `prepared-raster`; `input_metadata` records the validated
input and cache status. Both modes write the same detection mask, optional
vector detections and `result.json` metadata artifact.
