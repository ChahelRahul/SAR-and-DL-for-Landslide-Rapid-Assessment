# Issue 14 — ROI validation and processing limits

Issue 14 adds request preflight validation so malformed or accidentally global jobs are rejected before Sentinel-1 acquisition or model inference.

## Validation contract

Submitted ROIs must be GeoJSON `Polygon` or `MultiPolygon` geometries (directly or inside a GeoJSON `Feature`). Validation checks:

- valid JSON / GeoJSON object structure;
- Polygon or MultiPolygon geometry only;
- longitude in `[-180, 180]` and latitude in `[-90, 90]`;
- non-empty, non-zero-area geometry;
- valid topology according to Shapely;
- configured vertex-count limit;
- WGS84 geodesic area limit;
- geodesic bounding-box width and height limits;
- antimeridian detection.

Invalid self-intersections are rejected rather than silently repaired. Silent repair can change a disaster-response ROI and therefore requires an explicit future policy rather than an implicit geometry mutation.

ROIs crossing the antimeridian are detected using the shortest longitude arc, so they are not accidentally interpreted as near-global requests. The current operational policy is to reject them with a clear error until export, clipping, and output serialization are all explicitly dateline-safe.

## Default limits

```yaml
processing:
  max_roi_km2: 10000
  max_roi_width_km: 500
  max_roi_height_km: 500
  max_roi_vertices: 50000
```

Area is calculated geodesically on WGS84. Bounding-box dimensions are geodesic distances, not degree spans.

## Environment overrides

Public deployments can lower or raise limits without rebuilding the image:

```text
SAR_LRA_MAX_ROI_KM2
SAR_LRA_MAX_ROI_WIDTH_KM
SAR_LRA_MAX_ROI_HEIGHT_KM
SAR_LRA_MAX_ROI_VERTICES
```

Environment values override YAML/default configuration. Invalid environment values fail configuration loading.

CLI overrides remain available for the same limits.

## Event dates and Sentinel-1 availability

Earth Engine acquisition rejects:

- dates before `2014-10-03`, the beginning of the `COPERNICUS/S1_GRD` Earth Engine collection;
- future event dates.

This is only a temporal preflight. It does not claim that every ROI has usable VV/VH IW imagery for every valid date. During acquisition, the Earth Engine collection is filtered by ROI, date window and orbit; a request fails clearly if no relative orbit is available.

## Area reporting

Before acquisition, the CLI writes a `roi_validated` diagnostic containing:

- area in km²;
- bounding-box width and height in km;
- vertex count;
- geometry type;
- antimeridian status.

`validate-roi` also returns these values in its JSON result.

## Clipping

Earth Engine composites are clipped to the submitted ROI. Prepared-raster inference combines the raster valid-data mask with the rasterized ROI mask. Vector outputs are clipped to the ROI before serialization. No detection is intentionally produced outside the submitted processing geometry.

## API enforcement

`run_earth_engine()` and `run_raster()` repeat ROI validation so callers cannot bypass public-processing limits by skipping the CLI. `run_earth_engine()` also validates the event date before importing/initializing Earth Engine.

The validated ROI report is retained under:

```text
processing_metadata.roi_validation
```
