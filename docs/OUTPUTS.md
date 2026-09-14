# Outputs

SAR-LRA preserves probabilities before thresholding and can produce multiple geospatial artifacts depending on configuration.

Typical artifacts include:

- probability GeoTIFF — continuous model probability values
- binary mask GeoTIFF — thresholded detection mask when enabled
- vector detections — GeoJSON/GPKG and optional Shapefile ZIP according to configuration
- result metadata — request ID, orbit, model/version/checksum, effective settings, input and processing metadata

Remote acquisition also creates/caches a validated four-band intermediate raster. Planetary Computer cache metadata records collection, orbit, relative orbit, event date, scene counts, scale, and conversion/compositing information.

Async deployments can mirror completed job outputs to the configured S3/MinIO bucket and include object-store information in the job result.

The four model input bands are always expected in this semantic order:

1. `postVV`
2. `postVH`
3. `diffVV`
4. `diffVH`
