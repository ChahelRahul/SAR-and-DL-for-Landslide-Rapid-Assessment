# Issue 11 — GeoJSON and GeoPackage vector outputs

SAR-LRA operational vector outputs no longer use loose Shapefile sidecars.

## Default output

`GeoJSON` is the default vector format:

```text
detections.geojson
```

It is always written when `processing.vector_format: geojson`, including when
there are no detections. An empty result is represented as:

```json
{"type":"FeatureCollection","features":[]}
```

GeoJSON coordinates are written in **EPSG:4326**. This makes the default output
portable for APIs and web clients and avoids relying on the deprecated GeoJSON
`crs` member.

## GeoPackage

For larger outputs, select:

```yaml
processing:
  vector_format: geopackage
```

or request both formats:

```yaml
processing:
  vector_format: both
```

The CLI equivalents are:

```bash
--vector-format geopackage
--vector-format both
```

GeoPackage is a single file named `detections.gpkg` with a `detections` layer.
It preserves the source/intermediate raster CRS. Empty GeoPackages are valid and
contain an empty `detections` layer.

## ROI clipping

When an ROI was supplied to the pipeline, vector candidate areas are intersected
with that ROI before serialization. The exporter does not create candidate
geometry outside the submitted ROI. GeoJSON input ROIs are interpreted as
EPSG:4326 and transformed to the raster CRS for clipping.

## Shapefile compatibility

Shapefile is not part of the default API contract. It can be requested only as a
single ZIP archive:

```bash
--shapefile-zip
```

The resulting file is:

```text
detections-shapefile.zip
```

Loose `.shp`, `.dbf`, `.shx`, and `.prj` files are never returned as separate
pipeline artifacts. Because the Shapefile format has a 10-character field-name
limit, this compatibility export intentionally uses a reduced schema. GeoJSON or
GeoPackage should be used when full SAR-LRA metadata is required.

## Geometry semantics

The format change does not change the scientific meaning of the geometry.
Features remain thresholded candidate areas from the probability surface, not
validated or exact landslide boundaries.
