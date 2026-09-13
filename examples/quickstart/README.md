# Notebook-free quickstart fixture

This directory contains a small **synthetic Sentinel-1-like** prepared-raster fixture for end-user deployment smoke testing. It is copied from the project regression fixtures and is not real Haiti satellite imagery or an accuracy benchmark.

Files:

- `ascending-4band.tif` — four-band prepared raster with the SAR-LRA band/order/orbit metadata required by the validator.
- `roi.geojson` — matching ROI.
- `reference-metadata.json` — regression metadata for maintainers; it is not required to run inference.

The CPU container copies this directory to `/opt/sar-lra/examples/quickstart`, so a user can pull the image and run a prediction without cloning the repository or reading a notebook. See `docs/end-user-deployment.md`.
