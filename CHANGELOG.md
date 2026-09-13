# Changelog

All notable changes to SAR-LRA are documented here.

## [2.0.0] - 2026-09-13

First production-oriented container/API release of the refactored SAR-LRA toolchain.

### Added

- Standard importable `app/` Python package and `sar-lra` CLI.
- Validated YAML configuration and ROI processing limits.
- Separate Earth Engine acquisition and prepared-raster inference modes.
- Formal four-band Sentinel-1 raster validation and edge-safe sliding windows.
- Probability-surface outputs, re-thresholding, GeoJSON/GeoPackage vector outputs, and explicit geometry semantics.
- Reference-event regression fixtures and CI coverage.
- CPU and optional NVIDIA GPU container images.
- Secure Earth Engine credential discovery for container deployments.
- Streamed/batched inference with bounded patch-memory usage.
- FastAPI service plus Redis-backed asynchronous jobs.
- Timeout, cancellation, retry, queue, disk-space, and cleanup safeguards.
- Optional MinIO/S3 result mirroring and complete Docker Compose deployment.
- GHCR publication, SBOM generation, vulnerability scanning, provenance, and Cosign signing workflow.
- Notebook-free end-user quickstart fixture and deployment guide.

### Changed

- Package version promoted from `2.0.0b1` to `2.0.0`.
- GeoJSON is the default vector interchange format; Shapefile is compatibility-only as a ZIP bundle.
- Operational geometry is derived from the thresholded probability surface rather than NMS boxes.
- Historical focal-loss semantics are preserved explicitly for released-model compatibility while inference loads weights without compilation.

### Scientific compatibility

- Released ascending and descending model weight files are unchanged from the documented v2 beta model assets.
- Model identifier remains `sar-lra-v2.0.0-beta.1`; software release `2.0.0` does not imply retrained weights.
- Probability rasters are aggregated model-window scores and are not calibrated per-pixel landslide probabilities.

### Known release blocker

The code repository is MIT-licensed, but `model/weights-manifest.json` still records the trained-weight distribution licence as `UNCONFIRMED`. A public release containing or redistributing those weights must not represent them as MIT-licensed until the maintainer/rightsholder explicitly confirms the applicable licence.
