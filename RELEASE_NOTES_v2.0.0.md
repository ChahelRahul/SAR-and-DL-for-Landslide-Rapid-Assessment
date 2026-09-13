# SAR-LRA v2.0.0

SAR-LRA v2.0.0 is the first production-oriented release of the refactored Sentinel-1 landslide rapid-assessment toolchain.

## Highlights

- Pull-and-run CPU container and optional NVIDIA GPU image.
- CLI, synchronous FastAPI service, and Redis-backed asynchronous worker deployment.
- Earth Engine acquisition or fully offline prepared-raster inference.
- Validated four-band Sentinel-1 contract, ROI limits, streamed inference, deterministic regression fixtures, and explicit output metadata.
- Probability GeoTIFFs plus derived binary masks and GeoJSON/GeoPackage candidate-area outputs.
- Docker Compose reference deployment with Redis and optional MinIO/S3 result mirroring.
- GHCR supply-chain controls: provenance, SPDX SBOM, Trivy scanning, and keyless Cosign signatures.

## Install / run

```bash
docker pull ghcr.io/chahelrahul/sar-lra:2.0.0
```

Notebook-free smoke test:

```bash
mkdir -p results

docker run --rm \
  --mount type=bind,src="$PWD/results",dst=/output \
  ghcr.io/chahelrahul/sar-lra:2.0.0 \
  predict-raster \
  --input /opt/sar-lra/examples/quickstart/ascending-4band.tif \
  --roi /opt/sar-lra/examples/quickstart/roi.geojson \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

## Output semantics

`probability.tif` contains the maximum overlapping model-window score at each valid pixel. It is not a calibrated per-pixel landslide probability. Vector polygons are thresholded candidate areas for expert review; they are not exact landslide boundaries.

## Model compatibility

This software release does not introduce retrained weights. The model assets retain the identifier `sar-lra-v2.0.0-beta.1` and the documented SHA-256 checksums.

## Release prerequisite: trained-weight licence

Do not publish a release that claims the trained weights are MIT-licensed while `model/weights-manifest.json` remains `UNCONFIRMED`. Before tagging `v2.0.0`, the maintainer/rightsholder must explicitly confirm the distribution licence for both released `.hdf5` files and update the manifest/model card accordingly.

## Documentation

Start with `docs/end-user-deployment.md`. Scientific limitations and provenance are documented in `MODEL_CARD.md`.
