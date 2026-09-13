# Issue 15 — reproducible CPU Docker image

## Image contract

The repository root now contains a production CPU `Dockerfile`. It targets
`linux/amd64`, runs as UID/GID `10001`, exposes `/input` as the conventional
read-only input mount and `/output` as the writable result mount, and uses
`sar-lra` as its entrypoint.

The base is pinned to CPython 3.11.13 slim-bookworm by immutable manifest
digest. Runtime Python packages used by the image are exact-pinned in
`requirements/docker-cpu.txt`. The image installs the repository itself with
`--no-deps`, preventing `pyproject.toml` ranges from silently changing the
container dependency set.

The image intentionally uses `tensorflow-cpu==2.18.0` and sets
`CUDA_VISIBLE_DEVICES=-1`. It does not install CUDA, cuDNN, or an NVIDIA base
image.

## Build

```bash
docker build --platform linux/amd64 -t sar-lra:cpu .
```

To verify the embedded runtime:

```bash
docker run --rm \
  --entrypoint /opt/sar-lra/scripts/container_smoke_test.sh \
  sar-lra:cpu
```

## Prepared-raster inference

Mount inputs read-only and outputs read/write:

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/results:/output" \
  sar-lra:cpu predict-raster \
  --input /input/sentinel1-4band.tif \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

The released model weights are embedded in the image and verified during the
Docker build against `model/weights-manifest.json`.

## Earth Engine mode

Earth Engine support is installed, but credentials are **not** baked into the
image. Authentication material must be mounted or supplied by the deployment
platform. Never COPY local Earth Engine credentials into the image.

Example command shape:

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/results:/output" \
  -v "$HOME/.config/earthengine:/home/sarlra/.config/earthengine:ro" \
  sar-lra:cpu predict \
  --roi /input/roi.geojson \
  --event-date 2024-04-03 \
  --orbit ASCENDING \
  --output-dir /output
```

## Reproducibility boundary

The Docker base manifest, Python version, and direct Python dependencies are
pinned. `docker build` still downloads package artifacts from package indexes;
therefore byte-for-byte supply-chain reproduction additionally requires an
artifact mirror or a fully hash-locked wheelhouse. That stronger offline build
is outside Issue 15 and should be added before high-assurance deployment.

## CI

`.github/workflows/docker-cpu.yml` builds the image, runs the smoke test,
asserts UID `10001`, and validates a mounted reference ROI. These tests do not
require Earth Engine credentials.
