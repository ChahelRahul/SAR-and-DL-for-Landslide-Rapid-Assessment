# Issue 17 — Optional NVIDIA GPU image

SAR-LRA keeps the CPU image (`Dockerfile`, tag `sar-lra:cpu`) as the default deployment artifact. Issue 17 adds a separate optional NVIDIA GPU variant so GPU requirements do not leak into CPU deployments.

## Build

```bash
docker build --platform linux/amd64 -f Dockerfile.gpu -t sar-lra:gpu .
```

The image uses CPython 3.11.13 on the same digest-pinned slim-bookworm base as the CPU image and installs `tensorflow[and-cuda]==2.18.0`. TensorFlow's Linux pip GPU installation supplies the compatible CUDA/cuDNN user-space packages. The host must still provide a compatible NVIDIA driver and NVIDIA Container Toolkit.

A physical GPU is not required to **build** the image. GPU availability is a runtime property.

## Runtime prerequisites

- Linux x86-64 host.
- Supported NVIDIA GPU.
- NVIDIA driver compatible with the CUDA runtime required by TensorFlow 2.18.
- Docker configured with NVIDIA Container Toolkit.
- `docker run --gpus all ...` (or an equivalent explicit GPU device request).

The image does not contain a host NVIDIA kernel driver.

## Verify the accelerator

```bash
docker run --rm --gpus all \
  --entrypoint python \
  sar-lra:gpu \
  /opt/sar-lra/scripts/container_gpu_smoke_test.py
```

The smoke test fails if TensorFlow sees no GPU. It loads the released ascending model weights, performs inference on both CPU and GPU, verifies that the GPU tensor was actually placed on `/GPU:0`, and compares outputs using the Issue 12 GPU tolerance (`rtol=2e-5`, `atol=2e-5`).

## Run SAR-LRA

```bash
docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/results:/output" \
  sar-lra:gpu predict-raster \
  --input /input/sentinel1-4band.tif \
  --orbit ASCENDING \
  --weights /opt/sar-lra/model/weights/VV_VH_60_nn_noSlope_ASCENDING_60_12_6_size_64_filters_32_batch_size_512_lr_0.001_dropout_0.7_fil1_3_fil2_3_fil3_3.hdf5 \
  --output-dir /output
```

Earth Engine authentication follows the same mounted-secret/ADC policy documented in Issue 16.

## CPU fallback policy

The GPU image is an optional accelerator image, not the default fallback mechanism. If an NVIDIA runtime is unavailable, deploy `sar-lra:cpu`; do not silently treat a missing GPU as a successful GPU deployment. This makes scheduler/configuration errors observable.

## CI

`.github/workflows/docker-gpu.yml` has two levels:

1. Standard GitHub-hosted runners build the GPU-capable image and validate TensorFlow/NVIDIA user-space packages without claiming hardware acceleration.
2. A real GPU inference job runs only when repository variable `SAR_LRA_GPU_CI=true` and a self-hosted runner carrying labels `linux`, `x64`, and `gpu` is available.

This distinction prevents a no-GPU builder from being reported as a successful GPU runtime test.
