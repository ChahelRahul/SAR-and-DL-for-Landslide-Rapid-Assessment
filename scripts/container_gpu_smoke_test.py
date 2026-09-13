"""Runtime smoke test that must execute on an NVIDIA-enabled container host."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.config import ModelLoadConfig
from app.inference.model import load_model

ROOT = Path("/opt/sar-lra")
TOLERANCE = 2e-5


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit(
            "No TensorFlow GPU is visible. Start the container with `docker run --gpus all` "
            "on a host with an NVIDIA driver and NVIDIA Container Toolkit."
        )

    tf.config.set_soft_device_placement(False)
    manifest = json.loads((ROOT / "model/weights-manifest.json").read_text())
    ascending = next(item for item in manifest["weights"] if item["orbit"] == "ASCENDING")
    weights = ROOT / "model/weights" / ascending["filename"]
    config = ModelLoadConfig(
        orbit="ASCENDING",
        weights_path=weights,
        filters_first_layer=int(ascending["filters_first_layer"]),
        patch_size=int(ascending["input_shape"][0]),
    )
    model, _ = load_model(config)

    # Fixed input makes CPU/GPU comparison deterministic enough to detect a
    # broken accelerator path without claiming bit-for-bit kernel identity.
    data = np.linspace(-25.0, 5.0, 64 * 64 * 4, dtype=np.float32).reshape(1, 64, 64, 4)

    with tf.device("/CPU:0"):
        cpu = model(tf.convert_to_tensor(data), training=False).numpy()
    with tf.device("/GPU:0"):
        gpu = model(tf.convert_to_tensor(data), training=False)
        gpu_result = gpu.numpy()

    device = gpu.device.upper()
    if "GPU" not in device:
        raise AssertionError(f"Inference tensor was not placed on a GPU: {gpu.device}")
    if not np.isfinite(gpu_result).all():
        raise AssertionError("GPU inference produced non-finite output")
    if not ((gpu_result >= 0.0) & (gpu_result <= 1.0)).all():
        raise AssertionError(f"Sigmoid output outside [0, 1]: {gpu_result!r}")
    np.testing.assert_allclose(cpu, gpu_result, rtol=TOLERANCE, atol=TOLERANCE)

    print(f"tensorflow={tf.__version__}")
    print(f"gpu={gpus[0].name}")
    print(f"inference_device={gpu.device}")
    print(f"cpu_gpu_tolerance={TOLERANCE}")
    print(f"prediction={float(gpu_result.ravel()[0]):.8f}")


if __name__ == "__main__":
    main()
