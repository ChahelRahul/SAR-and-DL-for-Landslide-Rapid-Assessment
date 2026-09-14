#!/bin/sh
set -eu

sar-lra --help >/dev/null
python /opt/sar-lra/scripts/verify_model_weights.py
python - <<'PY'
import importlib.util
import tensorflow as tf

assert tf.__version__ == "2.21.0", tf.__version__
# These modules are installed by TensorFlow's `and-cuda` extra. This check is
# suitable for builders with no physical GPU.
required = ["nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_runtime"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
assert not missing, f"Missing NVIDIA user-space packages: {missing}"
print("tensorflow_gpu_runtime=2.21.0")
print("physical_gpus_visible_at_build_or_static_test=", len(tf.config.list_physical_devices("GPU")))
PY
