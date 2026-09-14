#!/bin/sh
set -eu

APP_ROOT="${SAR_LRA_APP_ROOT:-/opt/sar-lra}"

sar-lra --help >/dev/null
python "${APP_ROOT}/scripts/verify_model_weights.py" \
  --manifest "${APP_ROOT}/model/weights-manifest.json" \
  --directory "${APP_ROOT}/model/weights"
python - <<'PY'
import tensorflow as tf
assert tf.__version__ == "2.21.0", tf.__version__
assert tf.config.list_physical_devices("GPU") == [], "GPU visible in CPU image"
print("tensorflow_cpu=2.21.0")
PY

# Lightweight geospatial and CLI import checks.
python - <<'PY'
import rasterio, geopandas, shapely, pyproj
import app
print("sar_lra_import=ok")
print("rasterio=", rasterio.__version__)
PY
