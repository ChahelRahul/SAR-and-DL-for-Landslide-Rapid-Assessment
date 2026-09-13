#!/bin/sh
set -eu

sar-lra --help >/dev/null
python /opt/sar-lra/scripts/verify_model_weights.py
python - <<'PY'
import tensorflow as tf
assert tf.__version__ == "2.18.0", tf.__version__
assert tf.config.list_physical_devices("GPU") == [], "GPU visible in CPU image"
print("tensorflow_cpu=2.18.0")
PY

# Lightweight geospatial and CLI import checks.
python - <<'PY'
import rasterio, geopandas, shapely, pyproj
import app
print("sar_lra_import=ok")
print("rasterio=", rasterio.__version__)
PY
