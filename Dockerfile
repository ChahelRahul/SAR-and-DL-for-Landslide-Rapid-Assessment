# syntax=docker/dockerfile:1.7
# Reproducible CPU runtime for SAR-LRA.
# Pinned to the Python 3.13.14 slim-bookworm multi-platform manifest digest.
ARG PYTHON_IMAGE=python:3.13.14-slim-bookworm@sha256:dd86541a59b252667f4c12f8b2ee17216de37dd65ac773bf097bef996fa78860
FROM --platform=linux/amd64 ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="SAR-LRA CPU"
LABEL org.opencontainers.image.description="CPU-only Sentinel-1 SAR landslide rapid-assessment CLI"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SAR_LRA_CONTAINER=1 \
    HOME=/home/sarlra

WORKDIR /opt/sar-lra

RUN apt-get update \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install the large, pinned runtime dependency set before copying application
# sources so normal code changes do not invalidate the dependency layer.
COPY requirements/docker-cpu.txt /tmp/docker-cpu.txt
RUN python -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" "wheel==0.46.2" \
    && python -m pip install --only-binary=:all: -r /tmp/docker-cpu.txt

COPY pyproject.toml README.md LICENSE MODEL_CARD.md ./
COPY app ./app
COPY config ./config
COPY model ./model
COPY schemas ./schemas
COPY scripts ./scripts
RUN chmod 0755 /opt/sar-lra/scripts/*.sh
COPY docs ./docs
COPY examples ./examples

# The project itself is installed without dependency resolution: container
# dependencies are controlled only by requirements/docker-cpu.txt.
RUN python -m pip install --no-deps .

# Normalize security-sensitive packaging helpers after every dependency/project
# install so the final filesystem cannot retain vulnerable metadata versions.
RUN python -m pip install --upgrade --force-reinstall "wheel==0.46.2" "jaraco.context==6.1.0" \
    && find /usr/local/lib/python3.13/site-packages -maxdepth 1 -type d \
        \( -name 'wheel-0.45.1.dist-info' -o -name 'jaraco.context-5.3.0.dist-info' \) \
        -exec rm -rf {} + \
    && python - <<'PYSEC'
from importlib.metadata import version
from pathlib import Path
site = Path("/usr/local/lib/python3.13/site-packages")
assert version("wheel") == "0.46.2", version("wheel")
assert version("jaraco.context") == "6.1.0", version("jaraco.context")
assert not list(site.glob("wheel-0.45.1.dist-info")), "stale vulnerable wheel metadata remains"
assert not list(site.glob("jaraco.context-5.3.0.dist-info")), "stale vulnerable jaraco.context metadata remains"
print("security_pins=ok; stale_metadata=absent")
PYSEC

RUN python scripts/verify_model_weights.py \
    && sar-lra --help >/dev/null \
    && sar-lra-api --help >/dev/null \
    && python - <<'PY'
import tensorflow as tf
assert tf.config.list_physical_devices("GPU") == [], "CPU image exposed a GPU"
print("TensorFlow", tf.__version__, "CPU runtime verified")
PY

RUN groupadd --system --gid 10001 sarlra \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/sarlra sarlra \
    && mkdir -p /input /output \
    && chown -R 10001:10001 /home/sarlra /output \
    && chmod 0555 /input

USER 10001:10001
WORKDIR /work

VOLUME ["/input", "/output"]
ENTRYPOINT ["sar-lra"]
CMD ["--help"]
