# syntax=docker/dockerfile:1.7
# Reproducible CPU runtime for SAR-LRA.
# Pinned to the Python 3.11.13 slim-bookworm multi-platform manifest digest.
ARG PYTHON_IMAGE=python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1
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
RUN python -m pip install --no-deps . \
    && python scripts/verify_model_weights.py \
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
