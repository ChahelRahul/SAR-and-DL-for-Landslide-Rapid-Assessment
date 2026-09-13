#!/usr/bin/env python3
"""Representative bounded-memory inference benchmark for Issue 18.

This intentionally uses a deterministic NumPy model so it can run without
TensorFlow. It measures Python-process max RSS deltas around inference and also
reports the exact largest patch tensor materialized by the streaming engine.
"""
from __future__ import annotations

import json
import resource
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.config import ModelConfig, ProcessingConfig
from app.inference.predict import predict_stack


class MeanModel:
    def predict(self, patches, batch_size=None, verbose=0):
        x = np.asarray(patches, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-x.mean(axis=(1, 2, 3)) / 10.0))).reshape(-1, 1)


def rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. CI/container target is Linux.
    return float(value) / 1024.0


def run(size: int, batch_size: int) -> dict[str, float | int]:
    # Allocate input before recording RSS so the report isolates inference overhead.
    stack = np.zeros((size, size, 4), dtype=np.float32)
    stack[..., 0] = -18.0
    stack[..., 1] = -24.0
    before = rss_mib()
    with tempfile.TemporaryDirectory() as tmp:
        pred = predict_stack(
            MeanModel(), stack, ModelConfig(batch_size=batch_size),
            ProcessingConfig(inference_workers=1),
            surface_path=Path(tmp) / "surface.dat",
            mask_path=Path(tmp) / "mask.dat",
            retain_window_details=False,
        )
        after = rss_mib()
        return {
            "raster_size": size,
            "batch_size": batch_size,
            "window_count": pred.window_count,
            "peak_patch_batch_bytes": pred.peak_patch_batch_bytes,
            "max_rss_delta_mib": round(max(0.0, after - before), 3),
        }


if __name__ == "__main__":
    print(json.dumps([run(512, 8), run(2048, 8)], indent=2))
