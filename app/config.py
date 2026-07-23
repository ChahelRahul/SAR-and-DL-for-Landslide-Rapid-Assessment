from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MODEL_NAME = "SAR-LRA"
MODEL_VERSION = "sar-lra-v2.0.0-beta.1"
EXPECTED_BAND_ORDER = ("postVV", "postVH", "diffVV", "diffVH")
Orbit = Literal["ASCENDING", "DESCENDING"]

@dataclass(frozen=True, slots=True)
class ModelConfig:
    orbit: Orbit
    weights_path: Path
    filters_first_layer: int
    patch_size: int = 64
    channels: int = 4
    learning_rate: float = 0.001
    dropout: float = 0.7

@dataclass(frozen=True, slots=True)
class InferenceConfig:
    probability_threshold: float = 0.6
    nms_overlap: float = 0.1
    batch_size: int = 512
    window_step: int = 32

@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    pre_days: int = 60
    post_days: int = 12
    scale_metres: int = 10
    output_crs: str = "EPSG:4326"

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    output_dir: Path = Path("outputs")
    acquisition: AcquisitionConfig = AcquisitionConfig()
    inference: InferenceConfig = InferenceConfig()
