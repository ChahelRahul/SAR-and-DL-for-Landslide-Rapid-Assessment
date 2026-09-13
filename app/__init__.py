"""SAR-LRA package.

Importing this package performs no Earth Engine authentication, network access,
or model-weight loading.
"""

__version__ = "2.0.0b1"

from .config import AppConfig, ImageryConfig, ModelConfig, ProcessingConfig
from .schemas import EarthEngineRequest, PipelineResult, RasterInferenceRequest

__all__ = [
    "AppConfig",
    "ImageryConfig",
    "ModelConfig",
    "ProcessingConfig",
    "EarthEngineRequest",
    "RasterInferenceRequest",
    "PipelineResult",
]
