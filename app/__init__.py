"""SAR-LRA reusable application package.

Importing :mod:`app` performs no network access, Earth Engine authentication,
or model-weight downloads. Expensive optional dependencies are imported only
inside the functions that require them.
"""
from .config import MODEL_NAME, MODEL_VERSION, PipelineConfig
from .schemas import PipelineRequest, PipelineResult

__all__ = [
    "MODEL_NAME", "MODEL_VERSION", "PipelineConfig",
    "PipelineRequest", "PipelineResult",
]
