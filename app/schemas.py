from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .config import MODEL_NAME, MODEL_VERSION, Orbit

PipelineMode = Literal["earth-engine", "prepared-raster"]


@dataclass(frozen=True, slots=True)
class RasterInferenceRequest:
    request_id: str
    orbit: Orbit
    weights_path: Path
    input_raster: Path


@dataclass(frozen=True, slots=True)
class EarthEngineRequest:
    request_id: str
    orbit: Orbit
    event_date: date
    weights_path: Path
    roi_geojson: dict[str, Any]
    project: str | None = None
    authenticate: bool = False
    cache_dir: Path = Path("cache")


@dataclass(slots=True)
class OutputArtifact:
    kind: str
    path: Path


@dataclass(slots=True)
class PipelineResult:
    request_id: str
    orbit: Orbit
    status: str
    weights_sha256: str
    mode: PipelineMode
    artifacts: list[OutputArtifact] = field(default_factory=list)
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    effective_configuration: dict[str, Any] = field(default_factory=dict)
    input_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [
            {"kind": item.kind, "path": str(item.path)} for item in self.artifacts
        ]
        return data
