from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import os
from typing import Any, Literal, Mapping

MODEL_NAME = "SAR-LRA"
MODEL_VERSION = "sar-lra-v2.0.0-beta.1"
EXPECTED_BAND_ORDER = ("postVV", "postVH", "diffVV", "diffVH")
Orbit = Literal["ASCENDING", "DESCENDING"]
VALID_ORBITS = {"ASCENDING", "DESCENDING"}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    version: str = "v2"
    orbit: Orbit = "ASCENDING"
    probability_threshold: float = 0.6
    nms_overlap: float = 0.1
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.orbit not in VALID_ORBITS:
            raise ValueError(f"orbit must be one of {sorted(VALID_ORBITS)}")
        if not self.version.strip():
            raise ValueError("model.version must not be empty")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("model.probability_threshold must be between 0 and 1")
        if not 0.0 <= self.nms_overlap <= 1.0:
            raise ValueError("model.nms_overlap must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("model.batch_size must be positive")


@dataclass(frozen=True, slots=True)
class ImageryConfig:
    pre_days: int = 60
    post_days: int = 12
    scale_m: int = 10

    def __post_init__(self) -> None:
        if self.pre_days <= 0:
            raise ValueError("imagery.pre_days must be positive")
        if self.post_days <= 0:
            raise ValueError("imagery.post_days must be positive")
        if self.scale_m <= 0:
            raise ValueError("imagery.scale_m must be positive")


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    tile_size: int = 64
    overlap: float = 0.5
    max_roi_km2: float = 10_000.0
    max_roi_width_km: float = 500.0
    max_roi_height_km: float = 500.0
    max_roi_vertices: int = 50_000
    resolution_tolerance: float = 0.2
    max_nodata_fraction: float = 0.25
    value_min_db: float = -60.0
    value_max_db: float = 30.0
    probability_aggregation: str = "maximum"
    write_binary_mask: bool = True
    vector_format: str = "geojson"
    write_shapefile_zip: bool = False
    inference_workers: int = 1
    nms_diagnostic_limit: int = 100000
    output_rows_per_chunk: int = 1024

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("processing.tile_size must be positive")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("processing.overlap must be at least 0 and less than 1")
        if self.max_roi_km2 <= 0:
            raise ValueError("processing.max_roi_km2 must be positive")
        if self.max_roi_width_km <= 0 or self.max_roi_height_km <= 0:
            raise ValueError("processing ROI dimension limits must be positive")
        if self.max_roi_vertices <= 3:
            raise ValueError("processing.max_roi_vertices must be greater than 3")
        if not 0 <= self.resolution_tolerance <= 1:
            raise ValueError("processing.resolution_tolerance must be between 0 and 1")
        if not 0 <= self.max_nodata_fraction < 1:
            raise ValueError("processing.max_nodata_fraction must be at least 0 and less than 1")
        if self.value_min_db >= self.value_max_db:
            raise ValueError("processing.value_min_db must be less than value_max_db")
        if self.probability_aggregation != "maximum":
            raise ValueError("processing.probability_aggregation currently supports only maximum")
        if not isinstance(self.write_binary_mask, bool):
            raise ValueError("processing.write_binary_mask must be a boolean")
        if self.vector_format not in {"geojson", "geopackage", "both"}:
            raise ValueError("processing.vector_format must be geojson, geopackage, or both")
        if not isinstance(self.write_shapefile_zip, bool):
            raise ValueError("processing.write_shapefile_zip must be a boolean")
        if self.inference_workers <= 0:
            raise ValueError("processing.inference_workers must be positive")
        if self.nms_diagnostic_limit <= 0:
            raise ValueError("processing.nms_diagnostic_limit must be positive")
        if self.output_rows_per_chunk <= 0:
            raise ValueError("processing.output_rows_per_chunk must be positive")
        if self.window_step <= 0:
            raise ValueError("processing.overlap leaves no positive window step")

    @property
    def window_step(self) -> int:
        return max(1, round(self.tile_size * (1.0 - self.overlap)))


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    imagery: ImageryConfig = field(default_factory=ImageryConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["processing"]["window_step"] = self.processing.window_step
        return data

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AppConfig":
        raw = raw or {}
        _reject_unknown(raw, {"model", "imagery", "processing", "output_dir"}, "configuration")
        model_raw = _mapping(raw.get("model"), "model")
        imagery_raw = _mapping(raw.get("imagery"), "imagery")
        processing_raw = _mapping(raw.get("processing"), "processing")
        _reject_unknown(model_raw, {"version", "orbit", "probability_threshold", "nms_overlap", "batch_size"}, "model")
        _reject_unknown(imagery_raw, {"pre_days", "post_days", "scale_m"}, "imagery")
        _reject_unknown(processing_raw, {"tile_size", "overlap", "max_roi_km2", "max_roi_width_km", "max_roi_height_km", "max_roi_vertices", "resolution_tolerance", "max_nodata_fraction", "value_min_db", "value_max_db", "probability_aggregation", "write_binary_mask", "vector_format", "write_shapefile_zip", "inference_workers", "nms_diagnostic_limit", "output_rows_per_chunk"}, "processing")
        return cls(
            model=ModelConfig(**model_raw),
            imagery=ImageryConfig(**imagery_raw),
            processing=ProcessingConfig(**processing_raw),
            output_dir=Path(raw.get("output_dir", "outputs")),
        )

    def with_overrides(self, **overrides: Any) -> "AppConfig":
        model = self.model
        imagery = self.imagery
        processing = self.processing
        output_dir = self.output_dir
        model_fields = {"version", "orbit", "probability_threshold", "nms_overlap", "batch_size"}
        imagery_fields = {"pre_days", "post_days", "scale_m"}
        processing_fields = {"tile_size", "overlap", "max_roi_km2", "max_roi_width_km", "max_roi_height_km", "max_roi_vertices", "resolution_tolerance", "max_nodata_fraction", "value_min_db", "value_max_db", "probability_aggregation", "write_binary_mask", "vector_format", "write_shapefile_zip", "inference_workers", "nms_diagnostic_limit", "output_rows_per_chunk"}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in model_fields:
                model = replace(model, **{key: value})
            elif key in imagery_fields:
                imagery = replace(imagery, **{key: value})
            elif key in processing_fields:
                processing = replace(processing, **{key: value})
            elif key == "output_dir":
                output_dir = Path(value)
            else:
                raise ValueError(f"Unknown configuration override: {key}")
        return AppConfig(model=model, imagery=imagery, processing=processing, output_dir=output_dir)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return _with_environment_limits(AppConfig())
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires PyYAML") from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return _with_environment_limits(AppConfig.from_mapping(raw))


def _with_environment_limits(config: AppConfig) -> AppConfig:
    mapping = {
        "SAR_LRA_MAX_ROI_KM2": ("max_roi_km2", float),
        "SAR_LRA_MAX_ROI_WIDTH_KM": ("max_roi_width_km", float),
        "SAR_LRA_MAX_ROI_HEIGHT_KM": ("max_roi_height_km", float),
        "SAR_LRA_MAX_ROI_VERTICES": ("max_roi_vertices", int),
    }
    overrides: dict[str, Any] = {}
    for env_name, (field_name, cast) in mapping.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            overrides[field_name] = cast(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} has invalid value: {raw!r}") from exc
    return config.with_overrides(**overrides) if overrides else config


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    orbit: Orbit
    weights_path: Path
    filters_first_layer: int
    patch_size: int
    channels: int = 4
    learning_rate: float = 0.001
    dropout: float = 0.7
