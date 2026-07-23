from __future__ import annotations

import json
from pathlib import Path

from app.acquisition.local_raster import RasterInput, read_sentinel1_stack
from app.config import AppConfig, ModelLoadConfig
from app.inference.model import load_model
from app.inference.predict import predict_stack
from app.postprocessing.raster import write_detection_mask
from app.postprocessing.vector import write_detection_vectors
from app.schemas import EarthEngineRequest, OutputArtifact, PipelineResult, RasterInferenceRequest


def filters_for_orbit(orbit: str) -> int:
    return 32 if orbit == "ASCENDING" else 64


def run_raster(request: RasterInferenceRequest, config: AppConfig) -> PipelineResult:
    if request.orbit != config.model.orbit:
        raise ValueError("request orbit must match effective configuration orbit")
    raster = read_sentinel1_stack(request.input_raster, config=config, roi_geojson=request.roi_geojson)
    return _run_inference(
        request_id=request.request_id,
        weights_path=request.weights_path,
        raster=raster,
        config=config,
        mode="prepared-raster",
        extra_artifacts=[],
    )


def run_earth_engine(request: EarthEngineRequest, config: AppConfig) -> PipelineResult:
    if request.orbit != config.model.orbit:
        raise ValueError("request orbit must match effective configuration orbit")
    # Deliberately imported here so prepared-raster inference has no EE dependency.
    from app.acquisition.earth_engine import acquire_intermediate_raster

    raster, cache_hit = acquire_intermediate_raster(
        roi_geojson=request.roi_geojson,
        event_date=request.event_date,
        orbit=request.orbit,
        config=config,
        cache_dir=request.cache_dir,
        project=request.project,
        authenticate=request.authenticate,
    )
    raster.metadata["cache_hit"] = cache_hit
    return _run_inference(
        request_id=request.request_id,
        weights_path=request.weights_path,
        raster=raster,
        config=config,
        mode="earth-engine",
        extra_artifacts=[OutputArtifact("intermediate_raster", raster.path)],
    )


def _run_inference(
    *,
    request_id: str,
    weights_path: Path,
    raster: RasterInput,
    config: AppConfig,
    mode: str,
    extra_artifacts: list[OutputArtifact],
) -> PipelineResult:
    model, checksum = load_model(
        ModelLoadConfig(
            orbit=config.model.orbit,
            weights_path=weights_path,
            filters_first_layer=filters_for_orbit(config.model.orbit),
            patch_size=config.processing.tile_size,
        )
    )
    prediction = predict_stack(model, raster.data, config.model, config.processing)
    output_dir = config.output_dir / request_id
    effective_config = config.to_dict()
    raster_path = write_detection_mask(
        output_dir / "detection-mask.tif",
        prediction.mask,
        raster.profile,
        orbit=config.model.orbit,
        relative_orbit=None,
        weights_sha256=checksum,
        effective_config=effective_config,
    )
    vector_path = write_detection_vectors(
        output_dir / "detections.gpkg",
        prediction.mask,
        transform=raster.transform,
        crs=raster.crs,
        orbit=config.model.orbit,
        relative_orbit=None,
        weights_sha256=checksum,
        effective_config=effective_config,
    )
    artifacts = [*extra_artifacts, OutputArtifact("detection_mask", raster_path)]
    if vector_path:
        artifacts.append(OutputArtifact("detections", vector_path))
    result = PipelineResult(
        request_id=request_id,
        orbit=config.model.orbit,
        status="succeeded" if vector_path else "succeeded_empty",
        weights_sha256=checksum,
        mode=mode,  # type: ignore[arg-type]
        artifacts=artifacts,
        effective_configuration=effective_config,
        input_metadata=raster.metadata,
        processing_metadata={"raster_validation": raster.metadata.get("validation", {})},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "result.json"
    result.artifacts.append(OutputArtifact("metadata", metadata_path))
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


# Backward-compatible alias retained for one release.
def run_local(request: RasterInferenceRequest, config: AppConfig) -> PipelineResult:
    return run_raster(request, config)
