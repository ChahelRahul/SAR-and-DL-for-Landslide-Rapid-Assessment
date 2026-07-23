from __future__ import annotations
import json
from pathlib import Path
from app.acquisition.local_raster import read_sentinel1_stack
from app.config import ModelConfig, PipelineConfig
from app.inference.model import load_model
from app.inference.predict import predict_stack
from app.postprocessing.raster import write_detection_mask
from app.postprocessing.vector import write_detection_vectors
from app.schemas import OutputArtifact, PipelineRequest, PipelineResult


def filters_for_orbit(orbit: str) -> int:
    return 32 if orbit == "ASCENDING" else 64


def run_local(request: PipelineRequest, config: PipelineConfig = PipelineConfig()) -> PipelineResult:
    if request.input_raster is None:
        raise ValueError("run_local requires input_raster")
    raster = read_sentinel1_stack(request.input_raster)
    model, checksum = load_model(ModelConfig(
        orbit=request.orbit, weights_path=request.weights_path,
        filters_first_layer=filters_for_orbit(request.orbit)))
    prediction = predict_stack(model, raster.data, config.inference)
    output_dir = config.output_dir / request.request_id
    raster_path = write_detection_mask(output_dir / "detection-mask.tif", prediction.mask, raster.profile,
        orbit=request.orbit, relative_orbit=None, weights_sha256=checksum)
    vector_path = write_detection_vectors(output_dir / "detections.gpkg", prediction.mask,
        transform=raster.transform, crs=raster.crs, orbit=request.orbit,
        relative_orbit=None, weights_sha256=checksum)
    artifacts = [OutputArtifact("detection_mask", raster_path)]
    if vector_path: artifacts.append(OutputArtifact("detections", vector_path))
    result = PipelineResult(request.request_id, request.orbit,
        "succeeded" if vector_path else "succeeded_empty", checksum, artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "result.json"
    result.artifacts.append(OutputArtifact("metadata", metadata_path))
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result
