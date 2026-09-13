from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
from pathlib import Path

from app.acquisition.local_raster import RasterInput, read_sentinel1_stack
from app.config import AppConfig, ModelLoadConfig
from app.inference.model import load_model
from app.inference.predict import predict_stack, probability_surface_stats
from app.postprocessing.raster import write_detection_mask, write_probability_raster
from app.postprocessing.vector import (
    write_detection_geojson,
    write_detection_geopackage,
    write_detection_shapefile_zip,
)
from app.schemas import EarthEngineRequest, OutputArtifact, PipelineResult, RasterInferenceRequest
from app.roi import validate_event_date, validate_roi


def filters_for_orbit(orbit: str) -> int:
    return 32 if orbit == "ASCENDING" else 64



def _validate_processing_roi(roi_geojson: dict | None, config: AppConfig) -> dict | None:
    if roi_geojson is None:
        return None
    report = validate_roi(
        roi_geojson,
        max_area_km2=config.processing.max_roi_km2,
        max_width_km=config.processing.max_roi_width_km,
        max_height_km=config.processing.max_roi_height_km,
        max_vertices=config.processing.max_roi_vertices,
    )
    if report.crosses_antimeridian:
        raise ValueError(
            "ROI crosses the antimeridian; processing is rejected to avoid ambiguous/global exports"
        )
    return report.to_dict()


def run_raster(
    request: RasterInferenceRequest,
    config: AppConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    if request.orbit != config.model.orbit:
        raise ValueError("request orbit must match effective configuration orbit")
    roi_report = _validate_processing_roi(request.roi_geojson, config)
    if progress:
        progress("preprocessing")
    raster = read_sentinel1_stack(request.input_raster, config=config, roi_geojson=request.roi_geojson)
    return _run_inference(
        request_id=request.request_id,
        weights_path=request.weights_path,
        raster=raster,
        config=config,
        mode="prepared-raster",
        extra_artifacts=[],
        roi_geojson=request.roi_geojson,
        roi_report=roi_report,
        progress=progress,
    )


def run_earth_engine(
    request: EarthEngineRequest,
    config: AppConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    if request.orbit != config.model.orbit:
        raise ValueError("request orbit must match effective configuration orbit")
    roi_report = _validate_processing_roi(request.roi_geojson, config)
    validate_event_date(request.event_date)
    if progress:
        progress("acquiring")
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
    if progress:
        progress("preprocessing")
    return _run_inference(
        request_id=request.request_id,
        weights_path=request.weights_path,
        raster=raster,
        config=config,
        mode="earth-engine",
        extra_artifacts=[OutputArtifact("intermediate_raster", raster.path)],
        roi_geojson=request.roi_geojson,
        roi_report=roi_report,
        progress=progress,
    )


def _run_inference(
    *,
    request_id: str,
    weights_path: Path,
    raster: RasterInput,
    config: AppConfig,
    mode: str,
    extra_artifacts: list[OutputArtifact],
    roi_geojson: dict | None,
    roi_report: dict | None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    if progress:
        progress("inferencing")
    model, checksum = load_model(
        ModelLoadConfig(
            orbit=config.model.orbit,
            weights_path=weights_path,
            filters_first_layer=filters_for_orbit(config.model.orbit),
            patch_size=config.processing.tile_size,
        )
    )
    output_dir = config.output_dir / request_id
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / ".streaming"
    workspace.mkdir(parents=True, exist_ok=True)
    prediction = predict_stack(
        model,
        raster.data,
        config.model,
        config.processing,
        valid_mask=raster.valid_mask,
        surface_path=workspace / "probability.float32.memmap",
        mask_path=workspace / "mask.uint8.memmap",
        retain_window_details=False,
    )
    if progress:
        progress("postprocessing")
    effective_config = config.to_dict()
    relative_orbit_raw = raster.metadata.get("relative_orbit")
    try:
        relative_orbit = int(relative_orbit_raw) if relative_orbit_raw not in (None, "") else None
    except (TypeError, ValueError):
        relative_orbit = None

    probability_path = write_probability_raster(
        output_dir / "probability.tif",
        prediction.probability_surface,
        raster.profile,
        orbit=config.model.orbit,
        relative_orbit=relative_orbit,
        weights_sha256=checksum,
        effective_config=effective_config,
    )
    artifacts = [*extra_artifacts, OutputArtifact("probability_raster", probability_path)]

    if config.processing.write_binary_mask:
        mask_path = write_detection_mask(
            output_dir / "detection-mask.tif",
            prediction.mask,
            raster.profile,
            orbit=config.model.orbit,
            relative_orbit=relative_orbit,
            weights_sha256=checksum,
            effective_config=effective_config,
        )
        artifacts.append(OutputArtifact("detection_mask", mask_path))

    vector_kwargs = dict(
        transform=raster.transform,
        crs=raster.crs,
        orbit=config.model.orbit,
        relative_orbit=relative_orbit,
        weights_sha256=checksum,
        effective_config=effective_config,
        roi_geojson=roi_geojson,
    )
    geojson_path = None
    if config.processing.vector_format in {"geojson", "both"}:
        geojson_path = write_detection_geojson(
            output_dir / "detections.geojson",
            prediction.mask,
            prediction.probability_surface,
            **vector_kwargs,
        )
        artifacts.append(OutputArtifact("detections_geojson", geojson_path))

    if config.processing.vector_format in {"geopackage", "both"}:
        gpkg_path = write_detection_geopackage(
            output_dir / "detections.gpkg",
            prediction.mask,
            prediction.probability_surface,
            **vector_kwargs,
        )
        artifacts.append(OutputArtifact("detections_geopackage", gpkg_path))

    if config.processing.write_shapefile_zip:
        shp_zip = write_detection_shapefile_zip(
            output_dir / "detections-shapefile.zip",
            prediction.mask,
            prediction.probability_surface,
            **vector_kwargs,
        )
        artifacts.append(OutputArtifact("detections_shapefile_zip", shp_zip))

    surface_stats = probability_surface_stats(
        prediction.probability_surface, rows_per_chunk=config.processing.output_rows_per_chunk
    )
    probability_stats = {
        "aggregation": config.processing.probability_aggregation,
        "threshold": config.model.probability_threshold,
        "window_count": prediction.window_count,
        **surface_stats,
        "streamed": True,
        "batch_size": config.model.batch_size,
        "inference_workers": config.processing.inference_workers,
        "peak_patch_batch_bytes": prediction.peak_patch_batch_bytes,
    }
    nms_diagnostics = {
        "role": "diagnostic_only",
        "output_geometry_source": "thresholded_probability_surface",
        "candidate_window_count": prediction.candidate_count,
        "diagnostic_candidates_retained": len(prediction.candidate_windows),
        "diagnostics_truncated": prediction.nms_diagnostics_truncated,
        "legacy_metric": "intersection_over_comparison_box_area",
        "legacy_retained_count": len(prediction.legacy_selected_windows),
        "iou_metric": "intersection_over_union",
        "iou_retained_count": len(prediction.iou_selected_windows),
        "overlap_threshold": config.model.nms_overlap,
        "geometry_semantics": "model_windows_are_candidate_areas_not_landslide_boundaries",
    }
    result = PipelineResult(
        request_id=request_id,
        orbit=config.model.orbit,
        status="succeeded" if np.any(prediction.mask == 1) else "succeeded_empty",
        weights_sha256=checksum,
        mode=mode,  # type: ignore[arg-type]
        artifacts=artifacts,
        effective_configuration=effective_config,
        input_metadata=raster.metadata,
        processing_metadata={
            "raster_validation": raster.metadata.get("validation", {}),
            "roi_validation": roi_report,
            "probability_surface": probability_stats,
            "nms_diagnostics": nms_diagnostics,
            "vector_outputs": {
                "format": config.processing.vector_format,
                "geojson_crs": "EPSG:4326",
                "geopackage_crs": str(raster.crs),
                "roi_clipped": roi_geojson is not None,
                "shapefile_zip": config.processing.write_shapefile_zip,
                "empty_result_policy": "valid_empty_dataset",
            },
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "result.json"
    result.artifacts.append(OutputArtifact("metadata", metadata_path))
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    # Disk-backed working arrays are internal and are removed after durable outputs are written.
    del prediction
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)
    return result


# Backward-compatible alias retained for one release.
def run_local(request: RasterInferenceRequest, config: AppConfig) -> PipelineResult:
    return run_raster(request, config)
