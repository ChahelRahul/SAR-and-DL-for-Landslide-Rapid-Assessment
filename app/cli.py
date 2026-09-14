from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from app.config import AppConfig, Orbit, load_config

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_VALIDATION = 3
EXIT_ACQUISITION = 4
EXIT_PROCESSING = 5
EXIT_INTERRUPTED = 130

RESOURCE_HELP = """Resource requirements:
  predict/predict-raster: TensorFlow plus geospatial dependencies; memory scales with
  batch-size and 64x64x4 windows. Earth Engine prediction additionally requires an
  authenticated Earth Engine account and network access.
  acquire: Earth Engine + geemap + geospatial dependencies and network access.
  validate-roi: no TensorFlow or Earth Engine import.
All generated run artifacts are written below --output-dir.
"""


class CliValidationError(ValueError):
    pass


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-format", choices=("text", "json"), default="text",
        help="Diagnostic log format written to stderr (default: text)",
    )


def _add_config_options(parser: argparse.ArgumentParser, *, include_weights: bool = True) -> None:
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    if include_weights:
        parser.add_argument("--weights", type=Path, help="Local model weights")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--version")
    parser.add_argument("--orbit", choices=("ASCENDING", "DESCENDING"))
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--nms-overlap", type=float)
    parser.add_argument("--batch-size", type=int, help="Maximum model patches predicted at once")
    parser.add_argument("--inference-workers", type=int, help="Bounded patch-preparation worker count")
    parser.add_argument("--nms-diagnostic-limit", type=int, help="Maximum candidate windows retained for diagnostic NMS")
    parser.add_argument("--output-rows-per-chunk", type=int, help="Rows processed at once when finalizing disk-backed outputs")
    parser.add_argument("--pre-days", type=int)
    parser.add_argument("--post-days", type=int)
    parser.add_argument("--scale-m", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--max-roi-km2", type=float)
    parser.add_argument("--max-roi-width-km", type=float)
    parser.add_argument("--max-roi-height-km", type=float)
    parser.add_argument("--max-roi-vertices", type=int)
    parser.add_argument("--probability-aggregation", choices=("maximum",))
    parser.add_argument(
        "--vector-format", choices=("geojson", "geopackage", "both"),
        help="Vector output format; default is GeoJSON",
    )
    parser.add_argument(
        "--shapefile-zip", dest="write_shapefile_zip", action="store_true", default=None,
        help="Also write a zipped Shapefile compatibility export",
    )
    parser.add_argument(
        "--no-binary-mask", dest="write_binary_mask", action="store_false", default=None,
        help="Do not write the derived binary detection-mask GeoTIFF",
    )
    _add_runtime_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sar-lra",
        description="SAR-LRA command-line interface for Sentinel-1 landslide rapid assessment.",
        epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_runtime_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-roi", help="Validate ROI GeoJSON before acquisition", epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate.add_argument("--roi", type=Path, required=True)
    validate.add_argument("--config", type=Path)
    validate.add_argument("--max-roi-km2", type=float)
    validate.add_argument("--max-roi-width-km", type=float)
    validate.add_argument("--max-roi-height-km", type=float)
    validate.add_argument("--max-roi-vertices", type=int)
    _add_runtime_options(validate)

    acquire = subparsers.add_parser(
        "acquire", help="Acquire a prepared four-band Sentinel-1 raster", epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_options(acquire, include_weights=False)
    acquire.add_argument("--roi", type=Path, required=True)
    acquire.add_argument("--event-date", type=date.fromisoformat, required=True)
    acquire.add_argument("--provider", choices=("auto", "planetary-computer", "earth-engine"), default="auto")
    acquire.add_argument("--project", help="Earth Engine project")
    acquire.add_argument("--authenticate", action="store_true", help="Interactive local authentication only; containers should use mounted/ambient credentials")
    acquire.add_argument("--output-dir", type=Path, required=True)

    ee_parser = subparsers.add_parser(
        "predict", help="Acquire Sentinel-1 imagery and run inference", epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_options(ee_parser, include_weights=False)
    ee_parser.add_argument("--roi", type=Path, required=True, help="GeoJSON Feature or geometry")
    ee_parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    ee_parser.add_argument(
        "--orbits", default=None,
        help="Comma-separated orbit passes, e.g. ASCENDING,DESCENDING (overrides --orbit)",
    )
    ee_parser.add_argument("--weights", type=Path, help="Weights for a single-orbit run")
    ee_parser.add_argument("--ascending-weights", type=Path)
    ee_parser.add_argument("--descending-weights", type=Path)
    ee_parser.add_argument("--provider", choices=("auto", "planetary-computer", "earth-engine"), default="auto")
    ee_parser.add_argument("--project", help="Earth Engine project")
    ee_parser.add_argument("--authenticate", action="store_true", help="Interactive local authentication only; containers should use mounted/ambient credentials")
    ee_parser.add_argument("--output-dir", type=Path, required=True)

    raster_parser = subparsers.add_parser(
        "predict-raster", help="Run inference on a prepared raster", epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_options(raster_parser)
    raster_parser.add_argument("--input", type=Path, help="Prepared four-band GeoTIFF")
    legacy = raster_parser.add_mutually_exclusive_group()
    legacy.add_argument("--ascending", type=Path, help=argparse.SUPPRESS)
    legacy.add_argument("--descending", type=Path, help=argparse.SUPPRESS)
    raster_parser.add_argument("--roi", type=Path, help="Optional GeoJSON ROI for coverage validation")
    raster_parser.add_argument("--output-dir", type=Path, default=None)

    threshold_parser = subparsers.add_parser(
        "threshold-raster",
        help="Derive a binary mask from an existing probability raster without model inference",
        epilog=RESOURCE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    threshold_parser.add_argument("--probability-raster", type=Path, required=True)
    threshold_parser.add_argument("--threshold", type=float, required=True)
    threshold_parser.add_argument("--output", type=Path, required=True)
    _add_runtime_options(threshold_parser)
    return parser


def _effective_config(args: argparse.Namespace, *, orbit: str | None = None) -> AppConfig:
    inferred_orbit = orbit or getattr(args, "orbit", None)
    if getattr(args, "ascending", None) is not None:
        inferred_orbit = "ASCENDING"
    elif getattr(args, "descending", None) is not None:
        inferred_orbit = "DESCENDING"
    return load_config(getattr(args, "config", None)).with_overrides(
        version=getattr(args, "version", None), orbit=inferred_orbit,
        probability_threshold=getattr(args, "probability_threshold", None),
        nms_overlap=getattr(args, "nms_overlap", None), batch_size=getattr(args, "batch_size", None),
        pre_days=getattr(args, "pre_days", None), post_days=getattr(args, "post_days", None),
        scale_m=getattr(args, "scale_m", None), tile_size=getattr(args, "tile_size", None),
        overlap=getattr(args, "overlap", None), max_roi_km2=getattr(args, "max_roi_km2", None),
        max_roi_width_km=getattr(args, "max_roi_width_km", None),
        max_roi_height_km=getattr(args, "max_roi_height_km", None),
        max_roi_vertices=getattr(args, "max_roi_vertices", None),
        probability_aggregation=getattr(args, "probability_aggregation", None),
        write_binary_mask=getattr(args, "write_binary_mask", None),
        vector_format=getattr(args, "vector_format", None),
        write_shapefile_zip=getattr(args, "write_shapefile_zip", None),
        inference_workers=getattr(args, "inference_workers", None),
        nms_diagnostic_limit=getattr(args, "nms_diagnostic_limit", None),
        output_rows_per_chunk=getattr(args, "output_rows_per_chunk", None),
        output_dir=getattr(args, "output_dir", None),
    )


def _read_roi(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CliValidationError(f"ROI file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliValidationError(f"ROI is not valid GeoJSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliValidationError("ROI GeoJSON root must be an object")
    if payload.get("type") == "Feature":
        geometry = payload.get("geometry")
        if not isinstance(geometry, dict):
            raise CliValidationError("ROI Feature must contain a geometry")
        geometry_type = geometry.get("type")
    else:
        geometry_type = payload.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise CliValidationError("ROI must be a GeoJSON Polygon, MultiPolygon, or Feature containing one")
    return payload


def _safe_log_value(value: Any) -> Any:
    from app.acquisition.auth import redact_sensitive_text

    if isinstance(value, dict):
        return {str(k): _safe_log_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(v) for v in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def _emit(args: argparse.Namespace, level: str, event: str, message: str, **fields: Any) -> None:
    message = _safe_log_value(message)
    fields = _safe_log_value(fields)
    if getattr(args, "log_format", "text") == "json":
        print(json.dumps({"level": level, "event": event, "message": message, **fields}, sort_keys=True), file=sys.stderr)
    else:
        print(f"{level.upper()}: {message}", file=sys.stderr)


def _parse_orbits(raw: str | None, fallback: str | None) -> list[Orbit]:
    values = [x.strip().upper() for x in raw.split(",")] if raw else [fallback or "ASCENDING"]
    if not values or any(x not in {"ASCENDING", "DESCENDING"} for x in values):
        raise CliValidationError("orbits must contain ASCENDING and/or DESCENDING")
    # preserve order but reject duplicates to avoid overwriting results
    if len(set(values)) != len(values):
        raise CliValidationError("orbits must not contain duplicates")
    return values  # type: ignore[return-value]


def _bundled_weight(orbit: Orbit) -> Path | None:
    weight_dir = Path(__file__).resolve().parents[1] / "model" / "weights"
    matches = sorted(weight_dir.glob(f"*_{orbit}_*.hdf5"))
    return matches[0] if len(matches) == 1 else None


def _weight_for_orbit(args: argparse.Namespace, orbit: Orbit, orbit_count: int) -> Path:
    specific = args.ascending_weights if orbit == "ASCENDING" else args.descending_weights
    candidate = specific or (args.weights if orbit_count == 1 else None) or _bundled_weight(orbit)
    if candidate is None:
        option = "--ascending-weights" if orbit == "ASCENDING" else "--descending-weights"
        raise CliValidationError(f"No weights available for {orbit}; provide {option}")
    candidate = Path(candidate)
    if not candidate.is_file():
        raise CliValidationError(f"Weights file does not exist: {candidate}")
    return candidate


def _ensure_generated_under(output_dir: Path, paths: Iterable[Path]) -> None:
    root = output_dir.resolve()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved != root and root not in resolved.parents:
            raise RuntimeError(f"generated artifact escaped output directory: {path}")


def _validate_roi_payload(args: argparse.Namespace, roi: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    from app.roi import validate_roi

    report = validate_roi(
        roi,
        max_area_km2=config.processing.max_roi_km2,
        max_width_km=config.processing.max_roi_width_km,
        max_height_km=config.processing.max_roi_height_km,
        max_vertices=config.processing.max_roi_vertices,
    )
    if report.crosses_antimeridian:
        raise CliValidationError(
            "ROI crosses the antimeridian; this deployment rejects such requests to avoid ambiguous/global exports"
        )
    _emit(
        args, "info", "roi_validated",
        f"ROI area {report.area_km2:.2f} km²; bounds {report.bbox_width_km:.2f} x {report.bbox_height_km:.2f} km",
        **report.to_dict(),
    )
    return report.to_dict()


def _validate_event_date_arg(event_date: date) -> None:
    from app.roi import validate_event_date

    validate_event_date(event_date)


def _command_validate_roi(args: argparse.Namespace) -> dict[str, Any]:
    roi = _read_roi(args.roi)
    config = _effective_config(args)
    report = _validate_roi_payload(args, roi, config)
    return {"valid": True, "path": str(args.roi), **report}


def _command_acquire(args: argparse.Namespace) -> dict[str, Any]:
    roi = _read_roi(args.roi)
    config = _effective_config(args)
    roi_report = _validate_roi_payload(args, roi, config)
    _validate_event_date_arg(args.event_date)

    from app.acquisition.providers import resolve_provider
    orbit: Orbit = config.model.orbit
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = resolve_provider(args.provider)
    if provider == "planetary-computer":
        from app.acquisition.planetary_computer import acquire_intermediate_raster
        raster, cache_hit = acquire_intermediate_raster(
            roi_geojson=roi, event_date=args.event_date, orbit=orbit, config=config,
            cache_dir=output_dir / ".cache",
        )
    else:
        from app.acquisition.earth_engine import acquire_intermediate_raster
        raster, cache_hit = acquire_intermediate_raster(
            roi_geojson=roi, event_date=args.event_date, orbit=orbit, config=config,
            cache_dir=output_dir / ".cache", project=args.project, authenticate=args.authenticate,
        )
    target = output_dir / f"{orbit.lower()}.tif"
    if raster.path.resolve() != target.resolve():
        shutil.copy2(raster.path, target)
    _ensure_generated_under(output_dir, [raster.path, target])
    return {"status": "succeeded", "orbit": orbit, "raster": str(target), "cache_hit": cache_hit, "roi": roi_report}


def _command_predict_raster(args: argparse.Namespace) -> dict[str, Any]:
    from app.pipeline import run_raster
    from app.schemas import RasterInferenceRequest

    legacy_input = args.ascending or args.descending
    input_raster = args.input or legacy_input
    if input_raster is None:
        raise CliValidationError("predict-raster requires --input (or legacy --ascending/--descending)")
    config = _effective_config(args)
    if args.input is not None and args.orbit is None:
        raise CliValidationError("predict-raster with --input requires --orbit ASCENDING or DESCENDING")
    if args.weights is None:
        bundled = _bundled_weight(config.model.orbit)
        if bundled is None:
            raise CliValidationError("predict-raster requires --weights")
        args.weights = bundled
    if not Path(args.weights).is_file():
        raise CliValidationError(f"Weights file does not exist: {args.weights}")
    roi = _read_roi(args.roi) if args.roi else None
    if roi is not None:
        _validate_roi_payload(args, roi, config)
    result = run_raster(
        RasterInferenceRequest(
            request_id=args.request_id or uuid.uuid4().hex,
            orbit=config.model.orbit, weights_path=Path(args.weights), input_raster=input_raster,
            roi_geojson=roi,
        ), config,
    )
    _ensure_generated_under(config.output_dir, [a.path for a in result.artifacts])
    return result.to_dict()


def _command_predict(args: argparse.Namespace) -> dict[str, Any]:
    roi = _read_roi(args.roi)
    base_config = _effective_config(args)
    roi_report = _validate_roi_payload(args, roi, base_config)
    _validate_event_date_arg(args.event_date)
    orbits = _parse_orbits(args.orbits, args.orbit)

    from app.acquisition.providers import resolve_provider
    from app.schemas import EarthEngineRequest, PlanetaryComputerRequest
    provider = resolve_provider(args.provider)
    base_request_id = args.request_id or uuid.uuid4().hex
    results = []
    for orbit in orbits:
        config = _effective_config(args, orbit=orbit)
        weights = _weight_for_orbit(args, orbit, len(orbits))
        request_id = base_request_id if len(orbits) == 1 else f"{base_request_id}-{orbit.lower()}"
        if provider == "planetary-computer":
            from app.pipeline import run_planetary_computer
            result = run_planetary_computer(
                PlanetaryComputerRequest(
                    request_id=request_id, orbit=orbit, event_date=args.event_date,
                    weights_path=weights, roi_geojson=roi,
                    cache_dir=config.output_dir / ".cache" / orbit.lower(),
                ), config,
            )
        else:
            from app.pipeline import run_earth_engine
            result = run_earth_engine(
                EarthEngineRequest(
                    request_id=request_id, orbit=orbit, event_date=args.event_date,
                    weights_path=weights, roi_geojson=roi, project=args.project,
                    authenticate=args.authenticate, cache_dir=config.output_dir / ".cache" / orbit.lower(),
                ), config,
            )
        _ensure_generated_under(config.output_dir, [a.path for a in result.artifacts])
        item = result.to_dict()
        item["provider"] = provider
        results.append(item)
    if len(results) == 1:
        results[0]["roi_validation"] = roi_report
        return results[0]
    return {"status": "succeeded", "orbits": orbits, "roi_validation": roi_report, "results": results}


def _command_threshold(args: argparse.Namespace) -> dict[str, Any]:
    from app.postprocessing.raster import threshold_probability_raster

    path = threshold_probability_raster(args.probability_raster, args.output, threshold=args.threshold)
    return {"output": str(path), "probability_threshold": args.threshold}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(args, "info", "command_started", f"starting {args.command}", command=args.command)
        if args.command == "validate-roi":
            payload = _command_validate_roi(args)
        elif args.command == "acquire":
            payload = _command_acquire(args)
        elif args.command == "predict-raster":
            payload = _command_predict_raster(args)
        elif args.command == "predict":
            payload = _command_predict(args)
        else:
            payload = _command_threshold(args)
        print(json.dumps(payload, indent=2))
        _emit(args, "info", "command_succeeded", f"completed {args.command}", command=args.command)
        return EXIT_OK
    except KeyboardInterrupt:
        _emit(args, "warning", "interrupted", "operation cancelled by user")
        return EXIT_INTERRUPTED
    except CliValidationError as exc:
        _emit(args, "error", "validation_error", str(exc), exit_code=EXIT_VALIDATION)
        return EXIT_VALIDATION
    except (ValueError, FileNotFoundError) as exc:
        _emit(args, "error", "validation_error", str(exc), exit_code=EXIT_VALIDATION)
        return EXIT_VALIDATION
    except RuntimeError as exc:
        text = str(exc)
        acquisition_markers = ("Earth Engine", "earth-engine", "geemap", "authentication", "credentials")
        code = EXIT_ACQUISITION if any(x.lower() in text.lower() for x in acquisition_markers) else EXIT_PROCESSING
        _emit(args, "error", "processing_error", text, exit_code=code)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
