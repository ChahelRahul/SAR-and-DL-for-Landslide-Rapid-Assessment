from __future__ import annotations

import argparse
import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from app.config import AppConfig, load_config
from app.pipeline import run_earth_engine, run_raster
from app.schemas import EarthEngineRequest, RasterInferenceRequest


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--weights", type=Path, required=True, help="Local model weights")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--version")
    parser.add_argument("--orbit", choices=("ASCENDING", "DESCENDING"))
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--nms-overlap", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pre-days", type=int)
    parser.add_argument("--post-days", type=int)
    parser.add_argument("--scale-m", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--max-roi-km2", type=float)
    parser.add_argument("--output-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sar-lra")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ee_parser = subparsers.add_parser("predict", help="Acquire Sentinel-1 imagery and run inference")
    _add_config_options(ee_parser)
    ee_parser.add_argument("--roi", type=Path, required=True, help="GeoJSON Feature or geometry")
    ee_parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    ee_parser.add_argument("--source", choices=("earth-engine",), default="earth-engine")
    ee_parser.add_argument("--project", help="Earth Engine project")
    ee_parser.add_argument("--authenticate", action="store_true")
    ee_parser.add_argument("--cache-dir", type=Path, default=Path("cache"))

    raster_parser = subparsers.add_parser("predict-raster", help="Run inference on a prepared raster")
    _add_config_options(raster_parser)
    orbit_group = raster_parser.add_mutually_exclusive_group(required=True)
    orbit_group.add_argument("--ascending", type=Path)
    orbit_group.add_argument("--descending", type=Path)
    return parser


def _effective_config(args: argparse.Namespace) -> AppConfig:
    inferred_orbit = args.orbit
    if getattr(args, "ascending", None) is not None:
        inferred_orbit = "ASCENDING"
    elif getattr(args, "descending", None) is not None:
        inferred_orbit = "DESCENDING"
    return load_config(args.config).with_overrides(
        version=args.version,
        orbit=inferred_orbit,
        probability_threshold=args.probability_threshold,
        nms_overlap=args.nms_overlap,
        batch_size=args.batch_size,
        pre_days=args.pre_days,
        post_days=args.post_days,
        scale_m=args.scale_m,
        tile_size=args.tile_size,
        overlap=args.overlap,
        max_roi_km2=args.max_roi_km2,
        output_dir=args.output_dir,
    )


def _read_roi(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"ROI file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROI is not valid GeoJSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") not in {
        "Feature", "Polygon", "MultiPolygon"
    }:
        raise ValueError("ROI must be a GeoJSON Feature, Polygon, or MultiPolygon")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _effective_config(args)
        request_id = args.request_id or uuid.uuid4().hex
        if args.command == "predict-raster":
            input_raster = args.ascending or args.descending
            result = run_raster(
                RasterInferenceRequest(
                    request_id=request_id,
                    orbit=config.model.orbit,
                    weights_path=args.weights,
                    input_raster=input_raster,
                ),
                config,
            )
        else:
            result = run_earth_engine(
                EarthEngineRequest(
                    request_id=request_id,
                    orbit=config.model.orbit,
                    event_date=args.event_date,
                    weights_path=args.weights,
                    roi_geojson=_read_roi(args.roi),
                    project=args.project,
                    authenticate=args.authenticate,
                    cache_dir=args.cache_dir,
                ),
                config,
            )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Processing error: {exc}") from exc
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
