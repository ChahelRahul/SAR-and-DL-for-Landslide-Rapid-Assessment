from __future__ import annotations
import argparse, json
from datetime import date
from pathlib import Path
from app.config import PipelineConfig
from app.pipeline import run_local
from app.schemas import PipelineRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sar-lra")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--input-raster", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--orbit", choices=("ASCENDING", "DESCENDING"), required=True)
    parser.add_argument("--pre-end", type=date.fromisoformat, required=True)
    parser.add_argument("--post-start", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = PipelineRequest(args.request_id, args.orbit, args.pre_end, args.post_start,
                              args.weights, input_raster=args.input_raster)
    result = run_local(request, PipelineConfig(output_dir=args.output_dir))
    print(json.dumps(result.to_dict(), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
