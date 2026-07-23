from __future__ import annotations

import pytest

from app.cli import build_parser


def test_invalid_event_date_fails_during_argument_parsing() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "predict",
            "--roi", "roi.geojson",
            "--event-date", "not-a-date",
            "--weights", "weights.h5",
        ])


def test_predict_raster_infers_ascending_orbit() -> None:
    args = build_parser().parse_args([
        "predict-raster",
        "--ascending", "input.tif",
        "--weights", "weights.h5",
    ])
    assert args.ascending.name == "input.tif"
