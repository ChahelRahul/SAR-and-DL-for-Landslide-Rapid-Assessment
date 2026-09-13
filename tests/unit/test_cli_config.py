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


def test_threshold_raster_command_has_no_weights_requirement():
    args = build_parser().parse_args([
        "threshold-raster",
        "--probability-raster", "probability.tif",
        "--threshold", "0.75",
        "--output", "mask.tif",
    ])
    assert args.command == "threshold-raster"
    assert args.threshold == 0.75


def test_predict_raster_can_disable_binary_mask():
    args = build_parser().parse_args([
        "predict-raster",
        "--weights", "weights.hdf5",
        "--ascending", "input.tif",
        "--no-binary-mask",
    ])
    assert args.write_binary_mask is False
