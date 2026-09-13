from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.cli as cli


def test_help_documents_resource_requirements(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "Resource requirements" in capsys.readouterr().out


def test_validate_roi_does_not_import_tensorflow_or_earth_engine(tmp_path, monkeypatch, capsys):
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}))
    before = set(__import__("sys").modules)
    assert cli.main(["validate-roi", "--roi", str(roi), "--log-format", "json"]) == 0
    after = set(__import__("sys").modules)
    assert "tensorflow" not in after - before
    assert "ee" not in after - before
    err = capsys.readouterr().err.splitlines()
    assert json.loads(err[0])["event"] == "command_started"


def test_validate_roi_rejects_point_with_meaningful_exit(tmp_path):
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    assert cli.main(["validate-roi", "--roi", str(roi)]) == cli.EXIT_VALIDATION


def test_predict_raster_new_contract_parses():
    args = cli.build_parser().parse_args([
        "predict-raster", "--input", "input.tif", "--orbit", "ASCENDING",
        "--weights", "weights.h5", "--output-dir", "results",
    ])
    assert args.input == Path("input.tif")
    assert args.orbit == "ASCENDING"


def test_multi_orbit_parser():
    args = cli.build_parser().parse_args([
        "predict", "--roi", "roi.geojson", "--event-date", "2024-04-03",
        "--orbits", "ASCENDING,DESCENDING", "--output-dir", "results",
    ])
    assert cli._parse_orbits(args.orbits, args.orbit) == ["ASCENDING", "DESCENDING"]


def test_ctrl_c_returns_130(monkeypatch, tmp_path):
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}))
    monkeypatch.setattr(cli, "_command_validate_roi", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(["validate-roi", "--roi", str(roi)]) == 130


def test_output_containment_rejects_escape(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(RuntimeError, match="escaped output directory"):
        cli._ensure_generated_under(output, [tmp_path / "elsewhere.tif"])
