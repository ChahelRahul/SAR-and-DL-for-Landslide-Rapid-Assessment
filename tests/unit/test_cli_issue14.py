from __future__ import annotations

import json
from pathlib import Path

from app import cli


def write_roi(path: Path, coords) -> Path:
    path.write_text(json.dumps({"type": "Polygon", "coordinates": [coords]}), encoding="utf-8")
    return path


def test_validate_roi_returns_area_and_dimensions(tmp_path, capsys):
    roi = write_roi(tmp_path / "roi.geojson", [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]])
    assert cli.main(["validate-roi", "--roi", str(roi)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["area_km2"] > 0
    assert payload["bbox_width_km"] > 0
    assert payload["vertex_count"] == 5


def test_validate_roi_rejects_oversize_via_environment(tmp_path, monkeypatch, capsys):
    roi = write_roi(tmp_path / "roi.geojson", [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]])
    monkeypatch.setenv("SAR_LRA_MAX_ROI_KM2", "1")
    assert cli.main(["validate-roi", "--roi", str(roi)]) == cli.EXIT_VALIDATION
    assert "exceeds limit" in capsys.readouterr().err


def test_validate_roi_rejects_antimeridian_for_processing(tmp_path, capsys):
    roi = write_roi(tmp_path / "roi.geojson", [[179.5, 10], [-179.5, 10], [-179.5, 11], [179.5, 11], [179.5, 10]])
    assert cli.main(["validate-roi", "--roi", str(roi), "--max-roi-km2", "20000"]) == cli.EXIT_VALIDATION
    assert "antimeridian" in capsys.readouterr().err.lower()


def test_acquire_rejects_pre_sentinel_date_before_importing_earth_engine(tmp_path, monkeypatch, capsys):
    roi = write_roi(tmp_path / "roi.geojson", [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]])
    # If acquisition is reached this sentinel import replacement makes the test fail.
    import app.acquisition.earth_engine as ee_module
    monkeypatch.setattr(ee_module, "acquire_intermediate_raster", lambda **kwargs: (_ for _ in ()).throw(AssertionError("acquisition started")))
    code = cli.main([
        "acquire", "--roi", str(roi), "--event-date", "2014-10-02",
        "--orbit", "ASCENDING", "--output-dir", str(tmp_path / "out"),
    ])
    assert code == cli.EXIT_VALIDATION
    assert "predates supported Sentinel-1" in capsys.readouterr().err
