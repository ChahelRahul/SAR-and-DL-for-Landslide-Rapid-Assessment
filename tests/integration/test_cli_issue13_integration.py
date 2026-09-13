from __future__ import annotations

import json
from pathlib import Path

from app import cli
from app.schemas import OutputArtifact, PipelineResult


def test_prepared_raster_cli_routes_new_input_contract(tmp_path, monkeypatch, capsys):
    import app.pipeline

    input_raster = tmp_path / "input.tif"
    weights = tmp_path / "weights.h5"
    roi = tmp_path / "roi.geojson"
    output = tmp_path / "results"
    input_raster.write_bytes(b"fixture")
    weights.write_bytes(b"fixture")
    roi.write_text(json.dumps({"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}))

    def fake_run(request, config):
        run_dir = config.output_dir / request.request_id
        run_dir.mkdir(parents=True)
        artifact = run_dir / "result.json"
        artifact.write_text("{}")
        return PipelineResult(
            request_id=request.request_id,
            orbit=request.orbit,
            status="succeeded_empty",
            weights_sha256="fixture",
            mode="prepared-raster",
            artifacts=[OutputArtifact("metadata", artifact)],
        )

    monkeypatch.setattr(app.pipeline, "run_raster", fake_run)
    code = cli.main([
        "predict-raster", "--input", str(input_raster), "--orbit", "ASCENDING",
        "--weights", str(weights), "--roi", str(roi), "--output-dir", str(output),
        "--request-id", "cli-integration",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["request_id"] == "cli-integration"
    assert Path(payload["artifacts"][0]["path"]).resolve().is_relative_to(output.resolve())
