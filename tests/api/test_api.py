from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import ApiSettings, create_app
from app.schemas import PipelineResult


def settings(tmp_path: Path) -> ApiSettings:
    s = ApiSettings()
    s.input_root = (tmp_path / "input").resolve()
    s.output_root = (tmp_path / "output").resolve()
    s.input_root.mkdir()
    s.output_root.mkdir()
    s.max_concurrent = 1
    return s


def test_health_is_lightweight(tmp_path):
    app = create_app(settings(tmp_path))
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_path_traversal_rejected(tmp_path):
    s = settings(tmp_path)
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"x")
    response = TestClient(create_app(s)).post(
        "/v1/predict-raster",
        json={"input_raster": str(outside), "orbit": "ASCENDING"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_invalid_request_has_stable_error_envelope(tmp_path):
    response = TestClient(create_app(settings(tmp_path))).post(
        "/v1/predict-raster", json={"orbit": "SIDEWAYS"}
    )
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "request_validation_error"


def test_busy_service_returns_429(tmp_path):
    app = create_app(settings(tmp_path))
    raster = app.state.settings.input_root / "x.tif"
    raster.write_bytes(b"x")
    app.state.inference_gate.acquire()
    try:
        response = TestClient(app).post(
            "/v1/predict-raster", json={"input_raster": "x.tif", "orbit": "ASCENDING"}
        )
    finally:
        app.state.inference_gate.release()
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "busy"
    assert response.headers["retry-after"] == "5"


def test_predict_raster_routes_to_shared_pipeline(tmp_path, monkeypatch):
    s = settings(tmp_path)
    raster = s.input_root / "scene.tif"
    raster.write_bytes(b"fixture")
    seen = {}

    def fake_run(request, config):
        seen["request"] = request
        seen["config"] = config
        return PipelineResult(
            request_id=request.request_id,
            orbit=request.orbit,
            status="succeeded_empty",
            weights_sha256="abc",
            mode="prepared-raster",
        )

    monkeypatch.setattr("app.pipeline.run_raster", fake_run)
    response = TestClient(create_app(s)).post(
        "/v1/predict-raster",
        json={
            "input_raster": "scene.tif",
            "orbit": "ASCENDING",
            "request_id": "api-test",
            "probability_threshold": 0.7,
        },
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "api-test"
    assert seen["request"].input_raster == raster.resolve()
    assert seen["config"].output_dir == s.output_root
    assert seen["config"].model.probability_threshold == 0.7


def test_request_id_rejects_path_characters(tmp_path):
    s = settings(tmp_path)
    raster = s.input_root / "scene.tif"
    raster.write_bytes(b"fixture")
    response = TestClient(create_app(s)).post(
        "/v1/predict-raster",
        json={"input_raster": "scene.tif", "request_id": "../escape"},
    )
    assert response.status_code == 422
