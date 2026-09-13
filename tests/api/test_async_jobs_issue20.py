from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import ApiSettings, create_app
from app.jobs import InMemoryJobBackend, JobRecord, JobState
from app.schemas import PipelineResult
from app.worker import execute_job


def settings(tmp_path: Path) -> ApiSettings:
    s = ApiSettings()
    s.input_root = (tmp_path / "input").resolve()
    s.output_root = (tmp_path / "output").resolve()
    s.input_root.mkdir()
    s.output_root.mkdir()
    s.job_retention_seconds = 60
    return s


def test_submit_returns_202_and_queued_job(tmp_path):
    s = settings(tmp_path)
    (s.input_root / "scene.tif").write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    response = TestClient(create_app(s, job_backend=backend)).post(
        "/v1/jobs",
        json={"source": "prepared-raster", "input_raster": "scene.tif", "orbit": "ASCENDING", "request_id": "job-1"},
    )
    assert response.status_code == 202
    assert response.json()["job_id"] == "job-1"
    assert response.json()["state"] == "queued"
    assert backend.dequeue() == "job-1"
    assert backend.get("job-1").state == JobState.QUEUED


def test_job_status_and_result_contract(tmp_path):
    s = settings(tmp_path)
    backend = InMemoryJobBackend()
    record = JobRecord("done", JobState.COMPLETED, "prepared-raster", {}, result={"answer": 42})
    backend.create(record)
    client = TestClient(create_app(s, job_backend=backend))
    status = client.get("/v1/jobs/done")
    assert status.status_code == 200
    assert status.json()["state"] == "completed"
    assert "payload" not in status.json()
    result = client.get("/v1/jobs/done/result")
    assert result.status_code == 200
    assert result.json() == {"answer": 42}


def test_retry_completed_is_idempotent(tmp_path):
    backend = InMemoryJobBackend()
    backend.create(JobRecord("done", JobState.COMPLETED, "prepared-raster", {}, result={"ok": True}))
    client = TestClient(create_app(settings(tmp_path), job_backend=backend))
    response = client.post("/v1/jobs/done/retry")
    assert response.status_code == 202
    assert response.json() == {"job_id": "done", "state": "completed", "requeued": False}
    assert backend.dequeue() is None


def test_cancelled_queued_job_is_not_executed(tmp_path, monkeypatch):
    s = settings(tmp_path)
    backend = InMemoryJobBackend()
    backend.create(JobRecord("cancel-me", JobState.CANCELLED, "prepared-raster", {"orbit": "ASCENDING"}))
    called = False
    def fail(*args, **kwargs):
        nonlocal called
        called = True
    monkeypatch.setattr("app.pipeline.run_raster", fail)
    assert execute_job("cancel-me", backend, s) is None
    assert called is False


def test_worker_records_pipeline_stages_and_completion(tmp_path, monkeypatch):
    s = settings(tmp_path)
    scene = s.input_root / "scene.tif"
    scene.write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    backend.create(JobRecord("job-2", JobState.QUEUED, "prepared-raster", {"orbit": "ASCENDING", "input_raster": str(scene), "weights_path": None}))
    states = []
    original_save = backend.save
    def tracking_save(record):
        states.append(record.state)
        original_save(record)
    backend.save = tracking_save  # type: ignore[method-assign]

    def fake_weights(path, orbit, settings):
        return Path("weights.hdf5")
    monkeypatch.setattr("app.worker._weights", fake_weights)
    def fake_run(request, config, *, progress=None):
        progress("preprocessing")
        progress("inferencing")
        progress("postprocessing")
        return PipelineResult(request.request_id, request.orbit, "succeeded_empty", "abc", "prepared-raster")
    monkeypatch.setattr("app.pipeline.run_raster", fake_run)
    result = execute_job("job-2", backend, s)
    assert result["request_id"] == "job-2"
    assert JobState.PREPROCESSING in states
    assert JobState.INFERENCING in states
    assert JobState.POSTPROCESSING in states
    assert backend.get("job-2").state == JobState.COMPLETED


def test_failed_job_exposes_safe_summary(tmp_path, monkeypatch):
    s = settings(tmp_path)
    backend = InMemoryJobBackend()
    backend.create(JobRecord("bad", JobState.QUEUED, "prepared-raster", {"orbit": "ASCENDING", "input_raster": "x", "weights_path": None}))
    monkeypatch.setattr("app.worker._weights", lambda *a, **k: Path("weights.hdf5"))
    def explode(*args, **kwargs):
        raise RuntimeError("SECRET provider internals /credentials/key.json")
    monkeypatch.setattr("app.pipeline.run_raster", explode)
    execute_job("bad", backend, s)
    record = backend.get("bad")
    assert record.state == JobState.FAILED
    assert record.error == {"code": "processing_failed", "message": "job processing failed"}
    response = TestClient(create_app(s, job_backend=backend)).get("/v1/jobs/bad/result")
    assert response.status_code == 200
    assert "SECRET" not in response.text


def test_retention_expires_jobs():
    backend = InMemoryJobBackend()
    backend.create(JobRecord("short", JobState.QUEUED, "prepared-raster", {}, retention_seconds=1))
    assert backend.get("short") is not None
    # Directly age the deterministic in-memory backend instead of sleeping a full second.
    record, _ = backend.records["short"]
    backend.records["short"] = (record, time.time() - 1)
    assert backend.get("short") is None


def test_all_required_states_are_defined():
    assert {state.value for state in JobState} == {
        "queued", "acquiring", "preprocessing", "inferencing", "postprocessing", "completed", "failed", "cancelled"
    }


def test_redis_backend_applies_retention_ttl_without_live_redis():
    from app.jobs import RedisJobBackend

    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.expiries = {}
            self.queue = []
        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.values:
                return False
            self.values[key] = value
            self.expiries[key] = ex
            return True
        def get(self, key):
            return self.values.get(key)
        def rpush(self, key, value):
            self.queue.append((key, value))
        def blpop(self, key, timeout=0):
            for i, (queue_key, value) in enumerate(self.queue):
                if queue_key == key:
                    self.queue.pop(i)
                    return (key, value)
            return None

    redis_backend = RedisJobBackend("redis://unused")
    fake = FakeRedis()
    redis_backend._client = fake
    record = JobRecord("ttl", JobState.QUEUED, "prepared-raster", {}, retention_seconds=321)
    redis_backend.create(record)
    assert fake.expiries["sar-lra:job:ttl"] == 321
    loaded = redis_backend.get("ttl")
    assert loaded is not None and loaded.state == JobState.QUEUED
    redis_backend.enqueue("ttl")
    assert redis_backend.dequeue() == "ttl"
