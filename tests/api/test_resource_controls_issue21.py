from __future__ import annotations

import contextlib
import signal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import ApiSettings, create_app
from app.jobs import InMemoryJobBackend, JobRecord, JobState, RedisJobBackend
from app.worker import JobTimedOut, _hard_timeout, execute_job


def settings(tmp_path: Path) -> ApiSettings:
    s = ApiSettings()
    s.input_root = (tmp_path / "input").resolve()
    s.output_root = (tmp_path / "output").resolve()
    s.input_root.mkdir()
    s.output_root.mkdir()
    s.job_retention_seconds = 60
    s.job_timeout_seconds = 30
    s.job_max_attempts = 3
    s.max_queued_jobs = 10
    s.min_free_disk_mb = 0
    return s


def test_queue_backpressure_returns_429(tmp_path):
    s = settings(tmp_path)
    s.max_queued_jobs = 1
    scene = s.input_root / "scene.tif"
    scene.write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    backend.enqueue("already-waiting")
    response = TestClient(create_app(s, job_backend=backend)).post(
        "/v1/jobs",
        json={"source": "prepared-raster", "input_raster": "scene.tif", "orbit": "ASCENDING"},
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "queue_full"
    assert response.headers["retry-after"] == "30"


def test_retry_limit_is_enforced(tmp_path):
    s = settings(tmp_path)
    backend = InMemoryJobBackend()
    backend.create(
        JobRecord(
            "failed",
            JobState.FAILED,
            "prepared-raster",
            {},
            attempts=3,
            max_attempts=3,
        )
    )
    response = TestClient(create_app(s, job_backend=backend)).post("/v1/jobs/failed/retry")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "attempt_limit_reached"


def test_cancel_during_progress_cleans_partial_outputs(tmp_path, monkeypatch):
    s = settings(tmp_path)
    scene = s.input_root / "scene.tif"
    scene.write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    backend.create(
        JobRecord(
            "cancel-running",
            JobState.QUEUED,
            "prepared-raster",
            {"orbit": "ASCENDING", "input_raster": str(scene), "weights_path": None},
            timeout_seconds=30,
        )
    )
    monkeypatch.setattr("app.worker._weights", lambda *a, **k: Path("weights.hdf5"))

    def fake_run(request, config, *, progress=None):
        out = s.output_root / request.request_id
        out.mkdir(parents=True)
        (out / "partial.tif").write_bytes(b"partial")
        rec = backend.get(request.request_id)
        rec.state = JobState.CANCELLED
        backend.save(rec)
        progress("inferencing")
        raise AssertionError("cancel checkpoint should interrupt")

    monkeypatch.setattr("app.pipeline.run_raster", fake_run)
    assert execute_job("cancel-running", backend, s) is None
    assert backend.get("cancel-running").state == JobState.CANCELLED
    assert not (s.output_root / "cancel-running").exists()


def test_timeout_marks_job_failed_and_cleans_output(tmp_path, monkeypatch):
    s = settings(tmp_path)
    scene = s.input_root / "scene.tif"
    scene.write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    backend.create(
        JobRecord(
            "timeout-job",
            JobState.QUEUED,
            "prepared-raster",
            {"orbit": "ASCENDING", "input_raster": str(scene), "weights_path": None},
            timeout_seconds=1,
        )
    )
    monkeypatch.setattr("app.worker._weights", lambda *a, **k: Path("weights.hdf5"))

    @contextlib.contextmanager
    def immediate_timeout(_seconds):
        raise JobTimedOut("expired")
        yield

    monkeypatch.setattr("app.worker._hard_timeout", immediate_timeout)
    (s.output_root / "timeout-job").mkdir()
    (s.output_root / "timeout-job" / "partial.tif").write_bytes(b"partial")
    assert execute_job("timeout-job", backend, s) is None
    record = backend.get("timeout-job")
    assert record.state == JobState.FAILED
    assert record.error["code"] == "job_timeout"
    assert not (s.output_root / "timeout-job").exists()


def test_linux_hard_timeout_raises_job_timeout():
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "raise_signal"):
        pytest.skip("SIGALRM unavailable")
    with pytest.raises(JobTimedOut):
        with _hard_timeout(30):
            signal.raise_signal(signal.SIGALRM)


def test_worker_rejects_when_disk_floor_not_met(tmp_path, monkeypatch):
    s = settings(tmp_path)
    s.min_free_disk_mb = 100
    scene = s.input_root / "scene.tif"
    scene.write_bytes(b"fixture")
    backend = InMemoryJobBackend()
    backend.create(JobRecord("disk", JobState.QUEUED, "prepared-raster", {"orbit": "ASCENDING", "input_raster": str(scene)}))

    class Usage:
        total = 1000
        used = 999
        free = 10 * 1024 * 1024

    monkeypatch.setattr("app.worker.shutil.disk_usage", lambda _: Usage())
    assert execute_job("disk", backend, s) is None
    assert backend.get("disk").error["code"] == "insufficient_storage"


def test_redis_reserved_job_recovery_without_live_redis():
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.queues = {}
        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True
        def get(self, key):
            return self.values.get(key)
        def rpush(self, key, value):
            self.queues.setdefault(key, []).append(value)
        def lpop(self, key):
            q = self.queues.setdefault(key, [])
            return q.pop(0) if q else None
        def llen(self, key):
            return len(self.queues.setdefault(key, []))

    backend = RedisJobBackend("redis://unused")
    fake = FakeRedis()
    backend._client = fake
    backend.create(JobRecord("orphan", JobState.INFERENCING, "prepared-raster", {}))
    fake.rpush(backend.processing_queue_key, "orphan")
    assert backend.recover_reserved() == 1
    assert backend.get("orphan").state == JobState.QUEUED
    assert fake.queues[backend.queue_key] == ["orphan"]
