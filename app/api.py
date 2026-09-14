from __future__ import annotations

import argparse
import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import __version__ as package_version
from app.config import AppConfig, Orbit, load_config
from app.schemas import EarthEngineRequest, PlanetaryComputerRequest, RasterInferenceRequest
from app.jobs import InMemoryJobBackend, JobBackend, JobRecord, JobState, RedisJobBackend, utc_now

_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RasterPredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_raster: str
    orbit: Literal["ASCENDING", "DESCENDING"] = "ASCENDING"
    request_id: str | None = None
    weights_path: str | None = None
    roi: dict[str, Any] | None = None
    probability_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["auto", "planetary-computer", "earth-engine"] = "auto"
    roi: dict[str, Any]
    event_date: date
    orbit: Literal["ASCENDING", "DESCENDING"] = "ASCENDING"
    request_id: str | None = None
    weights_path: str | None = None
    project: str | None = None


class AsyncJobBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["prepared-raster", "auto", "planetary-computer", "earth-engine"]
    orbit: Literal["ASCENDING", "DESCENDING"] = "ASCENDING"
    request_id: str | None = None
    input_raster: str | None = None
    roi: dict[str, Any] | None = None
    event_date: date | None = None
    weights_path: str | None = None
    project: str | None = None
    probability_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ApiSettings:
    def __init__(self) -> None:
        self.input_root = Path(os.getenv("SAR_LRA_API_INPUT_ROOT", "/input")).resolve()
        self.output_root = Path(os.getenv("SAR_LRA_API_OUTPUT_ROOT", "/output")).resolve()
        self.config_path = os.getenv("SAR_LRA_API_CONFIG")
        self.max_concurrent = int(os.getenv("SAR_LRA_API_MAX_CONCURRENT", "1"))
        if self.max_concurrent <= 0:
            raise ValueError("SAR_LRA_API_MAX_CONCURRENT must be positive")
        self.redis_url = os.getenv("SAR_LRA_REDIS_URL", "redis://redis:6379/0")
        self.job_retention_seconds = int(os.getenv("SAR_LRA_JOB_RETENTION_SECONDS", "86400"))
        if self.job_retention_seconds <= 0:
            raise ValueError("SAR_LRA_JOB_RETENTION_SECONDS must be positive")
        self.job_timeout_seconds = int(os.getenv("SAR_LRA_JOB_TIMEOUT_SECONDS", "3600"))
        if self.job_timeout_seconds <= 0:
            raise ValueError("SAR_LRA_JOB_TIMEOUT_SECONDS must be positive")
        self.job_max_attempts = int(os.getenv("SAR_LRA_JOB_MAX_ATTEMPTS", "3"))
        if self.job_max_attempts <= 0:
            raise ValueError("SAR_LRA_JOB_MAX_ATTEMPTS must be positive")
        self.max_queued_jobs = int(os.getenv("SAR_LRA_MAX_QUEUED_JOBS", "100"))
        if self.max_queued_jobs <= 0:
            raise ValueError("SAR_LRA_MAX_QUEUED_JOBS must be positive")
        self.min_free_disk_mb = int(os.getenv("SAR_LRA_MIN_FREE_DISK_MB", "1024"))
        if self.min_free_disk_mb < 0:
            raise ValueError("SAR_LRA_MIN_FREE_DISK_MB must be non-negative")


def _safe_request_id(value: str | None) -> str:
    value = value or uuid4().hex
    if not _REQUEST_ID.fullmatch(value):
        raise ValueError("request_id must be 1-128 characters using letters, numbers, '.', '_' or '-'")
    return value


def _under_root(root: Path, value: str, *, must_exist: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must resolve under {root}")
    if must_exist and not resolved.is_file():
        raise ValueError(f"file does not exist: {resolved}")
    return resolved


def _bundled_weight(orbit: Orbit) -> Path:
    candidates = []
    if os.getenv("SAR_LRA_MODEL_DIR"):
        candidates.append(Path(os.environ["SAR_LRA_MODEL_DIR"]))
    candidates.extend([
        Path(__file__).resolve().parents[1] / "model" / "weights",
        Path("/opt/sar-lra/model/weights"),
    ])
    for weight_dir in candidates:
        matches = sorted(weight_dir.glob(f"*_{orbit}_*.hdf5"))
        if len(matches) == 1:
            return matches[0]
    raise RuntimeError(f"expected exactly one bundled {orbit} weight file")


def _weights(path: str | None, orbit: Orbit, settings: ApiSettings) -> Path:
    if path is None:
        return _bundled_weight(orbit)
    # Operator-supplied weights are restricted to the mounted input tree.
    return _under_root(settings.input_root, path)


def _config(settings: ApiSettings, orbit: Orbit, threshold: float | None = None) -> AppConfig:
    config = load_config(settings.config_path).with_overrides(
        orbit=orbit,
        probability_threshold=threshold,
        output_dir=settings.output_root,
    )
    return config


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    settings: ApiSettings | None = None,
    *,
    job_backend: JobBackend | None = None,
) -> FastAPI:
    settings = settings or ApiSettings()
    job_backend = job_backend or RedisJobBackend(settings.redis_url)
    gate = threading.BoundedSemaphore(settings.max_concurrent)
    app = FastAPI(
        title="SAR-LRA API",
        version=package_version,
        description="HTTP wrapper around the SAR-LRA validated inference pipeline.",
    )
    app.state.settings = settings
    app.state.inference_gate = gate
    app.state.job_backend = job_backend

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return _error(422, "validation_error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "request_validation_error", str(exc))

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", "http_error"))
            message = str(detail.get("message", detail.get("problems", detail)))
        else:
            code, message = "http_error", str(detail)
        response = _error(exc.status_code, code, message)
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # Do not echo arbitrary exception text: it may contain credential paths or provider details.
        return _error(500, "internal_error", "inference request failed")

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "sar-lra", "version": package_version}

    @app.get("/readyz")
    def ready() -> dict[str, Any]:
        problems: list[str] = []
        try:
            settings.output_root.mkdir(parents=True, exist_ok=True)
            if not os.access(settings.output_root, os.W_OK):
                problems.append("output root is not writable")
        except OSError as exc:
            problems.append(f"output root unavailable: {exc}")
        for orbit in ("ASCENDING", "DESCENDING"):
            try:
                _bundled_weight(orbit)  # type: ignore[arg-type]
            except RuntimeError as exc:
                problems.append(str(exc))
        if problems:
            raise HTTPException(status_code=503, detail={"code": "not_ready", "problems": problems})
        return {"status": "ready", "max_concurrent": settings.max_concurrent}

    def run_bounded(func):
        if not gate.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail={"code": "busy", "message": "inference concurrency limit reached"},
                headers={"Retry-After": "5"},
            )
        try:
            return func()
        finally:
            gate.release()

    @app.post("/v1/predict-raster")
    def predict_raster(body: RasterPredictBody) -> dict[str, Any]:
        request_id = _safe_request_id(body.request_id)
        input_raster = _under_root(settings.input_root, body.input_raster)
        orbit: Orbit = body.orbit
        weights = _weights(body.weights_path, orbit, settings)
        config = _config(settings, orbit, body.probability_threshold)

        def execute() -> dict[str, Any]:
            from app.pipeline import run_raster
            result = run_raster(
                RasterInferenceRequest(
                    request_id=request_id,
                    orbit=orbit,
                    weights_path=weights,
                    input_raster=input_raster,
                    roi_geojson=body.roi,
                ),
                config,
            )
            return result.to_dict()

        return run_bounded(execute)

    @app.post("/v1/predict")
    def predict(body: PredictBody) -> dict[str, Any]:
        request_id = _safe_request_id(body.request_id)
        orbit: Orbit = body.orbit
        weights = _weights(body.weights_path, orbit, settings)
        config = _config(settings, orbit)

        def execute() -> dict[str, Any]:
            from app.acquisition.providers import resolve_provider
            provider = resolve_provider(body.provider)
            if provider == "planetary-computer":
                from app.pipeline import run_planetary_computer
                result = run_planetary_computer(
                    PlanetaryComputerRequest(
                        request_id=request_id, orbit=orbit, event_date=body.event_date,
                        weights_path=weights, roi_geojson=body.roi,
                        cache_dir=settings.output_root / ".cache",
                    ),
                    config,
                )
            else:
                from app.pipeline import run_earth_engine
                result = run_earth_engine(
                    EarthEngineRequest(
                        request_id=request_id, orbit=orbit, event_date=body.event_date,
                        weights_path=weights, roi_geojson=body.roi, project=body.project,
                        authenticate=False, cache_dir=settings.output_root / ".cache",
                    ),
                    config,
                )
            return result.to_dict()

        return run_bounded(execute)


    @app.post("/v1/jobs", status_code=202)
    def create_job(body: AsyncJobBody) -> dict[str, Any]:
        job_id = _safe_request_id(body.request_id)
        orbit: Orbit = body.orbit
        # Validate and normalize all filesystem inputs before queueing.
        payload: dict[str, Any] = {
            "orbit": orbit,
            "weights_path": body.weights_path,
            "probability_threshold": body.probability_threshold,
        }
        if body.weights_path is not None:
            payload["weights_path"] = str(_weights(body.weights_path, orbit, settings))
        mode = body.source
        if body.source == "prepared-raster":
            if body.input_raster is None:
                raise ValueError("input_raster is required for prepared-raster jobs")
            payload["input_raster"] = str(_under_root(settings.input_root, body.input_raster))
            payload["roi"] = body.roi
        else:
            if body.roi is None or body.event_date is None:
                raise ValueError("roi and event_date are required for remote acquisition jobs")
            from app.acquisition.providers import resolve_provider
            provider = resolve_provider(body.source)
            payload.update({
                "roi": body.roi,
                "event_date": body.event_date.isoformat(),
                "project": body.project,
                "provider": provider,
            })
            mode = provider
        if job_backend.queue_length() >= settings.max_queued_jobs:
            raise HTTPException(
                status_code=429,
                detail={"code": "queue_full", "message": "asynchronous job queue is full"},
                headers={"Retry-After": "30"},
            )
        record = JobRecord(
            job_id=job_id,
            state=JobState.QUEUED,
            mode=mode,
            payload=payload,
            retention_seconds=settings.job_retention_seconds,
            timeout_seconds=settings.job_timeout_seconds,
            max_attempts=settings.job_max_attempts,
        )
        job_backend.create(record)
        job_backend.enqueue(job_id)
        return {
            "job_id": job_id,
            "state": JobState.QUEUED.value,
            "status_url": f"/v1/jobs/{job_id}",
            "result_url": f"/v1/jobs/{job_id}/result",
        }

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        _safe_request_id(job_id)
        record = job_backend.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "job not found or expired"})
        return record.to_dict()

    @app.get("/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict[str, Any]:
        _safe_request_id(job_id)
        record = job_backend.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "job not found or expired"})
        if record.state == JobState.FAILED:
            return {"job_id": job_id, "state": record.state.value, "error": record.error}
        if record.state != JobState.COMPLETED:
            raise HTTPException(status_code=409, detail={"code": "job_not_complete", "message": f"job is {record.state.value}"})
        return record.result or {"job_id": job_id, "state": record.state.value}

    @app.delete("/v1/jobs/{job_id}", status_code=202)
    def cancel_job(job_id: str) -> dict[str, Any]:
        _safe_request_id(job_id)
        record = job_backend.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "job not found or expired"})
        if record.state == JobState.COMPLETED:
            return {"job_id": job_id, "state": record.state.value}
        record.state = JobState.CANCELLED
        record.updated_at = utc_now()
        job_backend.save(record)
        return {"job_id": job_id, "state": record.state.value}

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> dict[str, Any]:
        _safe_request_id(job_id)
        record = job_backend.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "job not found or expired"})
        # Completed jobs are deliberately not re-enqueued: retries are idempotent.
        if record.state == JobState.COMPLETED:
            return {"job_id": job_id, "state": record.state.value, "requeued": False}
        if record.state not in {JobState.FAILED, JobState.CANCELLED}:
            raise HTTPException(status_code=409, detail={"code": "job_not_retryable", "message": f"job is {record.state.value}"})
        if record.attempts >= record.max_attempts:
            raise HTTPException(status_code=409, detail={"code": "attempt_limit_reached", "message": "job retry limit reached"})
        if job_backend.queue_length() >= settings.max_queued_jobs:
            raise HTTPException(status_code=429, detail={"code": "queue_full", "message": "asynchronous job queue is full"}, headers={"Retry-After": "30"})
        record.state = JobState.QUEUED
        record.error = None
        record.updated_at = utc_now()
        job_backend.save(record)
        job_backend.enqueue(job_id)
        return {"job_id": job_id, "state": record.state.value, "requeued": True}

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SAR-LRA FastAPI service")
    parser.add_argument("--host", default=os.getenv("SAR_LRA_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SAR_LRA_API_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("SAR_LRA_API_WORKERS", "1")))
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    import uvicorn
    uvicorn.run("app.api:app", host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
