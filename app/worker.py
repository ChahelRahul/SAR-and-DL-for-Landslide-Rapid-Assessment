from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from app.api import ApiSettings, _config, _weights
from app.jobs import JobBackend, JobState, RedisJobBackend, utc_now
from app.object_store import ObjectStoreSettings, mirror_job_output
from app.schemas import EarthEngineRequest, PlanetaryComputerRequest, RasterInferenceRequest


class JobCancelled(RuntimeError):
    pass


class JobTimedOut(TimeoutError):
    pass


def _safe_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, JobTimedOut):
        return {"code": "job_timeout", "message": "job exceeded its execution timeout"}
    if isinstance(exc, ValueError):
        return {"code": "validation_error", "message": "job validation failed"}
    if "insufficient free disk space" in str(exc):
        return {"code": "insufficient_storage", "message": "worker has insufficient free disk space"}
    return {"code": "processing_failed", "message": "job processing failed"}


def _job_output_dir(settings: ApiSettings, job_id: str) -> Path:
    root = settings.output_root.resolve()
    path = (root / job_id).resolve()
    if path != root and root not in path.parents:
        raise ValueError("job output path escaped output root")
    return path


def _cleanup_job_output(settings: ApiSettings, job_id: str) -> None:
    path = _job_output_dir(settings, job_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def _hard_timeout(seconds: int) -> Iterator[None]:
    """Best-effort hard timeout for the Linux worker process.

    The production containers are Linux. On platforms without SIGALRM, cooperative
    deadline checks still run at pipeline progress boundaries.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum, _frame):
        raise JobTimedOut("job execution timeout exceeded")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def execute_job(job_id: str, backend: JobBackend, settings: ApiSettings) -> dict[str, Any] | None:
    record = backend.get(job_id)
    if record is None:
        backend.acknowledge(job_id)
        return None
    # Duplicate delivery / retry is idempotent for terminal success and explicit cancellation.
    if record.state in {JobState.COMPLETED, JobState.CANCELLED}:
        backend.acknowledge(job_id)
        if record.state == JobState.CANCELLED:
            _cleanup_job_output(settings, job_id)
        return record.result
    if record.attempts >= record.max_attempts:
        record.state = JobState.FAILED
        record.error = {"code": "attempt_limit_reached", "message": "job retry limit reached"}
        record.finished_at = utc_now()
        record.updated_at = record.finished_at
        backend.save(record)
        backend.acknowledge(job_id)
        _cleanup_job_output(settings, job_id)
        return None

    record.attempts += 1
    record.error = None
    record.started_at = record.started_at or utc_now()
    record.heartbeat_at = utc_now()
    record.updated_at = record.heartbeat_at
    backend.save(record)
    payload = record.payload
    started_monotonic = time.monotonic()

    def checkpoint(stage: str | None = None) -> None:
        latest = backend.get(job_id)
        if latest is None:
            raise JobCancelled("job record expired or was removed")
        if latest.state == JobState.CANCELLED:
            raise JobCancelled("job cancelled")
        if time.monotonic() - started_monotonic > latest.timeout_seconds:
            raise JobTimedOut("job execution timeout exceeded")
        mapping = {
            "acquiring": JobState.ACQUIRING,
            "preprocessing": JobState.PREPROCESSING,
            "inferencing": JobState.INFERENCING,
            "postprocessing": JobState.POSTPROCESSING,
        }
        if stage in mapping:
            latest.state = mapping[stage]
        latest.heartbeat_at = utc_now()
        latest.updated_at = latest.heartbeat_at
        backend.save(latest)

    try:
        with _hard_timeout(record.timeout_seconds):
            checkpoint()
            settings.output_root.mkdir(parents=True, exist_ok=True)
            free_mb = shutil.disk_usage(settings.output_root).free // (1024 * 1024)
            if free_mb < settings.min_free_disk_mb:
                raise RuntimeError("insufficient free disk space for job execution")
            orbit = payload["orbit"]
            weights = _weights(payload.get("weights_path"), orbit, settings)
            config = _config(settings, orbit, payload.get("probability_threshold"))
            if record.mode == "prepared-raster":
                from app.pipeline import run_raster
                result = run_raster(
                    RasterInferenceRequest(
                        request_id=job_id,
                        orbit=orbit,
                        weights_path=weights,
                        input_raster=Path(payload["input_raster"]),
                        roi_geojson=payload.get("roi"),
                    ),
                    config,
                    progress=checkpoint,
                )
            elif record.mode in {"auto", "planetary-computer", "earth-engine"}:
                from app.acquisition.providers import resolve_provider
                provider = resolve_provider(payload.get("provider") or record.mode)
                if provider == "planetary-computer":
                    from app.pipeline import run_planetary_computer
                    result = run_planetary_computer(
                        PlanetaryComputerRequest(
                            request_id=job_id, orbit=orbit, event_date=date.fromisoformat(payload["event_date"]),
                            weights_path=weights, roi_geojson=payload["roi"],
                            cache_dir=settings.output_root / ".cache",
                        ),
                        config, progress=checkpoint,
                    )
                else:
                    from app.pipeline import run_earth_engine
                    result = run_earth_engine(
                        EarthEngineRequest(
                            request_id=job_id,
                            orbit=orbit,
                            event_date=date.fromisoformat(payload["event_date"]),
                            weights_path=weights,
                            roi_geojson=payload["roi"],
                            project=payload.get("project"),
                            authenticate=False,
                            cache_dir=settings.output_root / ".cache",
                        ),
                        config,
                        progress=checkpoint,
                    )
            else:
                raise ValueError(f"unknown job mode: {record.mode}")
            checkpoint()

        latest = backend.get(job_id)
        if latest is None or latest.state == JobState.CANCELLED:
            _cleanup_job_output(settings, job_id)
            return None
        result_dict = result.to_dict()
        object_store = mirror_job_output(
            job_id,
            _job_output_dir(settings, job_id),
            ObjectStoreSettings.from_env(),
        )
        if object_store is not None:
            result_dict["object_store"] = object_store
        latest.state = JobState.COMPLETED
        latest.result = result_dict
        latest.error = None
        latest.finished_at = utc_now()
        latest.heartbeat_at = latest.finished_at
        latest.updated_at = latest.finished_at
        backend.save(latest)
        return latest.result
    except JobCancelled:
        latest = backend.get(job_id)
        if latest is not None:
            latest.state = JobState.CANCELLED
            latest.error = None
            latest.finished_at = utc_now()
            latest.updated_at = latest.finished_at
            backend.save(latest)
        _cleanup_job_output(settings, job_id)
        return None
    except Exception as exc:
        latest = backend.get(job_id)
        if latest is not None and latest.state != JobState.CANCELLED:
            latest.state = JobState.FAILED
            latest.error = _safe_error(exc)
            latest.finished_at = utc_now()
            latest.updated_at = latest.finished_at
            backend.save(latest)
        _cleanup_job_output(settings, job_id)
        # Keep detailed diagnostics in worker stderr/log collector, never in API record.
        traceback.print_exc()
        return None
    finally:
        backend.acknowledge(job_id)


def worker_loop(backend: JobBackend, settings: ApiSettings, *, once: bool = False) -> int:
    # Reference deployment runs one worker process. Recover reservations left by a
    # previous crash before consuming new work.
    recover = getattr(backend, "recover_reserved", None)
    if callable(recover):
        recover()
    while True:
        job_id = backend.dequeue(timeout=1 if once else 5)
        if job_id:
            execute_job(job_id, backend, settings)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SAR-LRA Redis job worker")
    parser.add_argument("--redis-url", default=os.getenv("SAR_LRA_REDIS_URL", "redis://redis:6379/0"))
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit")
    args = parser.parse_args(argv)
    return worker_loop(RedisJobBackend(args.redis_url), ApiSettings(), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
