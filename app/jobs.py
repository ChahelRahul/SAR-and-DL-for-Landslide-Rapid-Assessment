from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

QUEUE_KEY = "sar-lra:jobs:queue"
PROCESSING_QUEUE_KEY = "sar-lra:jobs:processing"


class JobState(str, Enum):
    QUEUED = "queued"
    ACQUIRING = "acquiring"
    PREPROCESSING = "preprocessing"
    INFERENCING = "inferencing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    job_id: str
    state: JobState
    mode: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    attempts: int = 0
    retention_seconds: int = 86400
    timeout_seconds: int = 3600
    max_attempts: int = 3
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        if not include_payload:
            data.pop("payload", None)
        return data

    @classmethod
    def from_json(cls, raw: str) -> "JobRecord":
        data = json.loads(raw)
        data["state"] = JobState(data["state"])
        return cls(**data)

    def to_json(self) -> str:
        data = asdict(self)
        data["state"] = self.state.value
        return json.dumps(data, separators=(",", ":"), sort_keys=True)


class JobBackend(Protocol):
    def create(self, record: JobRecord) -> None: ...
    def get(self, job_id: str) -> JobRecord | None: ...
    def save(self, record: JobRecord) -> None: ...
    def enqueue(self, job_id: str) -> None: ...
    def dequeue(self, timeout: int = 5) -> str | None: ...
    def acknowledge(self, job_id: str) -> None: ...
    def queue_length(self) -> int: ...


class InMemoryJobBackend:
    """Deterministic backend for tests and local development; not multi-process safe."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[JobRecord, float]] = {}
        self.queue: list[str] = []

    def _cleanup(self) -> None:
        now = time.time()
        expired = [key for key, (_, expiry) in self.records.items() if expiry <= now]
        for key in expired:
            self.records.pop(key, None)

    def create(self, record: JobRecord) -> None:
        self._cleanup()
        if record.job_id in self.records:
            raise ValueError("job_id already exists")
        self.records[record.job_id] = (record, time.time() + record.retention_seconds)

    def get(self, job_id: str) -> JobRecord | None:
        self._cleanup()
        item = self.records.get(job_id)
        return item[0] if item else None

    def save(self, record: JobRecord) -> None:
        self.records[record.job_id] = (record, time.time() + record.retention_seconds)

    def enqueue(self, job_id: str) -> None:
        self.queue.append(job_id)

    def dequeue(self, timeout: int = 5) -> str | None:
        return self.queue.pop(0) if self.queue else None

    def acknowledge(self, job_id: str) -> None:
        return None

    def queue_length(self) -> int:
        return len(self.queue)


class RedisJobBackend:
    """Redis list + expiring job records; redis-py is imported only when used."""

    def __init__(self, url: str | None = None, *, queue_key: str = QUEUE_KEY) -> None:
        self.url = url or os.getenv("SAR_LRA_REDIS_URL", "redis://redis:6379/0")
        self.queue_key = queue_key
        self.processing_queue_key = f"{queue_key}:processing"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError("Redis support requires the 'api' extra (redis package)") from exc
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def _key(self, job_id: str) -> str:
        return f"sar-lra:job:{job_id}"

    def create(self, record: JobRecord) -> None:
        created = self.client.set(
            self._key(record.job_id), record.to_json(), nx=True, ex=record.retention_seconds
        )
        if not created:
            raise ValueError("job_id already exists")

    def get(self, job_id: str) -> JobRecord | None:
        raw = self.client.get(self._key(job_id))
        return JobRecord.from_json(raw) if raw else None

    def save(self, record: JobRecord) -> None:
        self.client.set(self._key(record.job_id), record.to_json(), ex=record.retention_seconds)

    def enqueue(self, job_id: str) -> None:
        self.client.rpush(self.queue_key, job_id)

    def dequeue(self, timeout: int = 5) -> str | None:
        # Reserve atomically so a worker crash does not silently lose the job ID.
        if hasattr(self.client, "brpoplpush"):
            return self.client.brpoplpush(self.queue_key, self.processing_queue_key, timeout=timeout)
        item = self.client.blpop(self.queue_key, timeout=timeout)
        return item[1] if item else None

    def acknowledge(self, job_id: str) -> None:
        if hasattr(self.client, "lrem"):
            self.client.lrem(self.processing_queue_key, 1, job_id)

    def queue_length(self) -> int:
        if hasattr(self.client, "llen"):
            return int(self.client.llen(self.queue_key))
        return 0

    def recover_reserved(self) -> int:
        """Requeue records left reserved by a previous crashed single-worker process.

        This is intended for the reference deployment, which runs one worker process.
        Multi-worker deployments should use an external lease/reaper policy instead.
        """
        if not hasattr(self.client, "lpop"):
            return 0
        recovered = 0
        while True:
            job_id = self.client.lpop(self.processing_queue_key)
            if not job_id:
                break
            record = self.get(job_id)
            if record is not None and record.state not in TERMINAL_STATES:
                record.state = JobState.QUEUED
                record.updated_at = utc_now()
                self.save(record)
                self.enqueue(job_id)
                recovered += 1
        return recovered


def transition(backend: JobBackend, job_id: str, state: JobState) -> JobRecord:
    record = backend.get(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.state == JobState.CANCELLED and state != JobState.CANCELLED:
        return record
    record.state = state
    record.updated_at = utc_now()
    backend.save(record)
    return record
