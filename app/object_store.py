from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ObjectStoreSettings:
    endpoint_url: str | None
    bucket: str | None
    access_key: str | None
    secret_key: str | None
    region: str
    prefix: str

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    @classmethod
    def from_env(cls) -> "ObjectStoreSettings":
        return cls(
            endpoint_url=os.getenv("SAR_LRA_S3_ENDPOINT_URL") or None,
            bucket=os.getenv("SAR_LRA_S3_BUCKET") or None,
            access_key=os.getenv("AWS_ACCESS_KEY_ID") or None,
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            prefix=os.getenv("SAR_LRA_S3_PREFIX", "jobs").strip("/"),
        )


def _client(settings: ObjectStoreSettings):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("S3 object storage requires boto3 (install the 'api' extra)") from exc
    kwargs: dict[str, Any] = {"region_name": settings.region}
    if settings.endpoint_url:
        kwargs["endpoint_url"] = settings.endpoint_url
    if settings.access_key:
        kwargs["aws_access_key_id"] = settings.access_key
    if settings.secret_key:
        kwargs["aws_secret_access_key"] = settings.secret_key
    return boto3.client("s3", **kwargs)


def mirror_job_output(
    job_id: str,
    output_dir: Path,
    settings: ObjectStoreSettings | None = None,
    *,
    client: Any | None = None,
) -> dict[str, Any] | None:
    """Mirror one completed job directory to an S3-compatible object store.

    The local filesystem remains the worker's active workspace. When S3/MinIO is
    configured, completed files are copied under ``<prefix>/<job_id>/...`` and a
    portable object manifest is returned for inclusion in the job result.
    """
    settings = settings or ObjectStoreSettings.from_env()
    if not settings.enabled:
        return None
    if not settings.bucket:
        return None
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise RuntimeError(f"job output directory does not exist: {output_dir}")
    s3 = client or _client(settings)
    objects: list[dict[str, Any]] = []
    root_prefix = "/".join(part for part in (settings.prefix, job_id) if part)
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        key = f"{root_prefix}/{relative}" if root_prefix else relative
        response = s3.upload_file(str(path), settings.bucket, key)
        # boto3 upload_file normally returns None; test doubles may return metadata.
        item: dict[str, Any] = {
            "key": key,
            "uri": f"s3://{settings.bucket}/{key}",
            "size_bytes": path.stat().st_size,
        }
        if isinstance(response, dict) and response.get("ETag"):
            item["etag"] = str(response["ETag"]).strip('"')
        objects.append(item)
    return {
        "provider": "s3",
        "bucket": settings.bucket,
        "prefix": root_prefix,
        "object_count": len(objects),
        "objects": objects,
    }
