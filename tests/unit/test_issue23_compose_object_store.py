from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.object_store import ObjectStoreSettings, mirror_job_output


class FakeS3:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(self, filename: str, bucket: str, key: str):
        self.uploads.append((filename, bucket, key))
        return None


def test_complete_compose_has_required_services_and_persistence():
    doc = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = doc["services"]
    assert {"api", "worker", "redis", "minio", "minio-init", "cli"} <= set(services)
    assert {"redis-data", "minio-data", "sar-lra-results"} <= set(doc["volumes"])
    assert services["redis"]["healthcheck"]
    assert services["minio"]["healthcheck"]
    assert services["api"]["healthcheck"]
    assert services["worker"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert services["worker"]["depends_on"]["minio-init"]["condition"] == "service_completed_successfully"
    init_script = "\n".join(services["minio-init"]["entrypoint"])
    assert "mc anonymous set none" in init_script


def test_cli_profile_is_queue_and_object_store_independent():
    doc = yaml.safe_load(Path("docker-compose.yml").read_text())
    cli = doc["services"]["cli"]
    assert cli["profiles"] == ["cli"]
    assert "depends_on" not in cli
    assert cli["entrypoint"] == ["sar-lra"]


def test_object_store_disabled_is_lazy_and_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("SAR_LRA_S3_BUCKET", raising=False)
    sys.modules.pop("boto3", None)
    settings = ObjectStoreSettings.from_env()
    assert not settings.enabled
    assert mirror_job_output("job-1", tmp_path, settings) is None
    assert "boto3" not in sys.modules


def test_object_store_mirrors_nested_job_files(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    nested = tmp_path / "vectors"
    nested.mkdir()
    (nested / "detections.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    fake = FakeS3()
    settings = ObjectStoreSettings(
        endpoint_url="http://minio:9000",
        bucket="sar-lra-results",
        access_key="test",
        secret_key="secret",
        region="us-east-1",
        prefix="jobs",
    )
    manifest = mirror_job_output("event-001", tmp_path, settings, client=fake)
    assert manifest is not None
    assert manifest["bucket"] == "sar-lra-results"
    assert manifest["prefix"] == "jobs/event-001"
    assert manifest["object_count"] == 2
    keys = [item[2] for item in fake.uploads]
    assert keys == ["jobs/event-001/result.json", "jobs/event-001/vectors/detections.geojson"]
    assert all(obj["uri"].startswith("s3://sar-lra-results/jobs/event-001/") for obj in manifest["objects"])
    assert "secret" not in str(manifest)
