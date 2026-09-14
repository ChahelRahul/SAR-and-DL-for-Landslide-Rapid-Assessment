from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "publish-images.yml"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_workflow_exists_and_parses():
    data = yaml.safe_load(_text())
    assert data["name"] == "publish-images"
    assert "publish-cpu" in data["jobs"]
    assert "publish-gpu" in data["jobs"]


def test_release_workflow_has_required_supply_chain_controls():
    text = _text()
    for required in (
        "docker/build-push-action@v6",
        "anchore/sbom-action@v0.24.0",
        "aquasecurity/trivy-action@v0.36.0",
        "sigstore/cosign-installer@v4.1.2",
        "cosign sign --yes",
        "id-token: write",
        "packages: write",
        "provenance: mode=max",
    ):
        assert required in text


def test_release_workflow_defines_semver_latest_and_sha_tags():
    text = _text()
    assert "type=semver,pattern={{version}}" in text
    assert "type=semver,pattern={{major}}.{{minor}}" in text
    assert "type=semver,pattern={{major}}" in text
    assert "type=raw,value=latest" in text
    assert "type=sha,prefix=sha-" in text


def test_gpu_is_release_tag_only():
    data = yaml.safe_load(_text())
    gpu = data["jobs"]["publish-gpu"]
    assert "refs/tags/v" in gpu["if"]
    assert "Dockerfile.gpu" in _text()


def test_semantic_release_tags_have_overwrite_guards():
    text = _text()
    assert text.count("docker buildx imagetools inspect") == 2
    assert "Refusing to overwrite immutable release tag" in text


def test_registry_names_match_public_contract():
    text = _text()
    assert "CPU_IMAGE: ghcr.io/${{ github.repository_owner }}/sar-lra" in text
    assert "GPU_IMAGE: ghcr.io/${{ github.repository_owner }}/sar-lra-gpu" in text
