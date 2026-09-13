from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dockerfile(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_cpu_image_marks_shell_smoke_scripts_executable():
    text = _dockerfile("Dockerfile")
    assert "COPY scripts ./scripts" in text
    assert "chmod 0755 /opt/sar-lra/scripts/*.sh" in text


def test_gpu_image_marks_shell_smoke_scripts_executable():
    text = _dockerfile("Dockerfile.gpu")
    assert "COPY scripts ./scripts" in text
    assert "chmod 0755 /opt/sar-lra/scripts/*.sh" in text


def test_container_oci_version_labels_match_release_version():
    for name in ("Dockerfile", "Dockerfile.gpu"):
        text = _dockerfile(name)
        assert 'org.opencontainers.image.version="2.0.0"' in text
        assert 'org.opencontainers.image.version="2.0.0-beta.1"' not in text
