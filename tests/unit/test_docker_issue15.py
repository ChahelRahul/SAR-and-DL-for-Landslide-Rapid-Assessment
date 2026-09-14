from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_is_cpu_only_pinned_and_non_root():
    text = (ROOT / "Dockerfile").read_text()
    assert "python:3.13.14-slim-bookworm@sha256:" in text
    assert "FROM --platform=linux/amd64" in text
    assert "tensorflow-cpu==2.21.0" not in text  # dependency belongs in lock file
    assert "CUDA_VISIBLE_DEVICES=-1" in text
    assert "USER 10001:10001" in text
    assert 'ENTRYPOINT ["sar-lra"]' in text
    assert 'VOLUME ["/input", "/output"]' in text


def test_cpu_dependency_file_uses_exact_pins():
    path = ROOT / "requirements" / "docker-cpu.txt"
    lines = [line.strip() for line in path.read_text().splitlines()]
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert "tensorflow-cpu==2.21.0" in requirements
    assert not any("tensorflow==" in req for req in requirements)
    assert all("==" in req for req in requirements)


def test_dockerignore_excludes_sensitive_and_large_dev_state():
    entries = set((ROOT / ".dockerignore").read_text().splitlines())
    assert ".git" in entries
    assert ".env" in entries
    assert "notebooks" in entries
    assert "tests" in entries


def test_cpu_docker_workflow_builds_and_checks_non_root():
    text = (ROOT / ".github" / "workflows" / "docker-cpu.yml").read_text()
    assert "docker build --platform linux/amd64" in text
    assert "container_smoke_test.sh" in text
    assert 'test "$uid" = "10001"' in text
