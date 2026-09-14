from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_dockerfile_is_separate_pinned_non_root_image():
    text = (ROOT / "Dockerfile.gpu").read_text()
    assert "python:3.13.14-slim-bookworm@sha256:" in text
    assert "FROM --platform=linux/amd64" in text
    assert "requirements/docker-gpu.txt" in text
    assert "CUDA_VISIBLE_DEVICES=-1" not in text
    assert "NVIDIA_VISIBLE_DEVICES=all" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in text
    assert "USER 10001:10001" in text
    assert 'ENTRYPOINT ["sar-lra"]' in text


def test_gpu_dependency_file_uses_tensorflow_cuda_extra_and_exact_pins():
    lines = (ROOT / "requirements/docker-gpu.txt").read_text().splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert "tensorflow[and-cuda]==2.21.0" in requirements
    assert "tensorflow-cpu==2.21.0" not in requirements
    assert all("==" in req for req in requirements)


def test_gpu_runtime_smoke_requires_real_gpu_and_checks_cpu_gpu_parity():
    text = (ROOT / "scripts/container_gpu_smoke_test.py").read_text()
    assert 'list_physical_devices("GPU")' in text
    assert 'tf.device("/GPU:0")' in text
    assert 'tf.device("/CPU:0")' in text
    assert "assert_allclose" in text
    assert "2e-5" in text


def test_gpu_workflow_builds_without_gpu_and_has_optional_real_gpu_job():
    text = (ROOT / ".github/workflows/docker-gpu.yml").read_text()
    assert "docker build --platform linux/amd64 --file Dockerfile.gpu" in text
    assert "container_gpu_static_smoke_test.sh" in text
    assert "SAR_LRA_GPU_CI" in text
    assert "self-hosted, linux, x64, gpu" in text
    assert "docker run --rm --gpus all" in text
    assert "container_gpu_smoke_test.py" in text


def test_cpu_image_contract_is_unchanged_by_gpu_variant():
    cpu = (ROOT / "Dockerfile").read_text()
    cpu_req = (ROOT / "requirements/docker-cpu.txt").read_text()
    assert "CUDA_VISIBLE_DEVICES=-1" in cpu
    assert "tensorflow-cpu==2.21.0" in cpu_req
    assert "tensorflow[and-cuda]" not in cpu_req
