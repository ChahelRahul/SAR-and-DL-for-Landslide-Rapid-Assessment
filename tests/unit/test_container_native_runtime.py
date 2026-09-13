from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dockerfile(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_cpu_image_installs_expat_runtime_for_rasterio():
    text = _dockerfile("Dockerfile")
    assert "apt-get install --yes --no-install-recommends libexpat1" in text
    assert "rm -rf /var/lib/apt/lists/*" in text


def test_gpu_image_installs_expat_runtime_for_rasterio():
    text = _dockerfile("Dockerfile.gpu")
    assert "apt-get install --yes --no-install-recommends libexpat1" in text
    assert "rm -rf /var/lib/apt/lists/*" in text
