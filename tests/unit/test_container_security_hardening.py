from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_container_images_apply_debian_security_updates():
    for name in ("Dockerfile", "Dockerfile.gpu"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "apt-get update" in text
        assert "apt-get upgrade --yes" in text
        assert "rm -rf /var/lib/apt/lists/*" in text


def test_container_python_security_pins():
    for relative in ("requirements/docker-cpu.txt", "requirements/docker-gpu.txt"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "geopandas==1.1.4" in text
        assert "fastapi==0.141.1" in text
        assert "starlette==1.6.0" in text
        assert "geopandas==1.0.1" not in text
        assert "fastapi==0.128.2" not in text


def test_container_build_tooling_is_patched():
    for name in ("Dockerfile", "Dockerfile.gpu"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert '"setuptools==80.9.0"' in text
        assert '"wheel==0.46.2"' in text
