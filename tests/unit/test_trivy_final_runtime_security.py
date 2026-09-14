from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_final_runtime_security_pins_are_enforced_in_both_images():
    for name in ("Dockerfile", "Dockerfile.gpu"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert 'python -m pip install --upgrade --force-reinstall "wheel==0.46.2" "jaraco.context==6.1.0"' in text
        assert 'assert version("wheel") == "0.46.2"' in text
        assert 'assert version("jaraco.context") == "6.1.0"' in text


def test_runtime_locks_pin_trivy_fixed_versions():
    for rel in ("requirements/docker-cpu.txt", "requirements/docker-gpu.txt"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "wheel==0.46.2" in text
        assert "jaraco.context==6.1.0" in text
        assert "wheel==0.45.1" not in text
        assert "jaraco.context==5.3.0" not in text
