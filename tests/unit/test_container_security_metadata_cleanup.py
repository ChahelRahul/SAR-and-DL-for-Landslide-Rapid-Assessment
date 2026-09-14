from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_cpu_image_removes_stale_vulnerable_python_metadata():
    text = _text("Dockerfile")
    assert "wheel-0.45.1.dist-info" in text
    assert "jaraco.context-5.3.0.dist-info" in text
    assert "stale vulnerable wheel metadata remains" in text
    assert "stale vulnerable jaraco.context metadata remains" in text


def test_gpu_image_removes_stale_vulnerable_python_metadata():
    text = _text("Dockerfile.gpu")
    assert "wheel-0.45.1.dist-info" in text
    assert "jaraco.context-5.3.0.dist-info" in text
    assert "stale vulnerable wheel metadata remains" in text
    assert "stale vulnerable jaraco.context metadata remains" in text
