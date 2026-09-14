from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pins(relative: str) -> dict[str, str]:
    pins = {}
    for raw in (ROOT / relative).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '==' not in line:
            continue
        name, version = line.split('==', 1)
        pins[name] = version
    return pins


def test_python313_runtime_and_dependency_pins():
    for relative in ('requirements/docker-cpu.txt', 'requirements/docker-gpu.txt'):
        pins = _pins(relative)
        assert pins['numpy'] == '2.2.6'
        assert pins['rioxarray'] == '0.23.0'
        tf_key = 'tensorflow-cpu' if relative.endswith('cpu.txt') else 'tensorflow[and-cuda]'
        assert pins[tf_key] == '2.21.0'


def test_dockerfiles_use_python313():
    expected = 'python:3.13.14-slim-bookworm@sha256:dd86541a59b252667f4c12f8b2ee17216de37dd65ac773bf097bef996fa78860'
    for relative in ('Dockerfile', 'Dockerfile.gpu'):
        text = (ROOT / relative).read_text()
        assert expected in text
        assert '/usr/local/lib/python3.13/site-packages' in text


def test_project_requires_python313():
    text = (ROOT / 'pyproject.toml').read_text()
    assert 'requires-python = ">=3.13,<3.14"' in text
