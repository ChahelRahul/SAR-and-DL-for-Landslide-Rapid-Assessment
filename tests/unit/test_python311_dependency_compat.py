from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pins(relative: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in (ROOT / relative).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '==' not in line:
            continue
        name, version = line.split('==', 1)
        result[name.lower()] = version
    return result


def test_python311_docker_locks_use_compatible_rioxarray():
    # rioxarray 0.20+ requires Python >=3.12. SAR-LRA Docker images are CPython 3.11.
    for relative in ('requirements/docker-cpu.txt', 'requirements/docker-gpu.txt'):
        pins = _pins(relative)
        assert pins['rioxarray'] == '0.19.0', f'{relative} must stay Python-3.11 compatible'
