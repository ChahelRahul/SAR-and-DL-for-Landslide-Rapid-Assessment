"""Legacy filename retained for checkout/overlay compatibility.

SAR-LRA moved from Python 3.11 to Python 3.13. This file intentionally
validates the current Python 3.13 Docker dependency locks so older checkouts
that still track this path do not enforce obsolete 3.11-only pins.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pins(relative: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in (ROOT / relative).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()
    return pins


def test_legacy_python311_check_tracks_python313_rioxarray_lock():
    """The project is Python 3.13; rioxarray 0.23.0 is the intended lock."""
    for relative in ("requirements/docker-cpu.txt", "requirements/docker-gpu.txt"):
        pins = _pins(relative)
        assert pins["rioxarray"] == "0.23.0", f"{relative} must match the Python-3.13 lock"
