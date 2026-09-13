from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RASTERS = (
    ROOT / "tests/fixtures/reference/haiti-2021-ascending.tif",
    ROOT / "tests/fixtures/reference/sumatra-2022-descending.tif",
    ROOT / "examples/quickstart/ascending-4band.tif",
)


def test_required_binary_fixtures_are_present():
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_RASTERS if not path.is_file()]
    assert not missing, f"required tracked raster fixtures are missing: {missing}"


def test_required_binary_fixtures_are_explicitly_unignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!tests/fixtures/reference/*.tif" in gitignore
    assert "!examples/quickstart/ascending-4band.tif" in gitignore


def test_docker_earth_engine_pin_is_installable_release_line():
    for relative in ("requirements/docker-cpu.txt", "requirements/docker-gpu.txt"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "earthengine-api==1.6.6" in text
        assert "earthengine-api==1.6.5" not in text
