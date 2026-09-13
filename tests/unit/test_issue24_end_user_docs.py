from pathlib import Path

import json

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "end-user-deployment.md"
ISSUE = ROOT / "docs" / "issue-24-end-user-deployment.md"
SAMPLE = ROOT / "examples" / "quickstart"


def test_end_user_guide_covers_issue24_topics():
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "CLI-only installation",
        "Prepared-raster mode",
        "Earth Engine mode",
        "Earth Engine credentials",
        "Docker",
        "Podman",
        "CPU and GPU expectations",
        "ROI and date examples",
        "HTTP API deployment",
        "Complete Compose deployment",
        "Output interpretation",
        "Troubleshooting",
        "Scientific limitations",
        "Citation and acknowledgement",
    )
    for heading in required:
        assert heading in text
    assert "not authoritative landslide inventories" in text
    assert "not a calibrated per-pixel landslide probability" in text


def test_notebook_free_quickstart_is_embedded_by_container_build():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY examples ./examples" in dockerfile
    assert (SAMPLE / "ascending-4band.tif").is_file()
    assert (SAMPLE / "roi.geojson").is_file()
    assert (SAMPLE / "README.md").is_file()
    guide = GUIDE.read_text(encoding="utf-8")
    assert "/opt/sar-lra/examples/quickstart/ascending-4band.tif" in guide
    assert "/opt/sar-lra/examples/quickstart/roi.geojson" in guide
    assert "predict-raster" in guide
    assert "notebook" in guide.lower()


def test_quickstart_roi_is_valid_geojson_shape():
    doc = json.loads((SAMPLE / "roi.geojson").read_text(encoding="utf-8"))
    assert doc["type"] in {"Feature", "FeatureCollection", "Polygon", "MultiPolygon"}


def test_issue24_acceptance_and_registry_caveat_are_documented():
    text = ISSUE.read_text(encoding="utf-8")
    assert "Acceptance path" in text
    assert "without cloning the repository" in text
    assert "cannot prove" in text
    assert "Issue 25" in text


def test_readme_links_primary_end_user_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/end-user-deployment.md" in readme
