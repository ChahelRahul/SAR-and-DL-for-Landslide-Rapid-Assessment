from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_software_release_version_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    runtime = (ROOT / "app" / "__init__.py").read_text()
    assert re.search(r'^version = "2\.0\.0"$', pyproject, re.M)
    assert re.search(r'^__version__ = "2\.0\.0"$', runtime, re.M)
    assert "Development Status :: 5 - Production/Stable" in pyproject


def test_model_version_remains_distinct_from_software_release() -> None:
    manifest = json.loads((ROOT / "model" / "weights-manifest.json").read_text())
    assert manifest["model_version"] == "sar-lra-v2.0.0-beta.1"
    assert manifest["license"]["weights"] == "UNCONFIRMED"


def test_release_notes_and_changelog_exist() -> None:
    assert (ROOT / "CHANGELOG.md").is_file()
    notes = (ROOT / "RELEASE_NOTES_v2.0.0.md").read_text()
    assert "SAR-LRA v2.0.0" in notes
    assert "not a calibrated per-pixel landslide probability" in notes
    assert "UNCONFIRMED" in notes


def test_release_workflow_creates_v2_release_after_images() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "publish-images.yml"
    raw = workflow_path.read_text()
    data = yaml.safe_load(raw)
    # PyYAML treats unquoted `on` as a YAML 1.1 boolean in some versions, so
    # inspect the raw text for the trigger and parsed jobs for release details.
    assert "tags: ['v[0-9]+.[0-9]+.[0-9]+']" in raw
    jobs = data["jobs"]
    release = jobs["create-release-v2"]
    assert release["if"] == "github.ref == 'refs/tags/v2.0.0'"
    assert set(release["needs"]) == {"test", "publish-cpu", "publish-gpu"}
    steps = "\n".join(str(step) for step in release["steps"])
    assert "RELEASE_NOTES_v2.0.0.md" in steps
    assert "SHA256SUMS" in steps
    assert "sar-lra-cpu-sbom" in steps
    assert "sar-lra-gpu-sbom" in steps
    assert "gh release create" in raw


def test_issue25_docs_use_v2_release_tag() -> None:
    docs = (ROOT / "docs" / "issue-25-first-versioned-release.md").read_text()
    assert "git tag -a v2.0.0" in docs
    assert "ghcr.io/chahelrahul/sar-lra:2.0.0" in docs
