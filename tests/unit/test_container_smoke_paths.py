from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_container_smoke_test_uses_absolute_app_root_paths():
    script = (ROOT / "scripts" / "container_smoke_test.sh").read_text(encoding="utf-8")
    assert 'APP_ROOT="${SAR_LRA_APP_ROOT:-/opt/sar-lra}"' in script
    assert '--manifest "${APP_ROOT}/model/weights-manifest.json"' in script
    assert '--directory "${APP_ROOT}/model/weights"' in script


def test_weight_verifier_defaults_are_independent_of_current_working_directory():
    script = (ROOT / "scripts" / "verify_model_weights.py").read_text(encoding="utf-8")
    assert 'project_root = Path(__file__).resolve().parents[1]' in script
    assert 'project_root / "model" / "weights-manifest.json"' in script
    assert 'project_root / "model" / "weights"' in script
