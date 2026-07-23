from __future__ import annotations

import importlib
import sys


def test_raster_pipeline_import_does_not_import_earth_engine() -> None:
    sys.modules.pop("ee", None)
    sys.modules.pop("app.acquisition.earth_engine", None)
    import app.pipeline
    importlib.reload(app.pipeline)
    assert "ee" not in sys.modules
    assert "app.acquisition.earth_engine" not in sys.modules


def test_cli_exposes_both_modes() -> None:
    from app.cli import build_parser

    help_text = build_parser().format_help()
    assert "predict-raster" in help_text
    assert "predict" in help_text
