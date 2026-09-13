from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app.config import load_config
from app.roi import RoiValidationError, validate_event_date, validate_roi


def polygon(coords):
    return {"type": "Polygon", "coordinates": [coords]}


def limits(**overrides):
    values = dict(max_area_km2=10_000, max_width_km=500, max_height_km=500, max_vertices=50_000)
    values.update(overrides)
    return values


def test_reports_geodesic_area_and_dimensions():
    report = validate_roi(
        polygon([[-72, 18], [-71.9, 18], [-71.9, 18.1], [-72, 18.1], [-72, 18]]),
        **limits(),
    )
    assert 100 < report.area_km2 < 130
    assert 10 < report.bbox_width_km < 12
    assert 10 < report.bbox_height_km < 12
    assert report.vertex_count == 5
    assert report.crosses_antimeridian is False


def test_rejects_self_intersection():
    with pytest.raises(RoiValidationError, match="invalid"):
        validate_roi(
            polygon([[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]),
            **limits(),
        )


def test_rejects_out_of_range_coordinates():
    with pytest.raises(RoiValidationError, match="longitude"):
        validate_roi(
            polygon([[181, 0], [181, 1], [179, 1], [179, 0], [181, 0]]),
            **limits(),
        )


def test_rejects_area_limit():
    with pytest.raises(RoiValidationError, match="area .* exceeds limit"):
        validate_roi(
            polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            **limits(max_area_km2=100),
        )


def test_rejects_bbox_dimension_for_long_thin_roi():
    with pytest.raises(RoiValidationError, match="bounding-box width"):
        validate_roi(
            polygon([[0, 0], [3, 0], [3, 0.01], [0, 0.01], [0, 0]]),
            **limits(max_width_km=100),
        )


def test_rejects_vertex_limit():
    coords = [[i / 1000, 0] for i in range(6)] + [[0.005, 0.001], [0, 0.001], [0, 0]]
    with pytest.raises(RoiValidationError, match="vertices"):
        validate_roi(polygon(coords), **limits(max_vertices=5))


def test_antimeridian_detected_without_global_width():
    report = validate_roi(
        polygon([[179.5, 10], [-179.5, 10], [-179.5, 11], [179.5, 11], [179.5, 10]]),
        **limits(max_area_km2=20_000, max_width_km=500),
    )
    assert report.crosses_antimeridian is True
    assert report.bbox_width_km < 120


def test_event_date_bounds():
    with pytest.raises(RoiValidationError, match="predates"):
        validate_event_date(date(2014, 10, 2))
    with pytest.raises(RoiValidationError, match="future"):
        validate_event_date(date.today() + timedelta(days=1))
    validate_event_date(date(2021, 8, 14))


def test_environment_overrides_processing_limits(monkeypatch):
    monkeypatch.setenv("SAR_LRA_MAX_ROI_KM2", "123.5")
    monkeypatch.setenv("SAR_LRA_MAX_ROI_WIDTH_KM", "77")
    monkeypatch.setenv("SAR_LRA_MAX_ROI_HEIGHT_KM", "66")
    monkeypatch.setenv("SAR_LRA_MAX_ROI_VERTICES", "4321")
    config = load_config()
    assert config.processing.max_roi_km2 == 123.5
    assert config.processing.max_roi_width_km == 77
    assert config.processing.max_roi_height_km == 66
    assert config.processing.max_roi_vertices == 4321


def test_invalid_environment_limit_fails(monkeypatch):
    monkeypatch.setenv("SAR_LRA_MAX_ROI_VERTICES", "many")
    with pytest.raises(ValueError, match="SAR_LRA_MAX_ROI_VERTICES"):
        load_config()
