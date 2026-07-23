from pathlib import Path

from app.schemas import RasterInferenceRequest


def test_raster_request_is_explicit() -> None:
    request = RasterInferenceRequest(
        request_id="x",
        orbit="ASCENDING",
        weights_path=Path("weights.h5"),
        input_raster=Path("input.tif"),
    )
    assert request.orbit == "ASCENDING"
