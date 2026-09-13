import numpy as np

from app.config import ModelConfig, ProcessingConfig
from app.inference.predict import predict_stack


class AlwaysPositiveModel:
    def __init__(self):
        self.last_shape = None

    def predict(self, patches, batch_size, verbose):
        self.last_shape = patches.shape
        return np.ones((len(patches), 1), dtype=np.float32)


def stack(height, width):
    result = np.zeros((height, width, 4), dtype=np.float32)
    result[..., 0] = -15
    result[..., 1] = -20
    result[..., 2] = 1
    result[..., 3] = -1
    return result


def test_subwindow_raster_is_padded_for_model_but_output_keeps_source_shape():
    model = AlwaysPositiveModel()
    prediction = predict_stack(model, stack(32, 48), ModelConfig(), ProcessingConfig())
    assert model.last_shape == (1, 64, 64, 4)
    assert prediction.mask.shape == (32, 48)
    assert prediction.probability_surface.shape == (32, 48)
    assert prediction.selected_windows[0].x1 <= 48
    assert prediction.selected_windows[0].y1 <= 32


def test_invalid_and_outside_roi_pixels_cannot_become_detections():
    valid = np.zeros((65, 65), dtype=bool)
    valid[10:50, 12:55] = True
    model = AlwaysPositiveModel()
    prediction = predict_stack(
        model,
        stack(65, 65),
        ModelConfig(),
        ProcessingConfig(),
        valid_mask=valid,
    )
    assert np.all(prediction.mask[~valid] == 0)
    assert prediction.mask[valid].any()
