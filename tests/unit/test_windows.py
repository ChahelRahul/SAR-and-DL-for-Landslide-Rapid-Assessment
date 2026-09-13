import numpy as np
import pytest

from app.inference.windows import axis_starts, sliding_windows


def coverage_for(shape, size=64, step=32):
    image = np.zeros((*shape, 4), dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.uint16)
    windows = list(sliding_windows(image, size=size, step=step))
    for window, patch in windows:
        assert patch.shape == (size, size, 4)
        coverage[window.y0:window.y1, window.x0:window.x1] += 1
    return windows, coverage


def test_axis_starts_appends_final_edge():
    assert axis_starts(100, 64, 32) == [0, 32, 36]
    assert axis_starts(65, 64, 32) == [0, 1]


@pytest.mark.parametrize(
    "shape,expected_count",
    [
        ((64, 64), 1),
        ((65, 65), 4),
        ((100, 100), 9),
        ((80, 130), 8),
        ((32, 48), 1),
    ],
)
def test_every_source_pixel_is_covered(shape, expected_count):
    windows, coverage = coverage_for(shape)
    assert len(windows) == expected_count
    assert np.all(coverage >= 1)


def test_exact_edge_coordinates_are_source_coordinates():
    image = np.zeros((100, 100, 4), dtype=np.float32)
    windows = list(sliding_windows(image, size=64, step=32))
    assert windows[-1][0].x1 == 100
    assert windows[-1][0].y1 == 100
    assert windows[-1][0].x0 == 36
    assert windows[-1][0].y0 == 36


def test_small_raster_is_edge_padded_but_window_extent_is_clipped():
    image = np.arange(32 * 48 * 4, dtype=np.float32).reshape(32, 48, 4)
    [(window, patch)] = list(sliding_windows(image, size=64, step=32))
    assert (window.x0, window.y0, window.x1, window.y1) == (0, 0, 48, 32)
    assert patch.shape == (64, 64, 4)
    # Edge padding repeats the final observed row/column rather than inventing 0 dB pixels.
    np.testing.assert_array_equal(patch[31, 47], image[31, 47])
    np.testing.assert_array_equal(patch[-1, -1], image[-1, -1])
