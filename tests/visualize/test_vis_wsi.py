import numpy as np
import pytest

# Assuming your functions are in a file named 'your_module.py'
from torchmil.visualize.vis_wsi import (
    patches_to_canvas,
    draw_patches_contour,
    draw_heatmap_wsi,
)


# Fixtures for common setup
@pytest.fixture
def sample_patches_data():
    patch_size = 10
    num_patches = 5
    patches_list = [
        np.random.randint(0, 256, size=(patch_size, patch_size, 3), dtype=np.uint8)
        for _ in range(num_patches)
    ]
    row_array = np.array([0, 1, 0, 2, 1])  # Example row indices
    column_array = np.array([0, 0, 1, 0, 2])  # Example column indices
    return patches_list, row_array, column_array, patch_size


@pytest.fixture
def sample_canvas_from_patches(sample_patches_data):
    patches_list, row_array, column_array, patch_size = sample_patches_data
    canvas = patches_to_canvas(patches_list, row_array, column_array, patch_size)
    return canvas, row_array, column_array, patch_size


@pytest.fixture
def sample_heatmap_data(sample_canvas_from_patches):
    canvas, row_array, col_array, patch_size = sample_canvas_from_patches
    num_patches = len(row_array)
    values = np.random.rand(num_patches)  # Example heatmap values
    return canvas, values, patch_size, row_array, col_array


# Tests for patches_to_canvas function
def test_patches_to_canvas_output_shape(sample_patches_data):
    patches_list, row_array, column_array, patch_size = sample_patches_data
    canvas = patches_to_canvas(patches_list, row_array, column_array, patch_size)
    max_row = row_array.max()
    max_column = column_array.max()
    expected_shape = ((max_row + 1) * patch_size, (max_column + 1) * patch_size, 3)
    assert canvas.shape == expected_shape, "Output shape is incorrect"


def test_patches_to_canvas_content(sample_patches_data):
    patches_list, row_array, column_array, patch_size = sample_patches_data
    canvas = patches_to_canvas(patches_list, row_array, column_array, patch_size)
    for i, patch in enumerate(patches_list):
        row = row_array[i]
        column = column_array[i]
        x_start = column * patch_size
        x_end = (column + 1) * patch_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        assert np.array_equal(
            canvas[y_start:y_end, x_start:x_end], patch
        ), f"Patch {i} content is incorrect"


def test_patches_to_canvas_empty_patches():
    patch_size = 10
    row_array = np.array([])
    column_array = np.array([])
    patches_list = []
    canvas = patches_to_canvas(patches_list, row_array, column_array, patch_size)
    assert canvas.shape == (patch_size, patch_size, 3), "Empty patches test failed"


# Tests for draw_patches_contour function
def test_draw_patches_contour_output_shape(sample_canvas_from_patches):
    canvas, row_array, column_array, patch_size = sample_canvas_from_patches
    canvas_with_contour = draw_patches_contour(
        canvas, row_array, column_array, patch_size
    )
    assert canvas_with_contour.shape == canvas.shape, "Output shape is incorrect"


def test_draw_patches_contour_contour_presence(sample_canvas_from_patches):
    canvas, row_array, column_array, patch_size = sample_canvas_from_patches
    canvas_with_contour = draw_patches_contour(
        canvas, row_array, column_array, patch_size, contour_prop=0.1
    )
    contour_width = int(0.1 * patch_size)

    for i in range(len(row_array)):
        row = row_array[i]
        column = column_array[i]
        x_start = column * patch_size
        x_end = (column + 1) * patch_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size

        # Check left contour
        left_boundary_start = max(0, x_start - contour_width)
        left_boundary_end = x_start + contour_width
        left_boundary = canvas_with_contour[
            y_start:y_end, left_boundary_start:left_boundary_end
        ]
        assert np.any(left_boundary == 0), f"Left contour failed for patch {i}"

        # Check right contour
        right_boundary_start = x_end - contour_width
        right_boundary_end = min(canvas.shape[1], x_end + contour_width)
        right_boundary = canvas_with_contour[
            y_start:y_end, right_boundary_start:right_boundary_end
        ]
        assert np.any(right_boundary == 0), f"Right contour failed for patch {i}"

        # Check top contour
        top_boundary_start = max(0, y_start - contour_width)
        top_boundary_end = y_start + contour_width
        top_boundary = canvas_with_contour[
            top_boundary_start:top_boundary_end, x_start:x_end
        ]
        assert np.any(top_boundary == 0), f"Top contour failed for patch {i}"

        # Check bottom contour
        bottom_boundary_start = y_end - contour_width
        bottom_boundary_end = min(canvas.shape[0], y_end + contour_width)
        bottom_boundary = canvas_with_contour[
            bottom_boundary_start:bottom_boundary_end, x_start:x_end
        ]
        assert np.any(bottom_boundary == 0), f"Bottom contour failed for patch {i}"


def test_draw_patches_contour_no_contour_if_prop_zero(sample_canvas_from_patches):
    canvas, row_array, column_array, patch_size = sample_canvas_from_patches
    canvas_with_contour = draw_patches_contour(
        canvas, row_array, column_array, patch_size, contour_prop=0.0
    )
    assert np.array_equal(
        canvas_with_contour, canvas
    ), "Contour drawn with zero proportion"


# Tests for draw_heatmap_wsi function
def test_draw_heatmap_wsi_output_shape(sample_heatmap_data):
    canvas, values, patch_size, row_array, col_array = sample_heatmap_data
    canvas_with_heatmap = draw_heatmap_wsi(
        canvas, values, patch_size, row_array, col_array
    )
    assert canvas_with_heatmap.shape == canvas.shape, "Output shape is incorrect"


def test_draw_heatmap_wsi_blending(sample_heatmap_data):
    canvas, values, patch_size, row_array, col_array = sample_heatmap_data
    alpha = 0.5
    max_color = np.array([1.0, 0.0, 0.0])  # Red
    min_color = np.array([0.0, 1.0, 0.0])  # Green
    canvas_with_heatmap = draw_heatmap_wsi(
        canvas,
        values,
        patch_size,
        row_array,
        col_array,
        alpha=alpha,
        max_color=max_color,
        min_color=min_color,
    )
    for i in range(len(row_array)):
        row = row_array[i]
        column = col_array[i]
        x_start = column * patch_size
        x_end = (column + 1) * patch_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        value = values[i]
        expected_color = value * max_color + (1 - value) * min_color  # in float
        # Convert both to float for comparison
        result_color = (
            canvas_with_heatmap[y_start:y_end, x_start:x_end].astype(np.float64) / 255.0
        )
        expected_color_float = expected_color
        original_color = canvas[y_start:y_end, x_start:x_end].astype(np.float64) / 255.0
        blended_color = (alpha * expected_color_float) + ((1 - alpha) * original_color)

        assert np.allclose(
            result_color, blended_color, atol=0.01
        ), f"Heatmap blending failed for patch {i}"


def test_draw_heatmap_wsi_alpha_zero(sample_heatmap_data):
    canvas, values, patch_size, row_array, col_array = sample_heatmap_data
    canvas_with_heatmap = draw_heatmap_wsi(
        canvas, values, patch_size, row_array, col_array, alpha=0.0
    )
    assert np.array_equal(canvas_with_heatmap, canvas), "Heatmap drawn with alpha=0"


def test_draw_heatmap_wsi_alpha_one(sample_heatmap_data):
    canvas, values, patch_size, row_array, col_array = sample_heatmap_data
    alpha = 1.0
    max_color = np.array([0.8, 0.2, 0.2])
    min_color = np.array([0.2, 0.8, 0.2])
    canvas_with_heatmap = draw_heatmap_wsi(
        canvas,
        values,
        patch_size,
        row_array,
        col_array,
        alpha=alpha,
        max_color=max_color,
        min_color=min_color,
    )
    for i in range(len(row_array)):
        row = row_array[i]
        column = col_array[i]
        x_start = column * patch_size
        x_end = (column + 1) * patch_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        value = values[i]
        expected_color_int = 255 * (value * max_color + (1 - value) * min_color)
        assert np.allclose(
            canvas_with_heatmap[y_start:y_end, x_start:x_end],
            expected_color_int,
            atol=5,
        ), f"Heatmap with alpha=1 failed for patch {i}"


def test_draw_heatmap_wsi_min_color_blank(sample_heatmap_data):
    canvas, values, patch_size, row_array, col_array = sample_heatmap_data
    alpha = 0.5
    max_color = np.array([1.0, 0.0, 0.0])  # Red
    min_color = "blank"
    canvas_with_heatmap = draw_heatmap_wsi(
        canvas,
        values,
        patch_size,
        row_array,
        col_array,
        alpha=alpha,
        max_color=max_color,
        min_color=min_color,
    )
    for i in range(len(row_array)):
        row = row_array[i]
        column = col_array[i]
        x_start = column * patch_size
        x_end = (column + 1) * patch_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        expected_color = 255 * max_color
        expected_alpha = values[i]
        result_color = canvas_with_heatmap[y_start:y_end, x_start:x_end]
        # Check if the color is close to the expected color and alpha is applied.

        assert np.allclose(
            result_color,
            (expected_alpha) * expected_color
            + (1 - (expected_alpha)) * canvas[y_start:y_end, x_start:x_end],
            atol=5,
        ), f"Heatmap with min_color='blank' failed for patch {i}"
