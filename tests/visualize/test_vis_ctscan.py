import numpy as np
import pytest

from torchmil.visualize.vis_ctscan import (
    slices_to_canvas,
    draw_slices_contour,
    draw_heatmap_ctscan,
)  # Replace your_module


# Fixtures for common setup
@pytest.fixture
def sample_slices():
    slice_size = 10
    num_slices = 5
    slices_list = [
        np.random.randint(0, 256, size=(slice_size, slice_size, 3), dtype=np.uint8)
        for _ in range(num_slices)
    ]
    return slices_list, slice_size


@pytest.fixture
def sample_canvas(sample_slices):
    slices_list, slice_size = sample_slices
    return slices_to_canvas(slices_list, slice_size), slice_size, len(slices_list)


@pytest.fixture
def sample_values(sample_canvas):
    _, _, num_slices = sample_canvas
    return np.random.rand(num_slices)


# Tests for slices_to_canvas function
def test_slices_to_canvas_output_shape(sample_slices):
    slices_list, slice_size = sample_slices
    canvas = slices_to_canvas(slices_list, slice_size)
    expected_shape = (slice_size, len(slices_list) * slice_size, 3)
    assert canvas.shape == expected_shape


def test_slices_to_canvas_content(sample_slices):
    slices_list, slice_size = sample_slices
    canvas = slices_to_canvas(slices_list, slice_size)
    for i, img in enumerate(slices_list):
        x_start = i * slice_size
        x_end = (i + 1) * slice_size
        assert np.array_equal(canvas[0:slice_size, x_start:x_end], img)


def test_slices_to_canvas_empty_list():
    slices_list = []
    slice_size = 10
    canvas = slices_to_canvas(slices_list, slice_size)
    assert canvas.shape == (slice_size, 0, 3)


# Tests for draw_slices_contour function
def test_draw_slices_contour_output_shape(sample_canvas):
    canvas, slice_size, _ = sample_canvas
    canvas_with_contour = draw_slices_contour(canvas, slice_size)
    assert canvas_with_contour.shape == canvas.shape


def test_draw_slices_contour_contour_presence(sample_canvas):
    canvas, slice_size, num_slices = sample_canvas
    canvas_with_contour = draw_slices_contour(canvas, slice_size, contour_prop=0.1)
    contour_width = int(0.1 * slice_size)

    print(f"canvas_with_contour shape: {canvas_with_contour.shape}")

    for i in range(num_slices):
        x_start = i * slice_size
        x_end = (i + 1) * slice_size

        # Check ONLY the contour pixels.
        left_boundary_start = x_start - contour_width
        left_boundary_end = x_start
        left_boundary = canvas_with_contour[
            0:slice_size, left_boundary_start:left_boundary_end
        ]
        print(
            f"Slice {i}: x_start={x_start}, x_end={x_end}, left_boundary_start={left_boundary_start}, left_boundary_end={left_boundary_end}"
        )
        print(f"Slice {i} left_boundary: {left_boundary}")
        if left_boundary_start >= 0:  # Only check if within bounds
            assert np.all(
                left_boundary == 0
            ), f"Left contour check failed for slice {i}, left_boundary_start={left_boundary_start}, left_boundary_end={left_boundary_end}, contour_width={contour_width}"

        right_boundary_start = x_end
        right_boundary_end = x_end + contour_width
        right_boundary = canvas_with_contour[
            0:slice_size, right_boundary_start:right_boundary_end
        ]
        print(
            f"Slice {i}:  right_boundary_start={right_boundary_start}, right_boundary_end={right_boundary_end}"
        )
        print(f"Slice {i} right_boundary: {right_boundary}")
        if right_boundary_end <= canvas.shape[1]:
            assert np.all(
                right_boundary == 0
            ), f"Right contour check failed for slice {i}, right_boundary_start={right_boundary_start}, right_boundary_end={right_boundary_end}, contour_width={contour_width}"


def test_draw_slices_contour_no_contour_if_prop_zero(sample_canvas):
    canvas, slice_size, _ = sample_canvas
    canvas_with_contour = draw_slices_contour(canvas, slice_size, contour_prop=0.0)
    assert np.array_equal(canvas_with_contour, canvas)


# Tests for draw_heatmap_ctscan function
def test_draw_heatmap_ctscan_output_shape(sample_canvas, sample_values):
    canvas, _, _ = sample_canvas
    canvas_with_heatmap = draw_heatmap_ctscan(canvas, sample_values, canvas.shape[0])
    assert canvas_with_heatmap.shape == canvas.shape


# Tests for draw_heatmap_ctscan function
def test_draw_heatmap_ctscan_blending(sample_canvas, sample_values):
    canvas, slice_size, num_slices = sample_canvas
    alpha = 0.5
    max_color = np.array([1.0, 0.0, 0.0])  # Red
    min_color = np.array([0.0, 1.0, 0.0])  # Green
    canvas_with_heatmap = draw_heatmap_ctscan(
        canvas,
        sample_values,
        slice_size,
        alpha=alpha,
        max_color=max_color,
        min_color=min_color,
    )
    for i in range(num_slices):
        value = sample_values[i]
        x_start = i * slice_size
        x_end = (i + 1) * slice_size
        original_slice = canvas[0:slice_size, x_start:x_end].astype(float) / 255.0
        heatmap_color = value * max_color + (1 - value) * min_color
        expected_color = (1 - alpha) * original_slice + alpha * heatmap_color
        blended_slice = (
            canvas_with_heatmap[0:slice_size, x_start:x_end].astype(float) / 255.0
        )
        assert np.allclose(
            blended_slice, expected_color, atol=1e-2, rtol=1e-5
        ), f"Heatmap blending failed for slice {i} with value {value}"


def test_draw_heatmap_ctscan_alpha_zero(sample_canvas, sample_values):
    canvas, slice_size, _ = sample_canvas
    canvas_with_heatmap = draw_heatmap_ctscan(
        canvas, sample_values, slice_size, alpha=0.0
    )
    assert np.array_equal(canvas_with_heatmap, canvas)


def test_draw_heatmap_ctscan_alpha_one(sample_canvas, sample_values):
    canvas, slice_size, num_slices = sample_canvas
    max_color = np.array([0.8, 0.2, 0.2])
    min_color = np.array([0.2, 0.8, 0.2])
    canvas_with_heatmap = draw_heatmap_ctscan(
        canvas,
        sample_values,
        slice_size,
        alpha=1.0,
        max_color=max_color,
        min_color=min_color,
    )
    for i in range(num_slices):
        value = sample_values[i]
        x_start = i * slice_size
        x_end = (i + 1) * slice_size
        expected_color_int = (value * max_color + (1 - value) * min_color) * 255
        assert np.allclose(
            canvas_with_heatmap[0:slice_size, x_start:x_end], expected_color_int, atol=5
        )  # Allow for slight conversion errors


def test_draw_heatmap_ctscan_different_color_specs(sample_canvas, sample_values):
    canvas, slice_size, _ = sample_canvas
    max_color = np.array([0.1, 0.5, 0.9])
    min_color = np.array([0.9, 0.3, 0.1])
    canvas_with_heatmap = draw_heatmap_ctscan(
        canvas, sample_values, slice_size, max_color=max_color, min_color=min_color
    )
    assert canvas_with_heatmap.shape == canvas.shape
