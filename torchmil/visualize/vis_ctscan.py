import numpy as np


def slices_to_canvas(
    slices_list: list[np.ndarray],
    slice_size: int,
) -> np.ndarray:
    """
    Given a list of images of CT scan slices, return a canvas with all the slices.

    Arguments:
        slices_list: List of images of CT scan slices. Each image is a numpy array with shape `(slice_size, slice_size, 3)`.
        slice_size: Size of the slices.

    Returns:
        canvas: Canvas with all the slices. It has shape `(slice_size, bag_len*slice_size, 3)`.
    """
    bag_len = len(slices_list)

    max_y = slice_size
    max_x = bag_len * slice_size

    canvas = np.full((max_y, max_x, 3), 255, dtype=np.uint8)

    for i, img in enumerate(slices_list):
        x = i * slice_size
        canvas[0:slice_size, x : x + slice_size] = img

    return canvas


def draw_slices_contour(
    canvas: np.ndarray,
    slice_size: int,
    contour_prop: float = 0.05,
) -> np.ndarray:
    """
    Given a canvas with CT scan slices already drawn, draw a contour around each slice.

    Arguments:
        canvas: Canvas with all the slices. It has shape `(slice_size, bag_len*slice_size, 3)`.
        slice_size: Size of the slices.
        contour_prop: Proportion of the slice size that the contour will cover.

    Returns:
        canvas: Canvas with the contours drawn. It has shape `(slice_size, bag_len*slice_size, 3)`.
    """

    canvas_copy = np.copy(canvas)

    contour_len = contour_prop * slice_size

    for i in range(canvas.shape[1]):
        x = i * slice_size
        canvas_copy[0:slice_size, int(x - contour_len) : int(x + contour_len)] = 0
        canvas_copy[
            0:slice_size,
            int(x + slice_size - contour_len) : int(x + slice_size + contour_len),
        ] = 0

    return canvas_copy


def draw_heatmap_ctscan(
    canvas: np.ndarray,
    values: np.ndarray,
    slice_size: int,
    alpha: float = 0.5,
    max_color: np.ndarray = np.array(
        [0.8392156862745098, 0.15294117647058825, 0.1568627450980392]
    ),
    min_color: np.ndarray = np.array(
        [0.17254901960784313, 0.6274509803921569, 0.17254901960784313]
    ),
) -> np.ndarray:
    """
    Given a canvas with CT scan slices already drawn, draw a heatmap on top of the slices.
    This heatmap is defined by `values`, which should be normalized between 0 and 1.

    Arguments:
        canvas: Canvas with all the slices. It has shape `(slice_size, bag_len*slice_size, 3)`.
        values: List of values to draw the heatmap. Each value should be normalized between 0 and 1.
        slice_size: Size of the slices.
        alpha: Alpha value for blending the heatmap with the canvas.
        max_color: Color for the maximum value in the heatmap.
        min_color: Color for the minimum value in the heatmap.

    Returns:
        canvas: Canvas with the heatmap drawn. It has shape `(slice_size, bag_len*slice_size, 3)`.
    """

    canvas_copy = np.copy(canvas).astype(
        float
    )  # Convert to float *before* calculations

    for i in range(len(values)):
        value = values[i]
        x = i * slice_size
        y = 0
        color = value * max_color + (1 - value) * min_color
        canvas_copy[y : y + slice_size, x : x + slice_size] = (1 - alpha) * canvas_copy[
            y : y + slice_size, x : x + slice_size
        ] + alpha * (
            color * 255
        )  # Scale color to 0-255

    return np.clip(canvas_copy, 0, 255).astype(
        np.uint8
    )  # Clip and convert back to uint8
