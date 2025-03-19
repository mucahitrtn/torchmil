import numpy as np

def slices_to_canvas(
        slices_list : list[np.ndarray],
        slice_size : int,
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
        x = i*slice_size
        canvas[0:slice_size, x:x+slice_size] = img
    
    return canvas

def draw_slices_contour(
    canvas : np.ndarray,
    slice_size : int,
    contour_prop : float = 0.05,
) -> np.ndarray:

    canvas_copy = np.copy(canvas)

    contour_len = contour_prop*slice_size

    for i in range(canvas.shape[1]):
        x = i*slice_size
        canvas_copy[0:slice_size, int(x-contour_len):int(x+contour_len)] = 0
        canvas_copy[0:slice_size, int(x+slice_size-contour_len):int(x+slice_size+contour_len)] = 0

    return canvas_copy

def draw_heatmap_ctscan(
        canvas : np.ndarray,
        values : np.ndarray,
        slice_size : int,
        alpha : float = 0.5,
        max_color : np.ndarray = np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
        min_color : np.ndarray = np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
    ) -> np.ndarray:

    canvas_copy = np.copy(canvas)

    for i in range(len(values)):
        value = values[i]
        x = i*slice_size
        y = 0
        color = value*max_color + (1-value)*min_color
        canvas_copy[y:y+slice_size, x:x+slice_size] = (1-alpha)*canvas_copy[y:y+slice_size, x:x+slice_size] + alpha*color
    
    return canvas_copy