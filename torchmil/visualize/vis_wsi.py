import numpy as np

def patches_to_canvas(
        patches_list : list,
        row_array : np.ndarray,
        column_array : np.ndarray,
        patch_size : int
    ) -> np.ndarray:
    max_row = row_array.max()
    max_column = column_array.max()
    max_h = (max_row + 1)*patch_size
    max_w = (max_column + 1)*patch_size
    bag_len = len(patches_list)

    canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
    for i in range(bag_len):
        row = row_array[i]
        column = column_array[i]
        patch = patches_list[i]
        canvas[row*patch_size:(row+1)*patch_size, column*patch_size:(column+1)*patch_size] = patch
    
    return canvas

def draw_patches_contour(
    canvas : np.ndarray,
    row_array : np.ndarray,
    column_array : np.ndarray,
    patch_size : int,
    contour_prop : float = 0.05,
) -> np.ndarray:

    canvas_copy = np.copy(canvas)

    for i in range(len(row_array)):
        row = row_array[i]
        column = column_array[i]
        x = column * patch_size
        y = row * patch_size
        contour_len = contour_prop*patch_size
        canvas_copy[y:y+patch_size, int(x-contour_len):int(x+contour_len)] = 0
        canvas_copy[y:y+patch_size, int(x+patch_size-contour_len):int(x+patch_size+contour_len)] = 0
        
        canvas_copy[int(y-contour_len):int(y+contour_len), x:x+patch_size] = 0
        canvas_copy[int(y+patch_size-contour_len):int(y+patch_size+contour_len), x:x+patch_size] = 0
    
    return canvas_copy

def draw_heatmap_wsi(
        canvas : np.ndarray,
        values : np.ndarray,
        patch_size : int,
        row_array : np.ndarray,
        col_array : np.ndarray,
        alpha : float = 0.5,
        max_color : np.ndarray = np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
        min_color : np.ndarray = np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
    ) -> np.ndarray:

    canvas_copy = np.copy(canvas)

    for i in range(len(row_array)):
        row = row_array[i]
        column = col_array[i]
        x = column * patch_size
        y = row * patch_size

        if min_color == 'blank':
            color = 255*max_color
            alpha = values[i]
        else:
            w = values[i]
            color = 255*(w*max_color + (1.0-w)*min_color)

        canvas_copy[y:y+patch_size, x:x+patch_size] = (alpha)*color + (1.0-alpha)*canvas[y:y+patch_size, x:x+patch_size]
            
    return canvas_copy