import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import zarr
import h5py
import cv2
import os
import numpy as np
import math

COLOR_DICT = {
    'yellow': np.array([1.0, 1.0, 0.0]),
    'blue': np.array([0.12156862745098039, 0.4666666666666667, 0.7058823529411765]),
    'green': np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
    'red': np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
}


def plot_att_score(s_pred, y_true, T_true, bag_idx):
    """
    Args:
        s_pred: (num_inst,) attention score of the instances
        y_true: (num_inst,) label of the instances
        T_true: (num_bags,) label of the bags
        bag_idx: (num_inst,) maps each instance to its bag    
    """

    print('Plotting attention score distribution')

    # Keep instances belonging to positive bags

    pos_bags_idx = np.where(T_true == 1)[0] # positive bags

    idx_keep = np.isin(bag_idx, pos_bags_idx)

    y_true = y_true[idx_keep]
    s_pred = s_pred[idx_keep]

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_inst = s_pred[pos_idx]
    neg_inst = s_pred[neg_idx]
    
    fig, ax = plt.subplots()
    counts, bins = np.histogram(neg_inst, bins=20, density=True)
    ax.hist(bins[:-1], bins, weights=counts, label='Negative instances', edgecolor='black', alpha=0.8)
    counts, bins = np.histogram(pos_inst, bins=20, density=True)
    ax.hist(bins[:-1], bins, weights=counts, label='Positive instances', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Attention score')
    ax.set_ylabel('Frequency')
    ax.legend()

    return fig

def plot_att_val(f_pred, y_true, T_true, bag_idx):
    """
    Args:
        s_pred: (num_inst,) attention score of the instances
        y_true: (num_inst,) label of the instances
        T_true: (num_bags,) label of the bags
        bag_idx: (num_inst,) maps each instance to its bag    
    """

    print('Plotting attention values distribution')

    pos_bags_idx = np.where(T_true == 1)[0] # positive bags

    idx_keep = np.isin(bag_idx, pos_bags_idx)

    y_true = y_true[idx_keep]
    f_pred = f_pred[idx_keep]
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_inst = f_pred[pos_idx]
    neg_inst = f_pred[neg_idx]
    
    fig, ax = plt.subplots()
    counts, bins = np.histogram(neg_inst, bins=20, density=True)
    ax.hist(bins[:-1], bins, weights=counts, label='Negative instances', edgecolor='black', alpha=0.8, color='tab:green')
    counts, bins = np.histogram(pos_inst, bins=20, density=True)
    ax.hist(bins[:-1], bins, weights=counts, label='Positive instances', edgecolor='black', alpha=0.7, color='tab:red')
    ax.set_xlabel('Attention value')
    ax.set_ylabel('Frequency')
    ax.legend()

    return fig

def plot_att_hist(ax, f_pred, y_true, T_true, bag_idx, legend=True):
    """
    Args:
        s_pred: (num_inst,) attention score of the instances
        y_true: (num_inst,) label of the instances
        T_true: (num_bags,) label of the bags
        bag_idx: (num_inst,) maps each instance to its bag    
    """

    print('Plotting attention values distribution')

    pos_bags_idx = np.where(T_true == 1)[0] # positive bags

    idx_keep = np.isin(bag_idx, pos_bags_idx)

    y_true = y_true[idx_keep]
    f_pred = f_pred[idx_keep]
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_inst = f_pred[pos_idx]
    neg_inst = f_pred[neg_idx]
    
    counts_pos, bins_pos = np.histogram(pos_inst, bins=20, density=False)
    counts_pos = counts_pos / counts_pos.sum()
    ax.hist(bins_pos[:-1], bins_pos, weights=counts_pos, label='Positive instances', edgecolor='black', alpha=0.7, color='tab:red')
    
    counts_neg, bins_neg = np.histogram(neg_inst, bins=20, density=False)
    counts_neg = counts_neg / counts_neg.sum()
    ax.hist(bins_neg[:-1], bins_neg, weights=counts_neg, label='Negative instances', edgecolor='black', alpha=0.7, color='tab:green')
    
    ax.set_ylim(0, 1)

    if legend:
        ax.legend(frameon=False)

    return ax

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def combine_attmap_wsi(attmap, wsi_img):
    """
    Input:
        attmap: numpy array, shape = (H, W)
        wsi_img: numpy array, shape = (H, W, 3), mode RGB
    Output:
        combined_img: numpy array, shape = (H, W, 3), mode RGB
    """
    # attmap = normalize(attmap) # (H, W)
    attmap_img_bgr = cv2.applyColorMap((attmap*255).astype(np.uint8), cv2.COLORMAP_JET) # (H, W, 3)
    wsi_img_bgr = cv2.cvtColor(wsi_img, cv2.COLOR_RGB2BGR) # (H, W, 3)
    combined_img_bgr = cv2.addWeighted(wsi_img_bgr, 0.5, attmap_img_bgr, 0.5, 0) # (H, W, 3)
    combined_img_rgb = cv2.cvtColor(combined_img_bgr, cv2.COLOR_BGR2RGB) # (H, W, 3)
    return combined_img_rgb

def plot_wsi_and_heatmap(
        ax,
        canvas_wsi, 
        attval = None, 
        plot_patch_contour=False,
        size = None,
        row_array = None,
        col_array = None,
        start_y = 0,
        start_x = 0,
        height = None,
        width = None,
        alpha = 0.8 ,
        p = 0.05,
        heatmap_mode = 'green_red',
        remove_axis = True
    ):

    if width is None:
        width = canvas_wsi.shape[0]
    if height is None:
        height = canvas_wsi.shape[1]


    canvas_wsi_copy = np.copy(canvas_wsi)

    # ax.imshow(canvas_wsi[start_y:start_y+width, start_x:start_x+height])
    if plot_patch_contour or (attval is not None):
        for i in range(len(row_array)):
            row_i = row_array[i]
            column_i = col_array[i]
            x_i = column_i * size
            y_i = row_i * size
            row = row_array[i]
            column = col_array[i]
            x = column * size
            y = row * size
            if y_i >= start_y and y_i <= start_y+width and x_i >= start_x and x_i <= start_x+height:

                if attval is None:
                    color = contour_color = 0.0
                else:
                    first_color = heatmap_mode.split('_')[0]
                    second_color = heatmap_mode.split('_')[1]
                    if first_color == 'blank':
                        color = 255*COLOR_DICT[second_color]
                        alpha = attval[i]
                    else:
                        w = attval[i]
                        color = 255*(w*COLOR_DICT[second_color] + (1.0-w)*COLOR_DICT[first_color])
                    contour_color = color

                    canvas_wsi_copy[y:y+size, x:x+size] = (alpha)*color + (1.0-alpha)*canvas_wsi[y:y+size, x:x+size]
                
                if plot_patch_contour:
                    contour_len = p*size
                    canvas_wsi_copy[y:y+size, int(x-contour_len):int(x+contour_len)] = contour_color
                    canvas_wsi_copy[y:y+size, int(x+size-contour_len):int(x+size+contour_len)] = contour_color
                    
                    canvas_wsi_copy[int(y-contour_len):int(y+contour_len), x:x+size] = contour_color
                    canvas_wsi_copy[int(y+size-contour_len):int(y+size+contour_len), x:x+size] = contour_color
                    
                    # ax.add_patch(plt.Rectangle((x, y), size, size, edgecolor='black', fill=False))
    ax.imshow(canvas_wsi_copy[start_y:start_y+width, start_x:start_x+height])
    if remove_axis:
        ax.axis('off')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    # remove axis ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.axis('off')
    return ax

def plot_scan_and_heatmap(
        ax,
        canvas_bag, 
        attval = None, 
        plot_patch_contour=False,
        size = None,
        alpha = 0.8,
        p = 0.05,
        heatmap_mode = 'green_red'
    ):

    width = canvas_bag.shape[0]
    height = canvas_bag.shape[1]
    
    tab_red = np.array([
        0.8392156862745098,
        0.15294117647058825,
        0.1568627450980392
    ])
    tab_green = np.array([
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313
    ])

    start_x = 0
    start_y = 0

    bag_len = height // size

    canvas_bag_copy = np.copy(canvas_bag)

    if plot_patch_contour or (attval is not None):  
        for i in range(bag_len):
            x = i * size
            y = 0
            color = 0

            if attval is None:
                color = contour_color = 0.0
            else:

                first_color = heatmap_mode.split('_')[0]
                second_color = heatmap_mode.split('_')[1]
                if first_color == 'blank':
                    color = 255*COLOR_DICT[second_color]
                    alpha = attval[i]
                else:
                    w = attval[i]
                    color = 255*(w*COLOR_DICT[second_color] + (1.0-w)*COLOR_DICT[first_color])
                contour_color = color

                canvas_bag_copy[y:y+size, x:x+size] = (alpha)*color + (1.0-alpha)*canvas_bag[y:y+size, x:x+size]

            if plot_patch_contour:
                # ax.add_patch(plt.Rectangle((x, y), size, size, edgecolor='black', fill=False))
                contour_len = p*size
                canvas_bag_copy[y:y+size, int(x-contour_len):int(x+contour_len)] = contour_color
                canvas_bag_copy[y:y+size, int(x+size-contour_len):int(x+size+contour_len)] = contour_color

                canvas_bag_copy[int(y-contour_len):int(y+contour_len), x:x+size] = contour_color
                canvas_bag_copy[int(y+size-contour_len):int(y+size+contour_len), x:x+size] = contour_color

    ax.imshow(canvas_bag_copy[start_y:start_y+width, start_x:start_x+height])
                
    ax.axis('off')
    return ax

def read_scan_slices(data_dir, inst_paths, size=512, resize_size=10):

    slices_list = []
    pbar = tqdm(total=len(inst_paths))
    for inst_path in inst_paths:
        inst_name = inst_path.split('/')[-1]
        pbar.update(1)
        img = np.load(data_dir+inst_name)
        # img = cv2.imread(f'{DATA_DIR}/images/' + patch_name + '.png')
        if resize_size != size:
            img = cv2.resize(img, (resize_size, resize_size))
        slices_list.append(img)
    return slices_list

def read_wsi_patches(data_dir, coords_path, bag_name, size=512, resize_size=10, ext='.tif'):
    wsi_path = os.path.join(data_dir, bag_name + ext)
    inst_coords = np.array(h5py.File(os.path.join(coords_path, bag_name + '.h5'), 'r')['coords'])
    bag_len = len(inst_coords)
    patches_list = []
    row_list = []
    column_list = []
    pbar = tqdm(range(bag_len), total=bag_len)
    for i in pbar:
        coord = inst_coords[i]
        x, y = coord
        
        store = tifffile.imread(wsi_path, aszarr=True)
        z = zarr.open(store, mode='r')
        wsi = z[0]
        patch = wsi[y:y+size, x:x+size]
        patch = cv2.resize(patch, (resize_size, resize_size))
        patches_list.append(patch)

        row = int(y / size)
        column = int(x / size)
        row_list.append(row)
        column_list.append(column)

    row_array = np.array(row_list)
    column_array = np.array(column_list)

    # normalize rows

    row_array = row_array - row_array.min()
    column_array = column_array - column_array.min()

    return patches_list, row_array, column_array

def slices_to_canvas(slices_list, resize_size=10):
    bag_len = len(slices_list)

    max_y = resize_size
    max_x = bag_len * resize_size

    canvas_bag = np.zeros((max_y, max_x, 3), dtype=np.int32) + 255

    for i, img in enumerate(slices_list):
        x = i*resize_size
        canvas_bag[0:resize_size, x:x+resize_size] = 255*img
    
    return canvas_bag

def patches_to_canvas(patches_list, row_array, column_array, resize_size=10):
    max_row = row_array.max()
    max_column = column_array.max()
    max_h = (max_row + 1)*resize_size
    max_w = (max_column + 1)*resize_size
    bag_len = len(patches_list)

    canvas_wsi = np.zeros((max_h, max_w, 3), dtype=np.uint8)+255
    pbar = tqdm(total=bag_len)
    for i in range(bag_len):
        pbar.update(1)
        row = row_array[i]
        column = column_array[i]
        patch = patches_list[i]
        canvas_wsi[row*resize_size:(row+1)*resize_size, column*resize_size:(column+1)*resize_size] = patch
    
    return canvas_wsi

def pad_canvas(canvas_wsi, row_array, column_array, num_pad=5, resize_size=10):
    num_pad = 5

    # pad the image
    canvas_wsi = np.pad(canvas_wsi, ((num_pad*resize_size, num_pad*resize_size), (num_pad*resize_size, num_pad*resize_size), (0, 0)), mode='constant', constant_values=255)

    # update row and column arrays
    row_array = row_array + num_pad
    column_array = column_array + num_pad

    return canvas_wsi, row_array, column_array