# -*- coding: utf-8 -*-

import tifffile
import numpy as np
from SN2N.utils import *  
import os



def normalize_percentage_all_data(
    image_data, 
    pmin=1, 
    pmax=99, 
    normalize_per_slice=True,
    spatial_axes=None  # e.g., (-2, -1) for last two dims as XY
    ):
    """
    Normalize multi-dimensional image data.
    
    Parameters:
        image_data: ndarray
        normalize_per_slice: if True, normalize each "frame" independently
        spatial_axes: tuple of axes considered as spatial (for per-slice norm)
                      Default: last 2 or 3 axes
    
    Returns:
        Normalized array with same shape as input, scaled to [0, 255], dtype=float32
    """
    image_data = np.asarray(image_data)
    ndim = image_data.ndim

    
    # Defults：last 2 axes（2D/3D）or 3 axes (>=4D)
    if spatial_axes is None:
        if ndim >= 4:
            spatial_axes = (-3,-2,-1)  # (-3,-2,-1) for ZYX
        else:
            spatial_axes = (-2, -1)  # YX
    spatial_axes = tuple(ax % ndim for ax in spatial_axes)
    if normalize_per_slice:
        all_axes = set(range(ndim))
        non_spatial_axes = tuple(sorted(all_axes - set(spatial_axes)))

        if not non_spatial_axes:
            normed = normalize_percentage(image_data, pmin=pmin, pmax=pmax )
        else:
            normed = np.empty_like(image_data, dtype=np.float32)

            non_spatial_shapes = [image_data.shape[i] for i in non_spatial_axes]
            for idx in np.ndindex(*non_spatial_shapes):

                full_idx = [slice(None)] * ndim
                for ax, i in zip(non_spatial_axes, idx):
                    full_idx[ax] = i
                block = image_data[tuple(full_idx)]
                normed[tuple(full_idx)] = normalize_percentage(block, pmin=pmin, pmax=pmax)
    else:

        normed = normalize_percentage(image_data, pmin=pmin, pmax=pmax)
    
    return 255 * normed  # still float32, [0, 255]



    



if __name__ == '__main__':
    
    input_dir = 'D:\SN2N-main\'  # Your File Path
    save_dir = 'D:\SN2N-main\'   # Output Path
    pmin=0
    pmax=100,
    normalize_per_slice=True
    spatial_axes=None
    
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    datapath_list = []
    for (root, dirs, files) in os.walk(input_dir):
        for j, Ufile in enumerate(files):
            path = os.path.join(root, Ufile)
            datapath_list.append(path)
            
            
    l = len(datapath_list)
    for ll in range(l):
        print('For number %d frame'%(ll + 1))
        image_data = tifffile.imread(datapath_list[ll])

        image_data_norm = normalize_percentage_all_data(
            image_data,
            pmin=pmin,
            pmax=pmax,
            normalize_per_slice=normalize_per_slice,
            spatial_axes=spatial_axes
            )
 

        image_data_uint8 = image_data_norm.astype(np.uint8)
 

        basename = os.path.basename(datapath_list[ll])
        name, ext = os.path.splitext(basename)
        output_filename = f"{name}_pre{ext}"
        output_path = os.path.join(save_dir, output_filename)
 

        tifffile.imwrite(output_path, image_data_uint8)
        print(f"Saved to: {output_path}")
