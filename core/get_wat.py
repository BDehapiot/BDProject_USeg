#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 
from skimage.transform import resize
from skimage.filters import threshold_li
from skimage.filters import sato, gaussian
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border
from skimage.morphology import binary_dilation, remove_small_objects

#%% To do list ----------------------------------------------------------------

'''
- Handle bit depth (convert to uint8)
- Extract advanced parameters
- Get an auto ridge_size
- back size wat as an option

'''

#%% Function (get_wat) --------------------------------------------------------

def get_wat(
        raw, 
        binning, 
        ridge_size, 
        thresh_coeff, 
        small_cell_cutoff,
        large_cell_cutoff,
        remove_border_cells=True, 
        parallel=False
        ):
        
    # Nested function ---------------------------------------------------------
    
    def _get_wat(raw):
        
        # Resize (according to binning)  
        rsize = resize(raw, (
            int(raw.shape[0]//binning), 
            int(raw.shape[1]//binning)),
            preserve_range=True, 
            anti_aliasing=True,
            )        
        
        # Subtract background
        rsize -= rolling_ball(
            gaussian(rsize, sigma=ridge_size//2), radius=ridge_size*2,
            )    
        
        # Apply ridge filter 
        ridges = sato(
            rsize, sigmas=ridge_size, mode='reflect', black_ridges=False,
            )
        
        # Get mask
        thresh = threshold_li(ridges, tolerance=1)
        mask = ridges > thresh*thresh_coeff
        mask = remove_small_objects(mask, min_size=np.sum(mask)*0.01)
        markers = label(np.invert(mask), connectivity=1)
        
        # Filter cells (according to area)
        props = regionprops(markers)
        info = pd.DataFrame(
            ([(i.label, i.area, 0) for i in props]),
            columns = ['idx', 'area', 'valid']
            )
        median = np.median(info['area'])
        for i in range(len(info)):
            # Remove small cells
            if info['area'][i] < median/small_cell_cutoff: 
                info['valid'][i] = 1
                markers[markers==info['idx'][i]] = 0  
            # Remove large cells
            if info['area'][i] > median*large_cell_cutoff: 
                info['valid'][i] = 2
                
        # Get watershed labels
        labels = watershed(
            ridges, 
            markers, 
            compactness=10/binning, 
            watershed_line=True
            ) 

        # Remove large cells
        for idx in info.loc[info['valid'] == 2, 'idx']:
            labels[labels==idx] = 0  

        # Remove border cells
        if remove_border_cells:
            labels = clear_border(labels)
            
        # Get watershed wat
        wat = labels == 0
        temp = np.invert(wat)
        temp = binary_dilation(temp)
        wat = np.minimum(wat, temp)
        
        return rsize, ridges, mask, markers, labels, wat
    
    # Main function -----------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
        
    # Adjust parameters according to binning
    ridge_size = ridge_size/binning
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-2)(
            delayed(_get_wat)(
                img,
                )
            for img in raw
            )
        
    else:
        
        # Run serial
        output_list = [_get_wat(
                img,
                )
            for img in raw
            ]
        
    # Extract output dictionary
    output_dict = {
        'rsize': np.stack(
            [img[0] for img in output_list], axis=0).squeeze(),
        'ridges': np.stack(
            [img[1] for img in output_list], axis=0).squeeze(),
        'mask': np.stack(
            [img[2] for img in output_list], axis=0).squeeze(),
        'markers': np.stack(
            [img[3] for img in output_list], axis=0).squeeze(),
        'labels': np.stack(
            [img[4] for img in output_list], axis=0).squeeze(),
        'wat':np.stack(
            [img[5] for img in output_list], axis=0).squeeze(),
        }
    
    return output_dict

#%% Run -----------------------------------------------------------------------

# File name
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# # Parameters
# binning = 2
# ridge_size = 3
# thresh_coeff = 0.75
# small_cell_cutoff = 10
# large_cell_cutoff = 10
# remove_border_cells = False

# # Open data
# raw = io.imread(Path('../data/', raw_name))

# # Process data (get_wat)
# start = time.time()
# print('get_wat')

# output_dict = get_wat(
#     raw, 
#     binning, 
#     ridge_size, 
#     thresh_coeff, 
#     small_cell_cutoff,
#     large_cell_cutoff,
#     remove_border_cells=remove_border_cells, 
#     parallel=True
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s')

#%% Display -------------------------------------------------------------------

# # All
# viewer = napari.Viewer()
# viewer.add_image(output_dict['wat'])
# viewer.add_labels(output_dict['labels'])
# viewer.add_labels(output_dict['markers'])
# viewer.add_image(output_dict['mask'])
# viewer.add_image(output_dict['ridges'])
# viewer.add_image(output_dict['rsize'])
# viewer.grid.enabled = True

# # Overlay
# viewer = napari.Viewer()
# viewer.add_image(
#     output_dict['rsize']
#     )
# viewer.add_image(
#     output_dict['wat'],
#     colormap='red',
#     blending='additive',
#     )