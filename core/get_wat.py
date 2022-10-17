#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 
from skimage.transform import resize
from skimage.filters import threshold_li
from skimage.filters import sato, gaussian
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border, expand_labels, find_boundaries
from skimage.morphology import remove_small_objects, binary_dilation, remove_small_holes, skeletonize

#%% Function (get_wat) --------------------------------------------------------

def get_wat(
        raw, 
        binning, 
        ridge_size, 
        thresh_coeff, 
        min_cell_size, 
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
        thresh = threshold_li(ridges)
        mask = ridges > thresh*thresh_coeff
        
        # Remove small objects (< 1% of max_size)
        props = regionprops(label(mask))
        max_size = max([i.area for i in props])
        mask = remove_small_objects(mask, min_size=max_size*0.01)
                
        # Get markers & labels
        markers = label(np.invert(mask), connectivity=1)  
        markers = remove_small_objects(markers, min_size=min_cell_size)
        labels = watershed(
            ridges, markers, compactness=10/binning, watershed_line=True
            ) 
        
        # Remove border cells
        if remove_border_cells:
            labels = clear_border(labels)
            
        # Get wat & updated labels
        wat = labels == 0
        temp_wat = binary_dilation(wat == 0)
        wat = np.invert(wat^temp_wat)
        labels = label(np.invert(wat), connectivity=1)
                
        return rsize, ridges, mask, markers, labels, wat
    
    # Run ---------------------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
        
    # Adjust parameters according to binning
    ridge_size = ridge_size/binning
    min_cell_size = min_cell_size/binning
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
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
raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'

# Parameters
binning = 2
ridge_size = 3
thresh_coeff = 0.5
min_cell_size = 25
remove_border_cells = True

# Open data
raw = io.imread(Path('../data/', raw_name))

# Process data (get_wat)
start = time.time()
print('get_wat')

output_dict = get_wat(
    raw, 
    binning, 
    ridge_size, 
    thresh_coeff, 
    min_cell_size,
    remove_border_cells=True, 
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(output_dict['wat'])
viewer.add_labels(output_dict['labels'])
viewer.add_labels(output_dict['markers'])
viewer.add_image(output_dict['mask'])
viewer.add_image(output_dict['ridges'])
viewer.add_image(output_dict['rsize'])
viewer.grid.enabled = True

#%% Tests ---------------------------------------------------------------------

# rsize = output_dict['rsize']
# ridges = output_dict['ridges']
# mask = output_dict['mask']
# markers = output_dict['markers']
# labels = output_dict['labels']
# wat = output_dict['wat']

# start = time.time()
# print('test')

# Option #1 -------------------------------------------------------------------

# wat = labels == 0
# temp_wat = wat == 0
# temp_wat = binary_dilation(temp_wat)
# wat = np.invert(wat^temp_wat)
# wat = label(np.invert(wat), connectivity=1)

# Option #2 -------------------------------------------------------------------

# end = time.time()
# print(f'  {(end-start):5.3f} s')

# Display
# viewer = napari.Viewer()
# viewer.add_labels(markers)
# viewer.add_labels(labels)
# viewer.add_labels(wat)
# viewer.add_image(temp_wat)
# viewer.grid.enabled = True