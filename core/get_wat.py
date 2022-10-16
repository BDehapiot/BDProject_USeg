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
        
        # # Get watershed 
        # markers = label(np.invert(mask), connectivity=1)  
        # labels = watershed(
        #     ridges, markers, compactness=1, watershed_line=False
        #     ) 
                
        # # Remove border cells
        # if remove_border_cells:
        #     labels = clear_border(labels)
            
        # wat = labels == 0
        
        # # Remove small cells
        # if min_cell_size > 0:
        #     temp_mask = binary_dilation(labels > 0)
        #     temp_mask = remove_small_holes(
        #         temp_mask, 
        #         area_threshold=min_cell_size
        #         )
        #     labels = expand_labels(labels, distance=min_cell_size)
        #     labels[temp_mask == 0] = 0  
        
        # # # Remove isolated cells
        # temp_mask = labels > 0
        # temp_mask = remove_small_objects(temp_mask, min_size = max_size*0.01)
        # labels[temp_mask==0] = 0  
        
        # # Clean wat & vertices
        # temp_mask = binary_dilation(labels > 0)
        # wat[temp_mask == 0] = 0

        return rsize, ridges, mask, # labels, wat
    
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
        # 'labels': np.stack(
        #     [img[3] for img in output_list], axis=0).squeeze(),
        # 'wat':np.stack(
        #     [img[4] for img in output_list], axis=0).squeeze(),
        }
    
    return output_dict

#%% Run -----------------------------------------------------------------------

# File name
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'

# Parameters
binning = 2
ridge_size = 3
thresh_coeff = 0.5
min_cell_size = 0

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
    remove_border_cells=False, 
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

#%% Display -------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(output_dict['wat'])
# viewer.add_labels(output_dict['labels'])
# viewer.add_image(output_dict['mask'])
# viewer.add_image(output_dict['ridges'])
# viewer.add_image(output_dict['rsize'])
# viewer.grid.enabled = True

#%% Tests ---------------------------------------------------------------------

rsize = output_dict['rsize']
ridges = output_dict['ridges']
mask = output_dict['mask']
# labels = output_dict['labels']
# wat = output_dict['wat']

# Parameters
min_cell_size = 50/binning
remove_border_cells = True

start = time.time()
print('test')

# Option #1 -------------------------------------------------------------------

# Get watershed 
markers = label(np.invert(mask), connectivity=1)  
labels = watershed(
    ridges, markers, compactness=5, watershed_line=False
    ) 

# Remove small cells
if min_cell_size > 0:
    
    props = regionprops(labels)
    for i, prop in enumerate(props):
        if prop.area < min_cell_size:
            labels[labels==i+1] = 0
            
    labels = expand_labels(labels, distance=min_cell_size)

# Remove border cells
if remove_border_cells:
    labels = clear_border(labels)    
    
# Get boundaries
wat = find_boundaries(labels)
wat = skeletonize(wat, method='zhang')
labels = label(np.invert(wat), connectivity=1)

# Option #2 -------------------------------------------------------------------

# # Get watershed 
# markers = label(np.invert(mask), connectivity=1)  
# labels = watershed(
#     ridges, markers, compactness=1, watershed_line=True
#     ) 

end = time.time()
print(f'  {(end-start):5.3f} s')

# Display
viewer = napari.Viewer()
viewer.add_image(wat)
viewer.add_labels(labels)
# viewer.grid.enabled = True