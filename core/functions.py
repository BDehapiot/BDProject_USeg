#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 
from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border
from skimage.filters import sato, threshold_li, gaussian
from skimage.morphology import binary_dilation, remove_small_objects, square, disk, white_tophat

from tools.conn import labconn
from tools.nan import nanreplace

#%% Preprocess ----------------------------------------------------------------

def preprocess(raw, binning, ridge_size, thresh_coeff, parallel=False):
    
    # Nested function ---------------------------------------------------------
            
    def _preprocess(raw):
        
        # Resize (according to binning)  
        rsize = resize(raw, (
            int(raw.shape[0]//binning), 
            int(raw.shape[1]//binning)),
            preserve_range=True, 
            anti_aliasing=True,
            )        
                
        # # Subtract background
        # temp = white_tophat(rsize, footprint=(disk(ridge_size*2)))
        # background = gaussian(rsize-temp, sigma=ridge_size*2, preserve_range=True)
        # rsize = rsize / background
        
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
        
        return rsize, ridges, mask
    
    # Run ---------------------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_preprocess)(
                img,
                )
            for img in raw
            )
        
    else:
        
        # Run serial
        output_list = [_preprocess(
                img,
                )
            for img in raw
            ]
        
    # Extract output dictionary
    output_dict = {
        'rsize': np.stack(
            [data[0] for data in output_list], axis=0).squeeze(),
        'ridges': np.stack(
            [data[1] for data in output_list], axis=0).squeeze(),
        'mask': np.stack(
            [data[2] for data in output_list], axis=0).squeeze(),
        }
    
    return output_dict

#%% Watershed -----------------------------------------------------------------

def get_watershed(output_dict, parallel=False):
    
    # Nested function ---------------------------------------------------------
    
    def _get_watershed(ridges, mask):
        
        # Get markers
        markers = label(np.invert(mask), connectivity=1)
        
        return markers
    
    # Run function ---------------------------------------------------------
        
    ridges = output_dict['ridges']
    mask = output_dict['mask']
    
    # Add one dimension (if ndim == 2)
    ndim = (ridges.ndim)        
    if ndim == 2:
        ridges = ridges.reshape((1, ridges.shape[0], ridges.shape[1]))  
        mask = mask.reshape((1, mask.shape[0], mask.shape[1])) 
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_get_watershed)(
                ridges, mask,
                )
            for ridges, mask 
            in zip(ridges, mask)
            )
        
    else:
        
        # Run serial
        output_list = [_get_watershed(
                ridges, mask,
                )
            for ridges, mask 
            in zip(ridges, mask)
            ]
        
    # Extract output dictionary
    output_dict['markers'] = np.stack(
        [data for data in output_list], axis=0).squeeze()
    
    return output_dict

#%% Run -----------------------------------------------------------------------

# File name
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
raw_name = 'Disc_Fixed_118hAEL_disc04_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Parameters
binning = 2
ridge_size = 4/binning
thresh_coeff = 0.5
small_cell_cutoff = 3
large_cell_cutoff = 10
remove_border_cells = False

# Open data
raw = io.imread(Path('../data/', raw_name))

# Preprocessing
start = time.time()
print('preprocess')

output_dict = preprocess(
    raw, 
    binning, 
    ridge_size, 
    thresh_coeff, 
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# Watershed
start = time.time()
print('watershed')

output_dict = get_watershed(
    output_dict,
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

viewer = napari.Viewer()
viewer.add_image(output_dict['rsize'])
viewer.add_image(output_dict['ridges'])
viewer.add_image(output_dict['mask'])
viewer.add_labels(output_dict['markers'])
viewer.grid.enabled = True

#%% Test ----------------------------------------------------------------------

# markers = output_dict['markers'][0]
# ridges = output_dict['ridges'][0]

markers = output_dict['markers']
ridges = output_dict['ridges']

start = time.time()
print('test')

# 
sort = np.argsort(markers.ravel())
sort_markers = markers.ravel()[sort]
lab, lab_start, count = np.unique(
    sort_markers, return_index=True, return_counts=True)
lin_idx = np.split(sort, lab_start[1:]) # does it apply for all?
idx = [np.unravel_index(lin_idx, markers.shape) for lin_idx in lin_idx]

# -----------------------------------------------------------------------------

valid = np.zeros_like(count)
median = np.median(count)
for i in range (len(idx)):
    if count[i] < median/small_cell_cutoff: 
        valid[i] = 1
        markers[idx[i]] = 0 # Remove small cells
    if count[i] > median*large_cell_cutoff: 
        valid[i] = 2    
        
# -----------------------------------------------------------------------------

# Get watershed labels
labels = watershed(
    ridges, 
    markers, 
    compactness=10/binning, 
    watershed_line=True
    ) 

# Remove large cells
for i in range (len(idx)):
    if valid[i] == 2:
        labels[labels==lab[i]] = 0

# Remove border cells
if remove_border_cells:
    labels = clear_border(labels)
    
# Get watershed wat
wat = labels == 0
temp = np.invert(wat)
temp = binary_dilation(temp)
wat = np.minimum(wat, temp)
        
end = time.time()
print(f'  {(end-start):5.3f} s')

viewer = napari.Viewer()
viewer.add_image(wat)
viewer.add_labels(markers)
viewer.add_labels(labels)
viewer.grid.enabled = True

