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

from tools.idx import rprops
from tools.conn import labconn
from tools.nan import nanreplace

#%% Pre-processing ------------------------------------------------------------

def pre_processing(raw, binning, ridge_size, thresh_coeff, parallel=False):
    
    # Nested function ---------------------------------------------------------
            
    def _pre_processing(raw):
        
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
        
        return rsize, ridges, mask
    
    # Run ---------------------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_pre_processing)(
                img,
                )
            for img in raw
            )
        
    else:
        
        # Run serial
        output_list = [_pre_processing(
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

#%% Get watershed -------------------------------------------------------------

def get_watershed(output_dict, parallel=False):
    
    # Nested function ---------------------------------------------------------
    
    def _get_watershed(ridges, mask):
        
        # Get markers
        markers = label(np.invert(mask), connectivity=1)
        
        # Get marker properties
        idx, lab, count = rprops(markers)

        # Filter cell according to area
        valid = np.zeros_like(count)
        median = np.median(count)
        for i in range (len(idx)):
            if count[i] < median/small_cell_cutoff: 
                valid[i] = 1 
                markers[idx[i]] = 0 # Remove small cells
            if count[i] > median*large_cell_cutoff: 
                valid[i] = 2 # Detect large cells    

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
                markers[markers==lab[i]] = 0
                labels[labels==lab[i]] = 0

        # Remove border cells
        if remove_border_cells:
            markers = clear_border(markers)
            labels = clear_border(labels)
            
        # Get watershed wat
        wat = labels == 0
        temp = np.invert(wat)
        temp = binary_dilation(temp)
        wat = np.minimum(wat, temp)
        
        return markers, labels, wat
    
    # Run ---------------------------------------------------------------------
        
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
        [data[0] for data in output_list], axis=0).squeeze()
    output_dict['labels'] = np.stack(
        [data[1] for data in output_list], axis=0).squeeze()
    output_dict['wat'] = np.stack(
        [data[2] for data in output_list], axis=0).squeeze()
    
    return output_dict

#%% Get bounds ----------------------------------------------------------------

def get_bounds(output_dict, parallel=False):
    
    # Nested function ---------------------------------------------------------
    
    def _get_bounds(labels, wat):
               
        # Get vertices
        vertices = labconn(wat, labels=labels, conn=2) > 2
        
        # Get bounds & endpoints
        bounds = wat.copy()
        bounds[binary_dilation(vertices, square(3)) == 1] = 0
        endpoints = wat ^ bounds ^ vertices
        
        # Label bounds
        bound_labels = label(bounds, connectivity=2).astype('float')
        bound_labels[endpoints == 1] = np.nan
        bound_labels = nanreplace(bound_labels, 3, 'max')
        bound_labels = bound_labels.astype('int')
        small_bounds = wat ^ (bound_labels > 0) ^ vertices
        small_bound_labels = label(small_bounds, connectivity=2)
        small_bound_labels = small_bound_labels + np.max(bound_labels)
        small_bound_labels[small_bound_labels == np.max(bound_labels)] = 0
        bound_labels = bound_labels + small_bound_labels
        
        return vertices, bound_labels
    
    # Run ---------------------------------------------------------------------
        
    labels = output_dict['labels']
    wat = output_dict['wat']
    
    # Add one dimension (if ndim == 2)
    ndim = (labels.ndim)        
    if ndim == 2:
        labels = labels.reshape((1, labels.shape[0], labels.shape[1]))  
        wat = wat.reshape((1, wat.shape[0], wat.shape[1])) 
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_get_bounds)(
                labels, wat,
                )
            for labels, wat
            in zip(labels, wat)
            )
        
    else:
        
        # Run serial
        output_list = [_get_bounds(
                labels, wat,
                )
            for labels, wat
            in zip(labels, wat)
            ]
        
    # Extract output dictionary
    output_dict['vertices'] = np.stack(
        [data[0] for data in output_list], axis=0).squeeze()
    output_dict['bound_labels'] = np.stack(
        [data[1] for data in output_list], axis=0).squeeze()
    
    return output_dict

#%% Run -----------------------------------------------------------------------

# File name
raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Parameters
binning = 2
ridge_size = 4/binning
thresh_coeff = 0.25
small_cell_cutoff = 10
large_cell_cutoff = 10
remove_border_cells = True

# Open data
raw = io.imread(Path('../data/', raw_name))

# -----------------------------------------------------------------------------

# Pre-processing
start = time.time()
print('Pre-processing')

output_dict = pre_processing(
    raw, 
    binning, 
    ridge_size, 
    thresh_coeff, 
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

# Get watershed
start = time.time()
print('Get watershed')

output_dict = get_watershed(
    output_dict,
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

# Get bounds
start = time.time()
print('Get bounds')

output_dict = get_bounds(
    output_dict,
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(output_dict['rsize'])
viewer.add_image(output_dict['ridges'])
viewer.add_image(output_dict['mask'])
viewer.add_labels(output_dict['markers'])
viewer.add_labels(output_dict['labels'])
viewer.add_image(output_dict['wat'])
viewer.add_image(output_dict['vertices'])
viewer.add_labels(output_dict['bound_labels'])
viewer.grid.enabled = True

#%% Test ----------------------------------------------------------------------

# rsize = output_dict['rsize']
# ridges = output_dict['ridges']
# mask = output_dict['mask']
# markers = output_dict['markers']
# labels = output_dict['labels']
# bound_labels = output_dict['bound_labels']

rsize = output_dict['rsize'][0]
ridges = output_dict['ridges'][0]
mask = output_dict['mask'][0]
markers = output_dict['markers'][0]
labels = output_dict['labels'][0]
bound_labels = output_dict['bound_labels'][0]

# start = time.time()
# print('test #1')

# # Get temp_mask
# temp_mask = labels > 0
# temp_mask = binary_dilation(
#     temp_mask, footprint=disk(ridge_size)
#     )

# # Get bg_blur (with a nan ignoring Gaussian blur)
# temp1 = rsize.astype('float')
# temp1[temp_mask == 0] = np.nan

# temp2 = temp1.copy()
# temp2[np.isnan(temp2)] = 0
# temp2_blur = gaussian(temp2, sigma=ridge_size*10)

# temp3 = 0 * temp1.copy() + 1
# temp3[np.isnan(temp3)] = 0
# temp3_blur = gaussian(temp3, sigma=ridge_size*10)

# np.seterr(divide='ignore', invalid='ignore')
# bg_blur = temp2_blur / temp3_blur

# # Get rsize_blur (divided by bg_blur)
# bound_norm = gaussian(rsize, sigma=ridge_size//2)
# bound_norm = bound_norm / bg_blur *1.5

# # Get bound properties
# idx, lab, count = rprops(bound_labels)

# # Get rsize intensities with bounds  
# temp_int = [np.mean(bound_norm[idx[i]]) for i in range(len(idx))]
# bound_int = np.zeros_like(rsize)
# for i in range(1,len(idx)):
#     bound_int[idx[i]] = temp_int[i]

# end = time.time()
# print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

# start = time.time()
# print('test #2')

# # Get temp_mask
# temp_mask = binary_dilation(
#     labels > 0, footprint=disk(ridge_size)
#     )

# temp1 = rsize.copy()
# temp1[temp_mask == 0] = 0
# temp2 = temp_mask.copy().astype('float')

# temp1_blur = gaussian(temp1, sigma=ridge_size*5)
# temp2_blur = gaussian(temp2, sigma=ridge_size*5)

# np.seterr(divide='ignore', invalid='ignore')
# bg_blur = temp1_blur / temp2_blur

# # Get rsize_blur (divided by bg_blur)
# bound_norm = gaussian(rsize, sigma=ridge_size//2)
# bound_norm = bound_norm / bg_blur

# end = time.time()
# print(f'  {(end-start):5.3f} s')

# viewer = napari.Viewer()
# viewer.add_image(rsize)
# viewer.add_image(temp_mask)
# viewer.add_image(temp1)
# viewer.add_image(temp2)
# viewer.add_image(temp1_blur)
# viewer.add_image(temp2_blur)
# viewer.add_image(bg_blur)
# viewer.add_image(bound_norm)
# viewer.grid.enabled = True

# -----------------------------------------------------------------------------

start = time.time()
print('test #3')

temp_mask = binary_dilation(labels > 0, footprint=disk(ridge_size*2))
temp_mask = gaussian(temp_mask, sigma=ridge_size*2)
temp_blur = gaussian(rsize, sigma=ridge_size*6)
rsize_norm = (rsize / temp_blur) * temp_mask
rsize_norm = gaussian(rsize_norm, sigma=ridge_size//2)

# Get bound properties
idx, lab, count = rprops(bound_labels)

# Get rsize intensities with bounds  
temp_int = [np.mean(rsize_norm[idx[i]]) for i in range(len(idx))]
bound_int = np.zeros_like(rsize)
for i in range(1,len(idx)):
    bound_int[idx[i]] = temp_int[i]

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('test #4')

temp_blur = gaussian(ridges, sigma=ridge_size*6)
ridge_norm = (ridges / temp_blur)

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(temp_mask)
viewer.add_image(rsize)
viewer.add_image(temp_blur)
viewer.add_image(rsize_norm)
viewer.add_image(bound_int, colormap='inferno')
viewer.add_image(ridge_norm)
viewer.grid.enabled = True