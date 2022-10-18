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
from scipy.ndimage import binary_fill_holes
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border, expand_labels
from skimage.morphology import remove_small_objects, binary_dilation, remove_small_holes

#%% To do list ----------------------------------------------------------------

'''
- Extract advanced parameters
- Auto ridge_size

'''

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
                
        # Get markers
        markers = label(np.invert(mask), connectivity=1)  
        
        # Remove small and big objects
        props = regionprops(markers)
        areas = np.array([i.area for i in props])
        idx = np.array([i.label for i in props])
        med = np.median(areas)
        for i, area in enumerate(areas):
            if area < med/5: # Remove small objects
                markers[markers==idx[i]] = 0        
            # if area > med*10: # Remove big objects
            #     markers[markers==idx[i]] = 0
        
        # Get labels (with watershed)
        labels = watershed(
            ridges, 
            markers, 
            compactness=10/binning, 
            watershed_line=True
            ) 
                
        # Remove border cells
        if remove_border_cells:
            labels = clear_border(labels)
             
        # Get wat & update labels
        wat = labels == 0
        
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
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Parameters
binning = 2
ridge_size = 1.5
thresh_coeff = 0.5
min_cell_size = 100
remove_border_cells = True

# Open data
raw = io.imread(Path('../data/', raw_name))

# # Process data (get_wat)
# start = time.time()
# print('get_wat')

# output_dict = get_wat(
#     raw, 
#     binning, 
#     ridge_size, 
#     thresh_coeff, 
#     min_cell_size,
#     remove_border_cells=remove_border_cells, 
#     parallel=True
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s')

#%% Display -------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(output_dict['wat'])
# viewer.add_labels(output_dict['labels'])
# viewer.add_labels(output_dict['markers'])
# viewer.add_image(output_dict['mask'])
# viewer.add_image(output_dict['ridges'])
# viewer.add_image(output_dict['rsize'])
# viewer.grid.enabled = True

#%% Tests ---------------------------------------------------------------------

# rsize = output_dict['rsize']
# ridges = output_dict['ridges']
# mask = output_dict['mask']
# markers = output_dict['markers']
# labels = output_dict['labels']
# wat = output_dict['wat']

# t = -1
# rsize = output_dict['rsize'][t,...]
# ridges = output_dict['ridges'][t,...]
# mask = output_dict['mask'][t,...]
# markers = output_dict['markers'][t,...]
# labels = output_dict['labels'][t,...]
# wat = output_dict['wat'][t,...]

# raw = raw[-1,...]

# -----------------------------------------------------------------------------

start = time.time()
print('Resize (according to binning) ')

# Resize (according to binning)  
rsize = resize(raw, (
    int(raw.shape[0]//binning), 
    int(raw.shape[1]//binning)),
    preserve_range=True, 
    anti_aliasing=True,
    ) 

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Subtract background')

# Subtract background
rsize -= rolling_ball(
    gaussian(rsize, sigma=ridge_size//2), radius=ridge_size*2,
    ) 

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Apply ridge filter ')

# Apply ridge filter 
ridges = sato(
    rsize, sigmas=ridge_size, mode='reflect', black_ridges=False,
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Get mask & markers')

# Get mask
thresh = threshold_li(ridges, tolerance=1)
mask = ridges > thresh*thresh_coeff
mask = remove_small_objects(mask, min_size=np.sum(mask)*0.01)
markers = label(np.invert(mask), connectivity=1)  

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Filter objects (according to labels area)')

# Filter objects (according to labels area)
props = regionprops(markers)
idxs = np.array([i.label for i in props])
areas = np.array([i.area for i in props])
median = np.median(areas)
for idx, area in zip(idxs, areas):
    if area < median/5: # Remove small objects
        markers[markers==idx] = 0 
    if area > median*10: # Remove large objects
        markers[markers==idx] = np.max(markers) + 1 
        
end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

start = time.time()
print('Get watershed labels & line')

# Get watershed labels & line
labels = watershed(
    ridges, 
    markers, 
    compactness=10/binning, 
    watershed_line=True
    ) 

if remove_border_cells:
    labels = clear_border(labels, bgval=np.max(labels)+1)

line = labels == 0
             
end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

# Display
viewer = napari.Viewer()
viewer.add_image(rsize)
viewer.add_image(ridges)
viewer.add_image(mask)
viewer.add_labels(markers)
viewer.add_labels(labels)
viewer.add_image(line)
viewer.grid.enabled = True

#%% Tests 

