#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from skimage.measure import label
from joblib import Parallel, delayed 
from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.segmentation import watershed, clear_border
from skimage.filters import sato, threshold_li, gaussian
from skimage.morphology import binary_dilation, remove_small_objects, square, disk

from tools.idx import rprops
from tools.conn import labconn
from tools.nan import nanreplace

#%% Pre-processing ------------------------------------------------------------

def get_watershed(
        raw, 
        binning, 
        ridge_size, 
        thresh_coeff, 
        small_cell_cutoff, 
        large_cell_cutoff, 
        remove_border_cells,
        parallel=False
        ):
    
    # Nested function ---------------------------------------------------------
            
    def _get_watershed(raw):
        
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
        
        # Get markers
        markers = label(np.invert(mask), connectivity=1)
        
        # Get marker properties
        idx, lab, count = rprops(markers)

        # Filter cell according to area
        valid = np.zeros_like(count)
        for i in range (len(idx)):
            if count[i] < small_cell_cutoff: 
                valid[i] = 1 
                markers[idx[i]] = 0 # Remove small cells
            if count[i] > large_cell_cutoff: 
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
        
        # Get rsize_norm
        temp_mask = binary_dilation(labels > 0, footprint=disk(ridge_size*2))
        temp_mask = gaussian(temp_mask, sigma=ridge_size*2)
        temp_blur = gaussian(rsize, sigma=ridge_size*6)
        np.seterr(divide='ignore', invalid='ignore')
        rsize_norm = (rsize / temp_blur) * temp_mask
        rsize_norm = gaussian(rsize_norm, sigma=ridge_size//2)

        # Get bound properties
        idx, lab, count = rprops(bound_labels)

        # Get rsize_norm intensities with bounds  
        temp_int = [np.mean(rsize_norm[idx[i]]) for i in range(len(idx))]
        bound_int = np.zeros_like(rsize)
        for i in range(1,len(idx)):
            bound_int[idx[i]] = temp_int[i]
        
        return rsize, ridges, mask, markers, labels, wat, vertices, bound_labels, rsize_norm, bound_int
    
    # Run ---------------------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
    
    # Adjust parameters to binning
    ridge_size = ridge_size/binning
    small_cell_cutoff = small_cell_cutoff/np.square(binning)
    large_cell_cutoff = large_cell_cutoff/np.square(binning)
    
    if parallel:
    
        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_get_watershed)(
                raw,
                )
            for raw in raw
            )
        
    else:
        
        # Run serial
        output_list = [_get_watershed(
                raw,
                )
            for raw in raw
            ]
        
    # Extract output dictionary
    output_dict = {
        'rsize': np.stack(
            [data[0] for data in output_list], axis=0).squeeze(),
        'ridges': np.stack(
            [data[1] for data in output_list], axis=0).squeeze(),
        'mask': np.stack(
            [data[2] for data in output_list], axis=0).squeeze(),
        'markers': np.stack(
            [data[3] for data in output_list], axis=0).squeeze(),
        'labels': np.stack(
            [data[4] for data in output_list], axis=0).squeeze(),
        'wat': np.stack(
            [data[5] for data in output_list], axis=0).squeeze(),
        'vertices': np.stack(
            [data[6] for data in output_list], axis=0).squeeze(),
        'bound_labels': np.stack(
            [data[7] for data in output_list], axis=0).squeeze(),
        'rsize_norm': np.stack(
            [data[8] for data in output_list], axis=0).squeeze(),
        'bound_int': np.stack(
            [data[9] for data in output_list], axis=0).squeeze(),
        }
    
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
ridge_size = 4
thresh_coeff = 0.5
small_cell_cutoff = 100
large_cell_cutoff = 20000
remove_border_cells = True

# Open data
raw = io.imread(Path('../data/', raw_name))

# -----------------------------------------------------------------------------

# Get watershed
start = time.time()
print('Get watershed')

output_dict = get_watershed(
    raw, 
    binning, 
    ridge_size, 
    thresh_coeff, 
    small_cell_cutoff, 
    large_cell_cutoff, 
    remove_border_cells,
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

# # All
# viewer = napari.Viewer()
# viewer.add_image(output_dict['rsize'])
# viewer.add_image(output_dict['ridges'])
# viewer.add_image(output_dict['mask'])
# viewer.add_labels(output_dict['markers'])
# viewer.add_labels(output_dict['labels'])
# viewer.add_image(output_dict['wat'])
# viewer.add_image(output_dict['vertices'])
# viewer.add_labels(output_dict['bound_labels'])
# viewer.add_image(output_dict['rsize_norm'])
# viewer.add_image(output_dict['bound_int'], colormap='inferno')
# viewer.grid.enabled = True

# # Overlay
# viewer = napari.Viewer()
# viewer.add_image(output_dict['rsize'], opacity=0.66)
# viewer.add_image(output_dict['wat'], blending='additive', colormap='red')

#%% Test ----------------------------------------------------------------------

bound_int_cutoff = 1.5

rsize = output_dict['rsize'][0]
labels = output_dict['labels'][0]
wat = output_dict['wat'][0]
rsize_norm = output_dict['rsize_norm'][0]
bound_labels = output_dict['bound_labels'][0]
vertices = output_dict['vertices'][0]

start = time.time()
print('test')

# Get bound properties
idx, lab, count = rprops(bound_labels)

# Get rsize_norm intensities with bounds  
temp_int = [np.mean(rsize_norm[idx[i]]) for i in range(len(idx))]
bound_int = np.zeros_like(rsize)
for i in range(1,len(idx)):
    bound_int[idx[i]] = temp_int[i]

# Remove weak bounds
temp1 = labels!=0
temp2 = 0 < bound_int < bound_int_cutoff


# wat_filt = wat.copy()
# if bound_int_cutoff > 0:
#     wat_filt[bound_int < bound_int_cutoff] = 0
#     wat_filt += vertices
    


end = time.time()
print(f'  {(end-start):5.3f} s')

# import matplotlib.pyplot as plt
# plt.hist(temp_int, bins = 50)

viewer = napari.Viewer()
viewer.add_image(temp1)
viewer.add_image(temp2)

# viewer.add_image(wat, colormap='red')
# viewer.add_image(wat_filt, blending='additive')