#%% Imports -------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border
from skimage.filters import sato, threshold_li, gaussian
from skimage.morphology import binary_dilation, remove_small_objects, square, disk

#%% Initialize ----------------------------------------------------------------

data_path = Path('data')
raw_name = "13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif"
raw = io.imread(data_path / raw_name)

#%% Parameters ----------------------------------------------------------------

binning = 2
ridge_size = 3
thresh_coeff = 0.5

# Rescale parameters acc. to binning
ridge_size /= binning

#%% Functions -----------------------------------------------------------------

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
    
    def _get_wat(raw, frame):
        
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
        cell_info = pd.DataFrame(
            ([(frame, i.label, i.area, 0) for i in props]),
            columns = ['frame', 'idx', 'area', 'valid']
            )
        median = np.median(cell_info['area'])
        for i in range(len(cell_info)):
            
            # Detect & remove small cells
            if cell_info.loc[i,'area'] < median/small_cell_cutoff: 
                cell_info.loc[i,'valid'] = 1
                markers[markers==cell_info['idx'][i]] = 0  
            
            # Detect large cells
            if cell_info.loc[i,'area'] > median*large_cell_cutoff: 
                cell_info.loc[i,'valid'] = 2                
                
        # Get watershed labels
        labels = watershed(
            ridges, 
            markers, 
            compactness=10/binning, 
            watershed_line=True
            ) 

        # Remove large cells
        for idx in cell_info.loc[cell_info['valid'] == 2, 'idx']:
            labels[labels==idx] = 0
            
        # Remove border cells
        if remove_border_cells:
            labels = clear_border(labels)
            
        # Update cell info
        props = regionprops(labels)
        cell_info = pd.DataFrame(
            ([(frame, i.label, i.area) for i in props]),
            columns = ['frame', 'idx', 'area']
            )

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
        
        # Get background
        rsize_bg = rsize.copy().astype('float')
        rsize_bg[mask==1] = np.nan        
        rsize_bg = nanreplace(
            rsize_bg, int(np.ceil(ridge_size*4)//2*2+1), 'mean')

        # Get bound intensities
        props = regionprops(
            bound_labels, 
            intensity_image=(gaussian(rsize, ridge_size))
            )
        props_bg = regionprops(
            bound_labels, 
            intensity_image=(gaussian(rsize_bg, ridge_size))
            )
        bound_info = pd.DataFrame(([
            (frame, i.label, i.intensity_mean, j.intensity_mean) 
            for i, j in zip(props, props_bg)]),
            columns = ['frame', 'idx', 'bound', 'bg'],
            )
        bound_info['ratio'] = bound_info['bound']/bound_info['bg']        
        
        return rsize, ridges, mask, markers, labels, wat, vertices, bound_labels, cell_info, bound_info
    
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
                img, frame,
                )
            for frame, img in enumerate(raw)
            )
        
    else:
        
        # Run serial
        output_list = [_get_wat(
                img, frame,
                )
            for frame, img in enumerate(raw)
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
        'cell_info' : pd.concat(
            [(data[8]) for i, data in enumerate(output_list)]),
        'bound_info' : pd.concat(
            [(data[9]) for i, data in enumerate(output_list)]),        
        }
            
    return output_dict

#%% Execute -------------------------------------------------------------------

print("  get_wat   :", end='')
t0 = time.time()

outputs = Parallel(n_jobs=-1)(
    delayed(_get_wat)(img)
    for img in raw
    )
rsize = np.stack([data[0] for data in outputs])
ridges = np.stack([data[1] for data in outputs])
mask = np.stack([data[2] for data in outputs])
markers = np.stack([data[3] for data in outputs])

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(rsize)
viewer.add_image(ridges)
viewer.add_image(mask)
# viewer.add_image(markers)
