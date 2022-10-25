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
from skimage.morphology import binary_dilation, remove_small_objects, square

from tools.conn import labconn
from tools.nan import nanreplace

#%% To do list ----------------------------------------------------------------

'''
- Handle bit depth (convert to uint8)
- Extract advanced parameters
- Get an auto ridge_size
- back size wat as an option

'''

#%% Function (get_bounds) --------------------------------------------------------

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
        
        # Test
        bound_ratio = np.zeros_like(rsize)
        
        
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

#%% Function (filt_bounds) --------------------------------------------------------


#%% Run -----------------------------------------------------------------------

# File name
raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Parameters
binning = 2
ridge_size = 3
thresh_coeff = 0.5
small_cell_cutoff = 3
large_cell_cutoff = 10
remove_border_cells = False

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
    small_cell_cutoff,
    large_cell_cutoff,
    remove_border_cells=remove_border_cells, 
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s')

#%% Test ----------------------------------------------------------------------

# start = time.time()
# print('filt_wat')

# bound_info = output_dict['bound_info']
# bound_ratio_cutoff = 2.0
# test = bound_info.loc[bound_info['ratio'] <= 2, ['frame', 'idx', 'ratio']]

# end = time.time()
# print(f'  {(end-start):5.3f} s')

#%% Display -------------------------------------------------------------------

# # All
# viewer = napari.Viewer()
# viewer.add_labels(output_dict['bound_labels'])
# viewer.add_image(output_dict['vertices'])
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

#%% Save ----------------------------------------------------------------------

# # Save rsize
# io.imsave(
#     Path('../data/', raw_name.replace('.tif', '_rsize.tif')),
#     output_dict['rsize'].astype('uint8'),
#     check_contrast=False,
#     )

# # Save wat
# io.imsave(
#     Path('../data/', raw_name.replace('.tif', '_wat.tif')),
#     output_dict['wat'].astype('uint8')*255,
#     check_contrast=False,
#     )