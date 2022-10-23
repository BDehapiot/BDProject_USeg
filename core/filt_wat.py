#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels
from skimage.morphology import binary_dilation, square, disk

from tools.conn import labconn
from tools.nan import nanreplace, nanoutliers

#%% To do list ----------------------------------------------------------------

#%% Function (filt_wat) -------------------------------------------------------

#%% Run -----------------------------------------------------------------------

# File name
rsize_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8_rsize.tif'
wat_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8_wat.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8_wat.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8_wat.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8_wat.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop_wat.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8_wat.tif'

# Parameters

# Open data
rsize = io.imread(Path('../data/', rsize_name))
wat = io.imread(Path('../data/', wat_name))

# # Process data (filt_wat)
# start = time.time()
# print('filt_wat')

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

#%% Tests ---------------------------------------------------------------------

rsize = rsize[0,...]
wat = wat[0,...]
wat = wat.astype('bool')

start = time.time()
print('Get vertices')

# Get vertices
vertices = labconn(wat, labels=None, conn=2) > 2

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Get bounds & endpoints')

# Get bounds & endpoints
bounds = wat.copy()
bounds[binary_dilation(vertices, square(3)) == 1] = 0
endpoints = wat ^ bounds ^ vertices

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Label bounds')

# Label bounds
bound_labels = label(bounds, connectivity=2).astype('float')
bound_labels[endpoints == 1] = np.nan
bound_labels = nanreplace(bound_labels, 3, 'max')
bound_labels = bound_labels.astype('int')

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Label small bounds ')

# Label small bounds 
small_bounds = wat ^ (bound_labels > 0) ^ vertices
small_bound_labels = label(small_bounds, connectivity=2)
small_bound_labels = small_bound_labels + np.max(bound_labels)
small_bound_labels[small_bound_labels == np.max(bound_labels)] = 0
bound_labels = bound_labels + small_bound_labels

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Get bound info ')

rsize_bg = rsize.copy().astype('float')
rsize_bg[binary_dilation(wat, footprint=disk(2))!=0] = np.nan
# rsize_bg = nanreplace(rsize_bg, 7, 'mean')
# rsize_bg = gaussian(rsize_bg, 2)

# rsize_bg[expanded_bound_labels!=0] = np.nan
# rsize_bg = nanoutliers(rsize_bg, 3, 'mean', sd_thresh=1)
# rsize_bg = nanreplace(rsize_bg, 5, 'mean')


# expanded_bound_labels = expand_labels(bound_labels, distance=1)

# def std(region, intensities):
#     return np.std(intensities[region], ddof=1)

# props = regionprops(
#     expanded_bound_labels,
#     intensity_image=rsize,
#     extra_properties=[std]
#     )

# info = pd.DataFrame(
#     ([(i.label, i.area, i.intensity_mean, i.std) for i in props]),
#     columns = ['label', 'area', 'mean', 'std']
#     )

end = time.time()
print(f'  {(end-start):5.3f} s')


#%% Display -------------------------------------------------------------------

# Bounds, vertices & endpoints
# viewer = napari.Viewer()
# viewer.add_image(bounds, colormap='blue')
# viewer.add_image(vertices, blending='additive')
# viewer.add_image(endpoints, blending='additive', colormap='red')

# # Bound labels
# viewer = napari.Viewer()
# viewer.add_labels(bound_labels)
# viewer.add_labels(small_bound_labels)

# Current
viewer = napari.Viewer()
viewer.add_image(rsize_bg)
