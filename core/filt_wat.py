#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

#%% To do list ----------------------------------------------------------------

#%% Function (filt_wat) -------------------------------------------------------

#%% Run -----------------------------------------------------------------------

# File name
wat_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8_wat.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8_wat.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8_wat.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8_wat.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop_wat.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8_wat.tif'

# Parameters

# Open data
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



#%% Display -------------------------------------------------------------------

# All
viewer = napari.Viewer()
viewer.add_image(wat)