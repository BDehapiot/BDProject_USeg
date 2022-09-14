#%% Imports

from skimage import io
from pathlib import Path

from functions import preprocess

#%% Select data

# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_Lite_uint8.tif"
# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif"
raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif"
# raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite_uint8.tif"
# raw_name  = "17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif"
# raw_name  = "Disc_ex_vivo_118hAEL_disc2_uint8.tif"
# raw_name  = "Disc_Fixed_118hAEL_disc04_uint8.tif"

#%% Open data

# Create paths
root_path = Path(__file__).parents[1]
raw_path = Path(root_path / 'data' / raw_name)

# Open data
raw = io.imread(raw_path) 

# Add one dimension (if ndim = 2)
ndim = (raw.ndim)        
if ndim == 2:
    raw = raw.reshape((1, raw.shape[0], raw.shape[1]))
    
#%%

rsize, ridges = preprocess(raw, rsize_factor, ridge_size, parallel=True)

