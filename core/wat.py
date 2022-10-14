#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

#%% Function 
# -----------------------------------------------------------------------------

def wat(raw):
    
    # --- Nest function ---
    
    def _wat(raw):
        
        
        
        return wat
    
    ''' --- Main function --- '''
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
        
    # 
        
    # Squeeze dimensions (if ndim == 2)    
    if ndim == 2:
        wat = wat.squeeze()
    
    return wat

#%% Run -----------------------------------------------------------------------

# Inputs
binning = 1

raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'

# Open data
raw = io.imread(Path('../data/', raw_name))

# Process