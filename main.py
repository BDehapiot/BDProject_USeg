#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.transform import resize
from skimage.filters import gaussian, sato
from skimage.restoration import rolling_ball

#%% Initialize ----------------------------------------------------------------

data_path = Path('data')
raw_name = "13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif"
raw = io.imread(data_path / raw_name)

#%% Parameters ----------------------------------------------------------------

binning = 2
ridge_size = 3

#%% Functions -----------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

def _get_wat(img):

    # Resize (acc. to binning)  
    rsize = resize(img, (
        int(img.shape[0] // binning), 
        int(img.shape[1] // binning)),
        preserve_range=True, 
        anti_aliasing=True,
        )    
    
    # Subtract background
    rsize -= rolling_ball(
        gaussian(rsize, sigma=ridge_size // 2), radius=ridge_size * 2,
        )    
    
    # Apply ridge filter 
    ridges = sato(
        rsize, sigmas=ridge_size, mode='reflect', black_ridges=False,
        )
    
    return rsize, ridges

outputs = Parallel(n_jobs=-1)(
    delayed(_get_wat)(img)
    for img in raw
    )
