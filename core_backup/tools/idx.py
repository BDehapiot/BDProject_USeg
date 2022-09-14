#%% 

import numpy as np

#%% rwhere

def rwhere(img, val):
    
    ''' General description.
    
    Parameters
    ----------
    img : np.ndarray
        Description.
        
    val : float
        Description.
        
    Returns
    -------  
    idx : ???
        Description.
        
    Notes
    -----   
    
    '''

    lin_idx = np.where(img.ravel() == val)
    idx = np.unravel_index(lin_idx, img.shape)
    
    return idx
