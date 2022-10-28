#%% 

import numpy as np

#%% rwhere

def rwhere(labels, val):
    
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

    lin_idx = np.where(labels.ravel() == val)
    idx = np.unravel_index(lin_idx, labels.shape)
    
    return idx

#%% rprops

def rprops(labels):
    
    rlabels = labels.ravel()
    sort = np.argsort(rlabels)
    sort_labels = rlabels[sort]
    lab, lab_start, count = np.unique(
        sort_labels, return_index=True, return_counts=True)
    lin_idx = np.split(sort, lab_start[1:])
    idx = [np.unravel_index(lin_idx, labels.shape) for lin_idx in lin_idx]
    
    return idx, lab, count

#%% Run -----------------------------------------------------------------------
