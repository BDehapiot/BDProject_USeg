#%%

import time
import numpy as np

#%%

from tools.piv import bd_openpiv
from functions import preprocess, watseg, process_bounds, display_bounds

#%% process

def process(
        raw,
        rsize_factor,
        time_window,
        ridge_size,
        thresh_coeff,
        thresh_min_size,
        piv,
        win_size
        ):
    
    ''' General description.
    
    Parameters
    ----------
    raw : np.ndarray (uint8, uint16)
        Description.
        
    rsize_factor : int
        Description.
        
    time_window : int
        Description.        
    
    ridge_size : float
        Description.
        
    thresh_coeff : float
        Description.    
        
    thresh_min_size : int    
        Description.
        
    win_size : int
        Description.
        
    piv  : bool
        Description.
    
    Returns
    -------  
    rsize : np.ndarray (float)
        Description.
        
    ridges : np.ndarray (float)
        Description.   
        
    mask : np.ndarray (bool)
        Description.  
            
    markers : np.ndarray (uint16)
        Description.  
                
    labels : np.ndarray (uint16)
        Description.  
        
    wat : np.ndarray (bool)
        Description.
        
    vector_field : np.ndarray (uint8)
        Description.  
        
    bound_labels : np.ndarray (uint16)
        Description.  
        
    Notes
    -----   
    
    '''
    
    # Preprocess ..............................................................
    
    start = time.time()
    print('Preprocess')
        
    rsize, ridges = preprocess(
        raw, 
        rsize_factor, 
        ridge_size, 
        parallel=True
        )
    
    end = time.time()
    print(f'  {(end - start):5.3f} s')
    
    # Watershed ...............................................................
    
    start = time.time()
    print('Watershed')
    
    mask, markers, labels, wat = watseg(
        ridges, 
        thresh_coeff, 
        thresh_min_size, 
        parallel=True
        )
    
    end = time.time()
    print(f'  {(end - start):5.3f} s')
    
    # PIV .....................................................................
    
    if piv:
    
        start = time.time()
        print('PIV')
    
        u, v, vector_field = bd_openpiv(
            rsize,
            time_window, 
            win_size,
            mask=labels>0,
            smooth_size=3,
            smooth_method='median',
            missing_frames='nearest', 
            bsize=True,
            display=True,
            parallel=True
            )
        
        end = time.time()
        print(f'  {(end - start):5.3f} s')   
        
    else:
        
        u = np.zeros(rsize.shape)
        v = np.zeros(rsize.shape)
        vector_field = np.zeros(rsize.shape)
    
    # Process bounds ..........................................................  

    start = time.time()
    print('Process bounds')

    bound_labels, bound_norm, bound_edm, bound_data = process_bounds(
        rsize, 
        labels, 
        wat, 
        u, v, 
        time_window,
        ridge_size,
        parallel=True
        )
    
    end = time.time()
    print(f'  {(end - start):5.3f} s')  

    # .........................................................................   

    # Extract outputs in a dictionnary
    outputs = {
        'rsize' : rsize,
        'ridges' : ridges,
        'mask' : mask,
        'markers' : markers,
        'labels' : labels,
        'wat' : wat,
        'u' : u, 'v' : v, 
        'vector_field' : vector_field,
        'bound_labels' : bound_labels,
        'bound_norm' : bound_norm,
        'bound_edm' : bound_edm,
        'bound_data' : bound_data
        }
    
    # ......................................................................... 

    return outputs
