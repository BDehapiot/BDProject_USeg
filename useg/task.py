#%%

import time
import napari
import numpy as np

#%%

from useg.core import preprocess, watseg, process_bounds
from useg.tools.piv import bd_openpiv

#%% preview

def preview(
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
    
    start = time.time()
    print('Preview')
    
    # Extract mid sequence for preview
    time_mid = raw.shape[0]//2
    time_step = time_window//2
    preview_raw = raw[time_mid-time_step:time_mid+time_step+1,...]
    
    # Preprocess
    rsize, ridges = preprocess(
        preview_raw, rsize_factor, ridge_size, parallel=False)
       
    # Watershed segmentation
    mask, markers, labels, wat = watseg(
        ridges, thresh_coeff, thresh_min_size, parallel=False)
    
    # Particle image velocimetry (PIV)
    if piv:

        u, v, vector_field = bd_openpiv(
            rsize,
            time_window, 
            win_size, 
            mask=labels>0,
            parallel=False
            )
    
    # Label bounds
    bound_labels = label_bounds(wat, labels, parallel=False)
       
    end = time.time()
    print(f'  {(end - start):5.3f} s')
    
    # .........................................................................
    
    if piv:
    
        return rsize, ridges, mask, markers, labels, wat, vector_field, bound_labels
    
    else:
        
        return rsize, ridges, mask, markers, labels, wat, bound_labels

#%% Process

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
    
    start = time.time()
    print('Preprocess')
        
    # Preprocess
    rsize, ridges = preprocess(
        raw, rsize_factor, ridge_size, parallel=True)
    
    end = time.time()
    print(f'  {(end - start):5.3f} s')
    
    # .........................................................................
    
    start = time.time()
    print('Watershed')
    
    # Watershed segmentation
    mask, markers, labels, wat = watseg(
        ridges, thresh_coeff, thresh_min_size, parallel=True)
    
    end = time.time()
    print(f'  {(end - start):5.3f} s')
    
    # .........................................................................
    
    if piv:
    
        start = time.time()
        print('PIV')
    
        # Particle image velocimetry (PIV)
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
        
        u = []
        v = []
        vector_field = []
    
    # .........................................................................  

    # start = time.time()
    # print('Process bounds')

    # rsize, labels, wat, u, v, bound_labels, bound_norm, bound_edm  = process_bounds(
    #     rsize, 
    #     labels, 
    #     wat, 
    #     u, 
    #     v, 
    #     time_window, 
    #     ridge_size,
    #     parallel=True
    #     )
    
    
    # end = time.time()
    # print(f'  {(end - start):5.3f} s')  

    # .........................................................................   

    # Extract outputs in a dictionnary
    outputs = {
        'rsize' : rsize,
        'ridges' : ridges,
        'mask' : mask,
        'markers' : markers,
        'labels' : labels,
        'wat' : wat,
        'u' : u, 
        'v' : v, 
        'vector_field' : vector_field,
        # 'bound_labels' : bound_labels,
        # 'bound_norm' : bound_norm,
        # 'bound_edm' : bound_edm
        }
    
    # ......................................................................... 

    return outputs     
