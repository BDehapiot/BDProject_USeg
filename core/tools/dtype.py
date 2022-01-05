#%% 

import numpy as np

#%% bd_uint8

def bd_uint8(img, int_range=0.99):

    ''' General description.
    
    Parameters
    ----------
    img : np.ndarray
        Description.
        
    int_range : float
        Description.
    
    Returns
    -------  
    img : np.ndarray
        Description.
        
    Notes
    -----   
    
    '''

    # Get data type 
    data_type = (img.dtype).name
    
    if data_type == 'uint8':
        
        raise ValueError('Input image is already uint8') 
        
    else:
        
        # Get data intensity range
        int_min = np.percentile(img, (1-int_range)*100)
        int_max = np.percentile(img, int_range*100) 
        
        # Rescale data
        img[img<int_min] = int_min
        img[img>int_max] = int_max 
        img = (img - int_min)/(int_max - int_min)
        img = (img*255).astype('uint8')
    
    return img
        