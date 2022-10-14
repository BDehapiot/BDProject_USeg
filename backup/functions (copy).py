#%% Imports

import numpy as np

from joblib import Parallel, delayed 

from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.filters import sato, gaussian, threshold_triangle
from skimage.segmentation import watershed, clear_border, expand_labels
from skimage.morphology import remove_small_objects, label, binary_dilation, remove_small_holes

#%% Preprocessing

def preprocess(raw, rsize_factor, ridge_size, parallel=True):
    
    ''' General description.
    
    Parameters
    ----------
    raw : np.ndarray (uint8, uint16)
        Description.
        
    rsize_factor : int
        Description.
    
    ridge_size : int
        Description.
        
    parallel : bool
        Description.
    
    Returns
    -------  
    rsize : np.ndarray
        Description.
        
    ridges : np.ndarray
        Description.    
        
    Notes
    -----   
    
    '''
    
    # Nested function ---------------------------------------------------------
    
    def _preprocess(raw, rsize_factor, ridge_size):
        
        # Resize raw
        rsize = resize(raw, (
            int(raw.shape[0]//rsize_factor), 
            int(raw.shape[1]//rsize_factor)),
            preserve_range=True, 
            anti_aliasing=True
            )            
                               
        # Background sub (! adjust)
        rsize -= rolling_ball(
            gaussian(rsize, sigma=ridge_size//2), radius=ridge_size*2) 

        # Ridge filter
        ridges = sato(
            rsize, 
            sigmas=ridge_size, 
            mode='reflect', 
            black_ridges=False
            )

        return rsize, ridges
    
    # Main function -----------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))         
    
    if parallel:
 
        # Run _preprocess (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_preprocess)(
                r_frame,
                rsize_factor,
                ridge_size
                )
            for r_frame in raw
            )
            
    else:
            
        # Run _preprocess
        output_list = [_preprocess(
                r_frame, 
                rsize_factor,
                ridge_size
                ) 
            for r_frame in raw
            ]
        
    # Extract outputs
    rsize = np.stack([arrays[0] for arrays in output_list], axis=0)
    ridges = np.stack([arrays[1] for arrays in output_list], axis=0)
    
    # Squeeze dimensions (if ndim == 2)    
    if ndim == 2:
        rsize = rsize.squeeze()
        ridges = ridges.squeeze()
    
    return rsize, ridges

#%% Watershed segmentation

def watseg(ridges, thresh_coeff, thresh_min_size, parallel=True):
    
    ''' General description.
    
    Parameters
    ----------
    ridges : np.ndarray
        Description.
        
    thresh_coeff : float
        Description.
    
    thresh_min_size : int
        Description.
        
    parallel : bool
        Description.

    Returns
    -------  
    mask : np.ndarray
        Description.
        
    markers : np.ndarray
        Description.
    
    labels : np.ndarray
        Description.
    
    wat : np.ndarray
        Description.        
        
    Notes
    -----   
    
    '''
    
    # Nested function ---------------------------------------------------------
    
    def _watseg(ridges, mask, thresh_min_size):
    
        # Get seed mask
        mask = remove_small_objects(mask, min_size=thresh_min_size) 
        
        # Get watershed 
        markers = label(np.invert(mask), connectivity=1)  
        labels = watershed(ridges, markers, compactness=1, watershed_line=True) 
        wat = labels == 0
            
        # Process labels
        labels = clear_border(labels)
        temp_mask = binary_dilation(labels > 0)
        temp_mask = remove_small_holes(temp_mask, area_threshold=2)
        labels = expand_labels(labels, distance=2)
        labels[temp_mask == 0] = 0
          
        # # Remove border cells    
        # temp_mask = binary_erosion(labels > 0)    
        # all_id = np.unique(labels)
        # for cell_id in all_id:
        #     idx = bd_where(labels, cell_id)    
        #     if False in temp_mask[idx]:
        #         labels[idx] = 0
                           
        # Remove isolated cells
        temp_mask = labels > 0
        temp_mask = remove_small_objects(temp_mask, min_size = thresh_min_size)
        labels[temp_mask==0] = 0  
        
        # Clean wat & vertices
        temp_mask = binary_dilation(labels > 0)
        wat[temp_mask == 0] = 0
         
        return mask, markers, labels, wat
    
    # Main function -----------------------------------------------------------
    
    # Add one dimension (if ndim = 2)
    ndim = (ridges.ndim)        
    if ndim == 2:
        ridges = ridges.reshape((1, ridges.shape[0], ridges.shape[1]))
    
    # Get threshold
    thresh = threshold_triangle(ridges)
    mask = ridges > thresh * thresh_coeff
    
    if parallel:
    
        # Run _watseg (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_watseg)(
                r_frame,
                m_frame,
                thresh_min_size
                )
            for r_frame, m_frame in zip(ridges, mask)
            )
        
    else:
        
        # Run _watseg
        output_list = [_watseg(
                r_frame, 
                m_frame,
                thresh_min_size
                ) 
            for r_frame, m_frame in zip(ridges, mask)
            ]

    # Extract outputs
    mask = np.stack([arrays[0] for arrays in output_list], axis=0)
    markers = np.stack([arrays[1] for arrays in output_list], axis=0)
    labels = np.stack([arrays[2] for arrays in output_list], axis=0)
    wat = np.stack([arrays[3] for arrays in output_list], axis=0)
    
    # Squeeze dimensions (if ndim == 2)     
    if ndim == 2:
        mask = mask.squeeze()
        markers = markers.squeeze()
        labels = labels.squeeze()
        wat = wat.squeeze()
    
    return mask, markers, labels, wat

#%% 

def getwat(raw, rsize_factor, ridge_size, thresh_coeff, parallel=True):
    
    ''' General description.
    
    Parameters
    ----------
    raw : np.ndarray (uint8, uint16)
        Description.
        
    rsize_factor : int
        Description.
    
    ridge_size : int
        Description.
        
    thresh_coeff : float
        Description.
        
    parallel : bool
        Description.
    
    Returns
    -------  
    rsize : np.ndarray
        Description.
        
    ridges : np.ndarray
        Description.    
        
    Notes
    -----   
    
    '''
    
    # Nested functions --------------------------------------------------------
    
    def _getwat(raw, rsize_factor, ridge_size, thresh_coeff):
        
        thresh_min_size = 1000 # should be auto
        
        # Resize raw
        rsize = resize(raw, (
            int(raw.shape[0]//rsize_factor), 
            int(raw.shape[1]//rsize_factor)),
            preserve_range=True, 
            anti_aliasing=True
            )            
                               
        # Background sub (! adjust)
        rsize -= rolling_ball(
            gaussian(rsize, sigma=ridge_size//2), radius=ridge_size*2) 

        # Ridge filter
        ridges = sato(
            rsize, 
            sigmas=ridge_size, 
            mode='reflect', 
            black_ridges=False
            )
        
        # Get seed mask
        thresh = threshold_triangle(ridges)
        mask = ridges > thresh * thresh_coeff
        mask = remove_small_objects(mask, min_size=thresh_min_size)

        # Get watershed 
        markers = label(np.invert(mask), connectivity=1)  
        labels = watershed(ridges, markers, compactness=1, watershed_line=True) 
        wat = labels == 0  
        
        # Process labels
        labels = clear_border(labels)
        temp_mask = binary_dilation(labels > 0)
        temp_mask = remove_small_holes(temp_mask, area_threshold=2)
        labels = expand_labels(labels, distance=2)
        labels[temp_mask == 0] = 0
                                     
        # Remove isolated cells
        temp_mask = labels > 0
        temp_mask = remove_small_objects(temp_mask, min_size = thresh_min_size)
        labels[temp_mask==0] = 0  
        
        # Clean wat & vertices
        temp_mask = binary_dilation(labels > 0)
        wat[temp_mask == 0] = 0
        
        #
        watsize = resize(wat, (
            raw.shape[0],
            raw.shape[1]),
            preserve_range=True
            )

        return rsize, ridges, mask, markers, labels, wat, watsize
        
    # Main functions --------------------------------------------------------
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1])) 
    
    if parallel:
 
        # Run _preprocess (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_getwat)(
                frame,
                rsize_factor,
                ridge_size,
                thresh_coeff,
                )
            for frame in raw
            )
            
    else:
            
        # Run _preprocess
        output_list = [_getwat(
                frame, 
                rsize_factor,
                ridge_size,
                thresh_coeff,
                ) 
            for frame in raw
            ]
        
    # Extract outputs
    rsize = np.stack([arrays[0] for arrays in output_list], axis=0)
    ridges = np.stack([arrays[1] for arrays in output_list], axis=0)
    mask = np.stack([arrays[2] for arrays in output_list], axis=0)
    markers = np.stack([arrays[3] for arrays in output_list], axis=0)
    labels = np.stack([arrays[4] for arrays in output_list], axis=0)
    wat = np.stack([arrays[5] for arrays in output_list], axis=0)
    watsize = np.stack([arrays[6] for arrays in output_list], axis=0)
    
    # Squeeze dimensions (if ndim == 2)    
    if ndim == 2:
        rsize = rsize.squeeze()
        ridges = ridges.squeeze()
        mask = mask.squeeze()
        markers = markers.squeeze()
        labels = labels.squeeze()
        wat = wat.squeeze()
        watsize = watsize.squeeze()
    
    return rsize, ridges, mask, markers, labels, wat, watsize

