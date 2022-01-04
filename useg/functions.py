#%% 

import time
import numpy as np

from joblib import Parallel, delayed 

from scipy import ndimage

from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.filters import sato, gaussian, threshold_triangle
from skimage.segmentation import watershed, clear_border, expand_labels
from skimage.morphology import remove_small_objects, label, binary_dilation, remove_small_holes, square, disk 

#%%

from useg.tools.conn import bd_labconn
from useg.tools.nan import bd_nanreplace

#%% Best ridge size

def best_ridge_size(raw, rsize_factor, sigma_inc=0.5):
    
    ''' General description.
    
    Parameters
    ----------
    raw : np.ndarray (uint8, uint16)
        Description.
        
    rsize_factor : int
        Description.
    
    sigma_inc : float
        Description.
    
    Returns
    -------  
    ridge_size : float
        Description.
        
    Notes
    -----   
    
    '''
    
    start = time.time()
    print('Best ridge size')
    
    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))         
    
    # Get test image
    test_raw = resize(raw[raw.shape[0]//2,...], (
        int(raw.shape[1]*rsize_factor), 
        int(raw.shape[2]*rsize_factor)), 
        preserve_range=True, 
        anti_aliasing=True)  

    # Initialize while loop 
    test_sigma = sigma_inc    
    test_ridge = sato(test_raw, 
        sigmas=test_sigma, 
        mode='reflect', 
        black_ridges=False)       
    test_mean = np.mean(test_ridge)
    test_mean_prev = test_mean
    
    # Determine best_ridge_size
    while test_mean >= test_mean_prev:
        
        test_mean_prev = test_mean        
        test_ridge = sato(test_raw, 
            sigmas=test_sigma, 
            mode='reflect', 
            black_ridges=False)           
        test_mean = np.mean(test_ridge)
        test_sigma = test_sigma + sigma_inc
    
    ridge_size = test_sigma     
    
    # Display result
    print(f'  ridge_size(auto) = {ridge_size}')

    end = time.time()
    print(f'  {(end - start):5.3f} s')             
        
    return ridge_size
       

#%% Preprocessing

def _preprocess(raw, rsize_factor, ridge_size):
    
    ''' General description.
    
    Parameters
    ----------
    raw : np.ndarray (uint8, uint16)
        Description.
        
    rsize_factor : int
        Description.
    
    ridge_size : int
        Description.
    
    Returns
    -------  
    rsize : np.ndarray (float)
        Description.
        
    ridges : np.ndarray (float)
        Description.    
        
    Notes
    -----   
    
    '''
    
    # Resize raw
    rsize = resize(raw, (
        int(raw.shape[0]*rsize_factor), 
        int(raw.shape[1]*rsize_factor)),
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

''' ........................................................................'''

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
    
    Returns
    -------  
    rsize : np.ndarray
        Description.
        
    ridges : np.ndarray
        Description.    
        
    Notes
    -----   
    
    '''
    
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

def _watseg(ridges, mask, thresh_min_size):
    
    ''' General description.
    
    Parameters
    ----------
    ridges : np.ndarray
        Description.
        
    mask : np.ndarray (bool)
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

''' ........................................................................'''

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

#%% process_bounds 

def _process_bounds_label(labels, wat):
    
    ''' General description.
    
    Parameters
    ----------    
    labels : np.ndarray (uint16)
        Description.
        
    wat : np.ndarray (bool)
        Description.

    Returns
    -------  
    bound_labels : np.ndarray (uint16)
        Description.    
        
    Notes
    -----   
    
    '''

    # Get vertices
    vertices = bd_labconn(wat, labels=labels, conn=2) > 2
    
    # Get bounds & endpoints
    bounds = wat.copy()
    bounds[binary_dilation(vertices, square(3)) == 1] = 0
    endpoints = wat ^ bounds ^ vertices
    
    # Label bounds
    bounds_labels = label(bounds, connectivity=2).astype('float')
    bounds_labels[endpoints == 1] = np.nan
    bounds_labels = bd_nanreplace(bounds_labels, 3, 'max')
    bounds_labels = bounds_labels.astype('int')
    
    # Label small bounds 
    small_bounds = wat ^ (bounds_labels > 0) ^ vertices
    small_bounds_labels = label(small_bounds, connectivity=2)
    small_bounds_labels = small_bounds_labels + np.max(bounds_labels)
    small_bounds_labels[small_bounds_labels == np.max(bounds_labels)] = 0
    
    # Merge bounds and small bounds
    bound_labels = bounds_labels + small_bounds_labels
    
    return bound_labels

''' ........................................................................'''

def _process_bounds_extend_time(stack, time_window, coeff=1):
    
    ''' General description.
    
    Parameters
    ----------
    stack : np.ndarray
        Description.
        
    time_window : int
        Description.
        
    coeff : float
        Descriptions.    

    Returns
    -------  
    stack : np.ndarray
        Description.    
        
    Notes
    -----   
    
    '''
    
    dtype = stack.dtype
    
    stack_prev = np.flip(
        stack[1:(time_window//2)+1,...], axis=0)*coeff
    stack_post = np.flip(
        stack[-1-(time_window//2):-1,...], axis=0)*coeff
    stack = np.concatenate(
        (stack_prev, stack, stack_post)).astype(dtype)
    
    return stack

''' ........................................................................'''

def _process_bounds_preprocess(rsize, labels, wat, ridge_size):
    
    ''' General description.
    
    Parameters
    ----------
    rsize : np.ndarray
        Description.
        
    labels : np.ndarray (uint16)
        Description.
        
    wat : np.ndarray (bool)
        Description.  
        
    ridge_size : float
        Description.

    Returns
    -------  
    bound_norm : np.ndarray (float)
        Description.    
        
    bound_edm : np.ndarray (float)
        Description. 
        
    Notes
    -----   
    
    '''
           
    # Get temp_mask
    temp_mask = labels > 0
    temp_mask = binary_dilation(
        temp_mask, selem = disk(ridge_size)
        )
    
    # Get bg_blur (with a nan ignoring Gaussian blur)
    temp1 = rsize.astype('float')
    temp1[temp_mask == 0] = np.nan

    temp2 = temp1.copy()
    temp2[np.isnan(temp2)] = 0
    temp2_blur = gaussian(temp2, sigma=ridge_size*10)
    
    temp3 = 0 * temp1.copy() + 1
    temp3[np.isnan(temp3)] = 0
    temp3_blur = gaussian(temp3, sigma=ridge_size*10)
    
    np.seterr(divide='ignore', invalid='ignore')
    bg_blur = temp2_blur / temp3_blur
    
    # Get rsize_blur (divided by bg_blur)
    bound_norm = gaussian(rsize, sigma=ridge_size//2)
    bound_norm = bound_norm / bg_blur *1.5
    
    # Get wat_edm_int 
    bound_edm = ndimage.distance_transform_edt(np.invert(wat))
    bound_edm[temp_mask == 0] = np.nan
    
    return bound_norm, bound_edm

''' ........................................................................'''

def _process_bounds_filt(
        wat,
        u, v,
        bound_labels, 
        bound_norm, 
        bound_edm, 
        time_window,
        time_range,
        time_idx
        ):
    
    ''' General description.
    
    Parameters
    ----------        
    wat : np.ndarray (bool)
        Description.
        
    u : np.ndarray (float) 
        Description.

    v : np.ndarray (float) 
        Description.
        
    bound_labels : np.ndarray (int)
        Description.  
        
    bound_norm : np.ndarray (float)
        Description.
        
    bound_edm : np.ndarray (float)
        Description.     
        
    time_window : int
        Description.
        
    time_range : np.ndarray (int)  
    
    time_idx : int
        Description.

    Returns
    -------  
    bound_int : np.ndarray (float)
        Description.    
        
    bound_edm : np.ndarray (float)
        Description. 
        
    Notes
    -----   
    
    '''
    
    # Initialize
    bound_data = []
    time_point = time_idx - time_window
        
    # Get id, area and linear indexes
    temp_bound_labels = bound_labels.ravel()
    idx_sort = np.argsort(temp_bound_labels)
    temp_bound_labels_sorted = temp_bound_labels[idx_sort]
    all_id, idx_start, all_area = np.unique(
        temp_bound_labels_sorted,
        return_index=True,
        return_counts=True
        )
    lin_idx = np.split(idx_sort, idx_start[1:]) 
        
    for j in range(len(all_id)):
        
        bound_id = all_id[j].astype('int')
        
        if bound_id > 0:
        
            # Get bound info                        
            bound_idx = np.unravel_index(lin_idx[j], bound_labels.shape)             
            bound_area = all_area[j].astype('int')
            bound_int = np.mean(bound_norm[bound_idx])
            bound_ctrd = (bound_idx[0].squeeze().mean(),
                          bound_idx[1].squeeze().mean())

            if time_window > 1:
            
                if u is not None:

                    # Extract local PIV info 
                    d_u = u[round(bound_ctrd[0]), round(bound_ctrd[1])]
                    d_v = v[round(bound_ctrd[0]), round(bound_ctrd[1])]
                                                                                   
                    if np.isnan(d_u):
                        
                        d_u = 0; d_v = 0  
                                            
                # Get 2D indexes
                idx_y = bound_idx[0].squeeze().astype('int')
                idx_x = bound_idx[1].squeeze().astype('int') 
                
                # Correct PIV according to t_point
                piv_correct = time_range[
                    time_idx-time_window:time_idx+time_window+1] - time_point
                d_u_cor = round(d_u) * piv_correct 
                d_v_cor = round(d_v) * piv_correct 
                   
                # Get bound_edm_range
                bound_edm_range = np.zeros(bound_edm.shape[0])
                for k in range(bound_edm.shape[0]):

                    # Get PIV corrected idx
                    idx_y_cor = idx_y + d_v_cor[k]
                    idx_x_cor = idx_x + d_u_cor[k]
                
                    # Remove out of frame idx (very slow!!!)
                    idx_valid = np.invert(
                        (idx_y_cor < 0) + (idx_y_cor > bound_edm.shape[1]-1)+
                        (idx_x_cor < 0) + (idx_x_cor > bound_edm.shape[2]-1))
                    idx_y_cor = idx_y_cor[idx_valid == True]
                    idx_x_cor = idx_x_cor[idx_valid == True]
                
                    # Measure 
                    bound_edm_range[k] = np.mean(
                        bound_edm[k, idx_y_cor, idx_x_cor])
                
                # Get wat_edm_int (avg of wat_edm_range excluding t0)               
                bound_edm_t0 = bound_edm_range[time_window]
                bound_edm_range[time_window] = np.nan
                bound_edm_int = np.nanmean(bound_edm_range)
                bound_edm_sd = np.nanstd(bound_edm_range)
                                
            else:
                
                bound_edm_int = 1
                bound_edm_sd = 0
                d_u = 0; d_v = 0  
                     
            # Append bound_data 
            temp_data = {
                'time_point' : time_point,
                'id' : bound_id,
                'area' : bound_area,
                'ctrd' : bound_ctrd,
                'idx' : bound_idx,
                'bound_int' : bound_int,
                'bound_edm_int' : bound_edm_int, 
                'bound_edm_sd' : bound_edm_sd, 
                'd_u' : d_u,
                'd_v' : d_v
                }
            
            bound_data.append(temp_data)
                
    return bound_data

''' ........................................................................'''

def process_bounds(
        rsize, 
        labels, 
        wat, 
        u, v,    
        time_window,
        ridge_size,
        parallel=True
        ):
    
    ''' General description.
    
    Parameters
    ----------
    wat : np.ndarray (bool)
        Description.
        
    labels : np.ndarray (uint16)
        Description.
        
    time_window : int
        Description.

    Returns
    -------  
    bound_labels : np.ndarray (uint16)
        Description.    
        
    Notes
    -----   
    
    '''
        
# Initialize ..................................................................
    
    # Add one dimension (if ndim = 2)
    ndim = (rsize.ndim)        
    if ndim == 2:
        rsize = rsize.reshape((1, rsize.shape[0], rsize.shape[1]))
        labels = labels.reshape((1, labels.shape[0], labels.shape[1]))
        wat = wat.reshape((1, wat.shape[0], wat.shape[1]))
        
    # Get time_info    
    time_0 = time_window // 2
    time_end = rsize.shape[0] + (time_0-1) 
    time_range = np.arange(0, rsize.shape[0]) 
    
# _process_bounds_extend_time .................................................    
        
    if time_window > 1:
        
        # Extend time (see description) 
        rsize = _process_bounds_extend_time(rsize, time_window)
        labels = _process_bounds_extend_time(labels, time_window)
        wat = _process_bounds_extend_time(wat, time_window)
        time_range = _process_bounds_extend_time(time_range, time_window)
        
        if u is not None:
            
            u = _process_bounds_extend_time(u, time_window, coeff=-1)
            v = _process_bounds_extend_time(v, time_window, coeff=-1)
        
# _process_bounds_label .......................................................        
    
    if parallel:
    
        output_list = Parallel(n_jobs=-1)(
            delayed(_process_bounds_label)(
                l_frame,
                w_frame
                )
            for l_frame, w_frame in zip(labels, wat)
            )
        
    else:
        
        output_list = [_process_bounds_label(
                l_frame,
                w_frame
                )
            for l_frame, w_frame in zip(labels, wat)
            ]

    # Extract output
    bound_labels = np.stack([arrays for arrays in output_list], axis=0)
        
    # Re-label to get unique object_id 
    mask = bound_labels > 0
    bound_labels_max = np.max(bound_labels, axis=(1,2))
    bound_labels_max_cum = np.cumsum(
        bound_labels_max, axis=0) - bound_labels_max[0]
    bound_labels = bound_labels + bound_labels_max_cum[:, None, None]
    bound_labels[mask == 0] = 0 
    
# _process_bounds_preprocess ..................................................

    if parallel:
        
        output_list = Parallel(n_jobs=-1)(
            delayed(_process_bounds_preprocess)(
                r_frame,
                l_frame,
                w_frame,
                ridge_size
                )
            for r_frame, l_frame, w_frame in zip(rsize, labels, wat)
            ) 
        
    else:
        
        output_list = [_process_bounds_preprocess(
                r_frame,
                l_frame,
                w_frame,
                ridge_size
                )
            for r_frame, l_frame, w_frame in zip(rsize, labels, wat)
            ]
        
    # Extract output
    bound_norm = np.stack([arrays[0] for arrays in output_list], axis=0)
    bound_edm = np.stack([arrays[1] for arrays in output_list], axis=0)
    
# _process_bounds_extract .....................................................

    # Get id, area and linear indexes
    temp_bound_labels = bound_labels.ravel()
    idx_sort = np.argsort(temp_bound_labels)
    temp_bound_labels_sorted = temp_bound_labels[idx_sort]
    all_id, idx_start, all_area = np.unique(
        temp_bound_labels_sorted,
        return_index=True,
        return_counts=True
        )
    lin_idx = np.split(idx_sort, idx_start[1:]) 

# Terminate ...................................................................
    
    # Squeeze dimensions (if ndim == 2)   
    if ndim == 2:
        rsize = rsize.squeeze()
        labels = labels.squeeze()
        wat = wat.squeeze()
        bound_labels = bound_labels.squeeze()

    return rsize, labels, wat, u, v, bound_labels, bound_norm, bound_edm

#%%

import time
from skimage import io

# Path
ROOT_PATH = 'data/'
RAW_NAME = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif"

RSIZE_NAME = RAW_NAME[0:-4] + '_rsize.tif'
LABELS_NAME = RAW_NAME[0:-4] + '_labels.tif'
WAT_NAME = RAW_NAME[0:-4] + '_wat.tif'
U_NAME = RAW_NAME[0:-4] + '_u.tif'
V_NAME = RAW_NAME[0:-4] + '_v.tif'
BOUND_LABELS_NAME = RAW_NAME[0:-4] + '_bound_labels.tif'
BOUND_NORM_NAME = RAW_NAME[0:-4] + '_bound_norm.tif'
BOUND_EDM_NAME = RAW_NAME[0:-4] + '_bound_edm.tif'

# Open data
rsize = io.imread(ROOT_PATH + RSIZE_NAME)
labels = io.imread(ROOT_PATH + LABELS_NAME)
wat = io.imread(ROOT_PATH + WAT_NAME)
u = io.imread(ROOT_PATH + U_NAME)
v = io.imread(ROOT_PATH + V_NAME)
bound_labels = io.imread(ROOT_PATH + BOUND_LABELS_NAME)
bound_norm = io.imread(ROOT_PATH + BOUND_NORM_NAME)
bound_edm = io.imread(ROOT_PATH + BOUND_EDM_NAME)

# .............................................................................

# ''' 2) General options '''
# RSIZE_FACTOR = 0.5 # must be >= 1
# TIME_WINDOW = 3 # must be odd (must be >= 3 if PIV)

# ''' 3) Preprocess '''
# RIDGE_SIZE = 'auto' 
# RIDGE_SIZE_COEFF = 0.75

# ''' 4) Watershed '''
# THRESH_COEFF = 0.5 
# THRESH_MIN_SIZE = int(3000*RSIZE_FACTOR)  

# ''' 5) PIV '''
# PIV = True 
# PIV_WIN_SIZE = int(96*RSIZE_FACTOR)

# .............................................................................

# # Get id, area and linear indexes
# bound_lin_labels = bound_labels.ravel()
# idx_sort = np.argsort(bound_lin_labels)
# bound_lin_labels_sorted = bound_lin_labels[idx_sort]
# bound_id, idx_start, bound_area = np.unique(
#     bound_lin_labels_sorted,
#     return_index=True,
#     return_counts=True
#     )

# Bound_lin_idx = np.split(idx_sort, idx_start[1:]) 
# bound_idx = ([np.unravel_index(lin_idx, bound_labels.shape)
#                       for lin_idx in Bound_lin_idx])

# .............................................................................

# bound_int = bound_norm.ravel().take(idx_sort)
# lin_int = np.split(bound_int, idx_start[1:]) 

# # test = np.array(list(map(np.mean, lin_int)))

# test = np.array([ints.mean() for ints in lin_int])

#             bound_idx = np.unravel_index(lin_idx[j], bound_labels.shape)             
#             bound_area = all_area[j].astype('int')
#             bound_int = np.mean(bound_norm[bound_idx])
#             bound_ctrd = (bound_idx[0].squeeze().mean(),
#                           bound_idx[1].squeeze().mean())

