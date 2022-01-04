#%%

import warnings
import numpy as np
import io as std_io
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from openpiv import pyprocess

from skimage.transform import resize

#%%

from useg.tools.nan import bd_nanfilt, bd_nanoutliers, bd_nanreplace

#%% bd_openpiv

def _bd_openpiv_process(img, img_ref, time_window, win_size):
    
    """General description.

    Parameters
    ----------
    img : np.ndarray
        Description.
        
    img_ref : np.ndarray
        Description.

    time_window : int
        Description.

    win_size : int
        Description.

    Returns
    -------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    Notes
    -----

    """

    # Compute PIV
    u, v, s2n = pyprocess.extended_search_area_piv(
                    img_ref, img, dt = 1,
                    window_size = win_size,
                    overlap = win_size//2,
                    search_area_size = win_size,
                    sig2noise_method = 'peak2peak'
                    )
    
    # Rescale u, v according to time_window
    u = u/(time_window-1)
    v = v/(time_window-1)
    
    return u, v

''' ........................................................................'''

def _bd_openpiv_mframe(u, v, time_window, missing_frames):
    
    """General description.

    Parameters
    ----------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    time_window : int
        Description.

    missing_frames : str
        Description.

    Returns
    -------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    Notes
    -----

    """
    
    # Create missing frames    
    if missing_frames == 'empty':
        prev_frames_u = np.full([time_window//2,u.shape[1],u.shape[2]], np.nan)
        post_frames_u = np.full([time_window//2,u.shape[1],u.shape[2]], np.nan)
        prev_frames_v = np.full([time_window//2,v.shape[1],v.shape[2]], np.nan)
        post_frames_v = np.full([time_window//2,v.shape[1],v.shape[2]], np.nan)
        
    if missing_frames == 'nearest':  
        prev_frames_u = np.repeat(
            np.expand_dims(u[0,...], axis=0), time_window//2, axis=0)
        post_frames_u = np.repeat(
            np.expand_dims(u[-1,...], axis=0), time_window//2, axis=0)
        prev_frames_v = np.repeat(
            np.expand_dims(v[0,...], axis=0), time_window//2, axis=0)
        post_frames_v = np.repeat(
            np.expand_dims(v[-1,...], axis=0), time_window//2, axis=0)
        
    # Concatenate missing frames
    u = np.concatenate((prev_frames_u, u, post_frames_u), axis=0)
    v = np.concatenate((prev_frames_v, v, post_frames_v), axis=0)  
    
    return u, v

''' ........................................................................'''

def _bd_openpiv_filt(u, v, mask, smooth_size, smooth_method):
    
    """General description.

    Parameters
    ----------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.
        
    mask : np.ndarray (bool)
        Description.        

    smooth_size : int
        Description.

    smooth_method : str
        Description.

    Returns
    -------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    Notes
    -----

    """
    
    # Mask out vector field
    if mask is not None:
        
        u[mask==False] = np.nan
        v[mask==False] = np.nan
    
    # Filt vector field   
    u = bd_nanoutliers(u, smooth_size, smooth_method, sd_thresh=1.5)
    u = bd_nanreplace(u, smooth_size, smooth_method, mask)
    u = bd_nanfilt(u, smooth_size, smooth_method)

    v = bd_nanoutliers(v, smooth_size, smooth_method, sd_thresh=1.5)
    v = bd_nanreplace(v, smooth_size, smooth_method, mask)
    v = bd_nanfilt(v, smooth_size, smooth_method)

    return u, v

''' ........................................................................'''

def _bd_openpiv_display(img, u, v, x_grid, y_grid, rescale, **kwargs):
    
    """General description.

    Parameters
    ----------
    img : np.ndarray 
        Description.
    
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.
        
    x_grid : np.ndarray (int)
        Description.        

    y_grid : np.ndarray (int)
        Description.

    rescale : int
        Description.
        
    **kwargs are passed to the matplotib `quiver` method 

    Returns
    -------
    display : np.ndarray
        Description.

    Notes
    -----

    """
    
    # Warnings
    warnings.filterwarnings(
        action="ignore", message="invalid value encountered in double_scalars")
    warnings.filterwarnings(
        action="ignore", message="Mean of empty slice")
    
    # Plot figure 
    ny, nx = img.shape
    dpi = max(nx, ny)
    fig = plt.Figure(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap="gray_r", extent=(0, nx, ny, 0))
    ax.quiver(x_grid, y_grid, u, v, pivot='mid', **kwargs)
    ax.set_axis_off()
    
    # Extract figure 
    with std_io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', dpi=rescale*dpi)
        io_buf.seek(0)

        vector_field = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
            )[..., 0] # keep only one channel from RGB
        
    plt.close(fig)
    
    # vector_field = img
    
    return vector_field

''' ........................................................................'''

def _bd_openpiv_bsize(stack, u, v, win_size):
    
    """General description.

    Parameters
    ----------
    stack : np.ndarray
        Description.    
    
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.     

    win_size : int
        Description.

    Returns
    -------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    Notes
    -----

    """
            
    # Resize u & v 
    u = resize(u,
        (u.shape[0], u.shape[1]*win_size//2, u.shape[2]*win_size//2),
        preserve_range=True, order=0)
    
    v = resize(v,
        (v.shape[0], v.shape[1]*win_size//2, v.shape[2]*win_size//2),
        preserve_range=True, order=0)
    
    # Pad u & v borders with NaNs
    pad_y = stack.shape[1] - u.shape[1]
    pad_x = stack.shape[2] - u.shape[2]        
    
    u = np.pad(u, pad_width=((0, 0),
            (int(np.ceil(pad_y/2)), int(np.floor(pad_y/2))),
            (int(np.ceil(pad_x/2)), int(np.floor(pad_x/2)))),
            mode="constant", constant_values=np.nan,
            )
    
    v = np.pad(v, pad_width=((0, 0),
            (int(np.ceil(pad_y/2)), int(np.floor(pad_y/2))),
            (int(np.ceil(pad_x/2)), int(np.floor(pad_x/2)))),
            mode="constant", constant_values=np.nan,
            )
    
    return u, v

''' ........................................................................'''

def bd_openpiv(
        stack, 
        time_window, 
        win_size, 
        mask=None,
        smooth_size=3,
        smooth_method='median',
        missing_frames='empty', 
        bsize=True,
        display=True,
        parallel=True
        ):
    
    """General description.

    Parameters
    ----------
    stack : np.ndarray
        Description.    
    
    time_window : int
        Description.
        
    win_size : int
        Description.     

    mask : np.ndarray (bool)
        Description.
        
    smooth_size : int
        Description.

    smooth_method : str
        Description.
        
    missing_frames : str
        Description.
        
    bsize : bool
        Description.
        
    display : bool
        Description.
        
    parallel : bool 
        Description.

    Returns
    -------
    u : np.ndarray (float)
        Description.
        
    v : np.ndarray (float)
        Description.

    Notes
    -----

    """
    
# Initialize ..................................................................
   
    # Define PIV time parameters
    t_int = np.arange(
        time_window-1, stack.shape[0], dtype=int)
    t_ref = np.arange(
        0, stack.shape[0]-(time_window-1), dtype=int)
    
    # Get PIV window coordinates
    x_grid, y_grid = pyprocess.get_coordinates(
        image_size = stack[0].shape,
        search_area_size = win_size,
        overlap = win_size//2
        )
    
    x_grid = x_grid.astype('int')
    y_grid = y_grid.astype('int')
    
    if mask is not None:   
        
        # Resize mask to match xy grid
        y1 = np.min(y_grid) - win_size//2
        y2 = np.max(y_grid) + win_size//2
        x1 = np.min(x_grid) - win_size//2
        x2 = np.max(x_grid) + win_size//2
       
        mask = resize(mask[:,y1:y2,x1:x2], 
            (mask.shape[0], x_grid.shape[0], x_grid.shape[1]), 
            preserve_range=True) 
        
# Run _bd_openpiv_process .....................................................
    
    if parallel:
        
        # Parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_bd_openpiv_process)(
                stack[t_int[i],:,:],
                stack[t_ref[i],:,:],
                time_window,
                win_size
                )
            for i in range(len(t_int))
            )
        
    else:
        
        # Serial
        output_list = [_bd_openpiv_process(
                stack[t_int[i],:,:],
                stack[t_ref[i],:,:],
                time_window,
                win_size
                ) 
            for i in range(len(t_int))
            ]
    
    # Extract outputs
    u = np.stack([arrays[0] for arrays in output_list], axis=0)
    v = np.stack([arrays[1] for arrays in output_list], axis=0)
    
# Run _bd_openpiv_mframe ......................................................

    u, v = _bd_openpiv_mframe(u, v, time_window, missing_frames)
    
# Run _bd_openpiv_filt ........................................................
    
    u, v = _bd_openpiv_filt(u, v, mask, smooth_size, smooth_method)
    
# Run _bd_openpiv_display .....................................................
    
    if display:
        
        if parallel:
            
            # Parallel
            output_list = Parallel(n_jobs=-1)(
                delayed(_bd_openpiv_display)(
                    stack[i,...], u[i,...], -v[i,...],
                    x_grid, y_grid, 1
                    )
                for i in range(stack.shape[0])
                )
            
        else:
            
            # Serial
            output_list = [_bd_openpiv_display(
                    stack[i,...], u[i,...], -v[i,...],
                    x_grid, y_grid, 1
                    ) 
                for i in range(stack.shape[0])
                ]        

        # Extract outputs
        vector_field = np.stack([arrays for arrays in output_list], axis=0)
        
# Run _bd_openpiv_bsize .......................................................    
        
    if bsize:
        
        u, v = _bd_openpiv_bsize(stack, u, v, win_size)
        
# Terminate ...................................................................        
        
    if display:
        
        return u, v, vector_field
        
    else:
        
        return u, v

#%% Standalone exe

# import time
# from skimage import io

# # Path
# ROOT_PATH = '../../data/'
# RAW_NAME = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif"

# RSIZE_NAME = RAW_NAME[0:-4] + '_rsize.tif'
# LABELS_NAME = RAW_NAME[0:-4] + '_labels.tif'

# # Open data
# rsize = io.imread(ROOT_PATH + RSIZE_NAME)
# labels = io.imread(ROOT_PATH + LABELS_NAME)

# # # .............................................................................

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

# # # .............................................................................

# if PIV:

#     start = time.time()
#     print('PIV')

#     # Create piv_mask
#     piv_mask = labels > 0

#     u, v, vector_field = bd_openpiv(
#         rsize,
#         TIME_WINDOW, 
#         PIV_WIN_SIZE, 
#         mask=piv_mask,
#         smooth_size=3,
#         smooth_method='median',
#         missing_frames='nearest', 
#         bsize=True,
#         display=True,
#         parallel=True
#         )
        
#     end = time.time()
#     print(f'  {(end - start):5.3f} s')

# io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_u.tif', u.astype("float32"), check_contrast=False) 
# io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_v.tif', v.astype("float32"), check_contrast=False)
# io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_display.tif', display.astype("uint8"), check_contrast=False)
