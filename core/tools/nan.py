#%%

import warnings
import numpy as np

#%%

from core.tools.idx import bd_where

#%% bd_nanfilt

def bd_nanfilt(img, kernel_size, method):

    """General description.

    Parameters
    ----------
    img : np.ndarray
        Description.

    kernel_size : int
        Description.

    method : str
        Description.

    Returns
    -------
    img_filt : np.ndarray
        Description.

    Notes
    -----

    """

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and > 1")

    # Add one dimension (if ndim = 2)
    ndim = img.ndim
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    # Pad img border with NaNs
    pad = (kernel_size - 1) // 2
    img_pad = np.pad(
        img,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=np.nan,
    )

    # Find non-NaNs coordinates
    idx = bd_where(~np.isnan(img), True)
    idx_t = idx[0].squeeze().astype("int")
    idx_y = idx[1].squeeze().astype("int")
    idx_x = idx[2].squeeze().astype("int")

    if idx:

        # Define kernels
        kernel_t = np.zeros([kernel_size, kernel_size], dtype=int)
        kernel_x, kernel_y = np.meshgrid(
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
        )
        idx_tt = idx_t[:, None, None] + kernel_t
        idx_xx = idx_x[:, None, None] + kernel_x
        idx_yy = idx_y[:, None, None] + kernel_y

        # Filter image
        img_nanfilt = img.copy()
        functions = {
            'mean': np.nanmean, 
            'median': np.nanmedian, 
            'std': np.nanstd, 
            'max': np.nanmax, 
            'min': np.nanmin
            } 
        all_kernels = img_pad[idx_tt, idx_yy + pad, idx_xx + pad]
        all_kernels = functions[method](all_kernels, axis=(1, 2))
        for i in range(len(all_kernels)):
            img_nanfilt[idx_t[i], idx_y[i], idx_x[i]] = all_kernels[i]

    # Remove one dimension (if ndim = 2)
    if ndim == 2:
        img = img.squeeze()
        img_nanfilt = img_nanfilt.squeeze()

    return img_nanfilt


#%% bd_nanreplace

def bd_nanreplace(img, kernel_size, method, mask=None):

    """General description.

    Parameters
    ----------
    img : np.ndarray
        Description.

    kernel_size : int
        Description.

    method : str
        Description.

    mask : np.ndarray
        Description.

    Returns
    -------
    img_nanreplace : np.ndarray
        Description.

    Notes
    -----

    """

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and > 1")

    # Add one dimension (if ndim = 2)
    ndim = img.ndim
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    # Pad img border with NaNs
    pad = (kernel_size - 1) // 2
    img_pad = np.pad(
        img,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=np.nan,
    )

    # Find NaNs coordinates
    if mask is None:
        idx = bd_where(np.isnan(img), True)
    else:
        idx = bd_where(np.isnan(img) & (mask > 0), True)
    idx_t = idx[0].squeeze().astype("int")
    idx_y = idx[1].squeeze().astype("int")
    idx_x = idx[2].squeeze().astype("int")

    if idx:

        # Define kernels
        kernel_t = np.zeros([kernel_size, kernel_size], dtype=int)
        kernel_x, kernel_y = np.meshgrid(
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
        )
        idx_tt = idx_t[:, None, None] + kernel_t
        idx_xx = idx_x[:, None, None] + kernel_x
        idx_yy = idx_y[:, None, None] + kernel_y

        # Replace NaNs
        img_nanreplace = img.copy()
        functions = {
            'mean': np.nanmean, 
            'median': np.nanmedian, 
            'std': np.nanstd, 
            'max': np.nanmax, 
            'min': np.nanmin
            } 
        all_kernels = img_pad[idx_tt, idx_yy + pad, idx_xx + pad]
        all_kernels = functions[method](all_kernels, axis=(1, 2))
        for i in range(len(all_kernels)):
            img_nanreplace[idx_t[i], idx_y[i], idx_x[i]] = all_kernels[i]

    # Remove one dimension (if ndim = 2)
    if ndim == 2:
        img = img.squeeze()
        img_nanreplace = img_nanreplace.squeeze()

    return img_nanreplace


#%% bd_nanoutliers

def bd_nanoutliers(img, kernel_size, method, sd_thresh=1.5):

    """General description.

    Parameters
    ----------
    img : np.ndarray
        Description.

    kernel_size : int
        Description.

    method : str
        Description.

    sd_thresh : float
        Description.

    Returns
    -------
    img_outfilt : np.ndarray
        Description.

    Notes
    -----

    """

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and > 1")

    # Add one dimension (if ndim = 2)
    ndim = img.ndim
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    # nanfilt_2D
    pad = (kernel_size - 1) // 2
    img_filt = bd_nanfilt(img, kernel_size=kernel_size, method=method)

    # Detect outliers
    residuals = img - img_filt
    residuals_sd = np.nanstd(residuals)
    outliers = np.zeros(img.shape)
    outliers[residuals > (residuals_sd) * sd_thresh] = 1
    outliers[residuals < (residuals_sd * -1) * sd_thresh] = 1

    # Replace outliers by NaNs
    img_nanoutliers = img.copy()
    img_nanoutliers[outliers == 1] = np.nan

    # Pad img border with NaNs
    img_pad = np.pad(
        img,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=np.nan,
    )

    # Find outliers coordinates
    idx = bd_where(outliers, 1)
    idx_t = idx[0].squeeze().astype("int")
    idx_y = idx[1].squeeze().astype("int")
    idx_x = idx[2].squeeze().astype("int")

    if idx:

        # Define kernels
        kernel_t = np.zeros([kernel_size, kernel_size], dtype=int)
        kernel_x, kernel_y = np.meshgrid(
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
        )
        idx_tt = idx_t[:, None, None] + kernel_t
        idx_xx = idx_x[:, None, None] + kernel_x
        idx_yy = idx_y[:, None, None] + kernel_y

        # Replace outliers
        functions = {
            'mean': np.nanmean, 
            'median': np.nanmedian, 
            'std': np.nanstd, 
            'max': np.nanmax, 
            'min': np.nanmin
            } 
        all_kernels = img_pad[idx_tt, idx_yy + pad, idx_xx + pad]
        all_kernels = functions[method](all_kernels, axis=(1, 2))
        for i in range(len(all_kernels)):
            img_nanoutliers[idx_t[i], idx_y[i], idx_x[i]] = all_kernels[i]

    # Remove one dimension (if ndim = 2)
    if ndim == 2:
        img = img.squeeze()
        img_nanoutliers = img_nanoutliers.squeeze()

    return img_nanoutliers

#%% Standalone exe

# import time
# from skimage import io

# # Path
# ROOT_PATH = 'C:/Datas/3-GitHub_BDehapiot/BD_USeg/data/'
# U_NAME = '13-12-06_40x_GBE_Ctrl_#19_Lite_uint8_u.tif'
# MASK_NAME = '13-12-06_40x_GBE_Ctrl_#19_Lite_uint8_u_mask.tif'

# # Parameters
# kernel_size = 7
# method = 'mean'
# sd_thresh = 1.5

# # Open data
# img = io.imread(ROOT_PATH + U_NAME)
# mask = io.imread(ROOT_PATH + MASK_NAME)

# # Get stack dimension
# nT = img.shape[0]
# nY = img.shape[1]
# nX = img.shape[2]

# ttest = 74
# img_test = img[ttest]
# mask_test = mask[ttest]

# start = time.time()
# print("Run time")

# img_filt = bd_nanoutliers(img_test, kernel_size, method, sd_thresh)
# img_filt = bd_nanreplace(img_filt, kernel_size, method, mask_test)
# img_filt = bd_nanfilt(img_filt, kernel_size, method)

# img_nanfilt = bd_nanfilt(img, kernel_size, method)
# img_nanreplace = bd_nanreplace(img, kernel_size, method, mask)
# img_nanoutliers = bd_nanoutliers(img, kernel_size, method, sd_thresh)

# end = time.time()
# print(f"  {(end - start):5.3f} s")

# io.imsave(ROOT_PATH+U_NAME[0:-4]+'_nanfilt.tif', img_nanfilt.astype("float32"), check_contrast=False)
# io.imsave(ROOT_PATH+U_NAME[0:-4]+'_nanreplace.tif', img_nanreplace.astype("float32"), check_contrast=False)
# io.imsave(ROOT_PATH+U_NAME[0:-4]+'_nanoutliers.tif', img_nanoutliers.astype("float32"), check_contrast=False)


#%% Plot results
# import matplotlib.pyplot as plt

# f, axes = plt.subplots(4, 1, figsize=(22, 22))
# axes[0].imshow(raw, cmap='gray')
# axes[1].imshow(raw_filt, cmap='gray')
# axes[2].imshow(outliers, cmap='gray')
# axes[3].imshow(raw_outfilt, cmap='gray')
# axes[0].title.set_text('raw')
# axes[1].title.set_text('raw_filt')
# axes[2].title.set_text('outliers')
# axes[3].title.set_text('raw_outfilt')
# plt.show()
