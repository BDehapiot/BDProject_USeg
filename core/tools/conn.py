#%% 

import numpy as np

from skimage.morphology import label

#%%

from tools.idx import rwhere

#%% pixconn

def pixconn(img, conn=2):
    
    conn1_selem = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
    
    conn2_selem = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])

    # Binarize input image
    img = img.astype('int')
    img[img > 0] = 1
    
    # Add one dimension (if ndim = 2)
    ndim = (img.ndim)        
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
        
    # Pad labels border with zeros
    img_pad = np.pad(img, pad_width=((0, 0), (1, 1), (1, 1)),
                      mode='constant', constant_values=0)   
    
    # Find wat non-zero coordinates
    idx = rwhere(img > 0, True) 
    idx_t = idx[0].squeeze().astype('int')
    idx_y = idx[1].squeeze().astype('int')
    idx_x = idx[2].squeeze().astype('int')
    
    # Define kernels
    kernel_t = np.zeros([3,3], dtype=int)
    kernel_x, kernel_y = np.meshgrid(
        np.arange(-1, 2),
        np.arange(-1, 2)
        )
    idx_tt = idx_t[:, None, None] + kernel_t
    idx_xx = idx_x[:, None, None] + kernel_x
    idx_yy = idx_y[:, None, None] + kernel_y  
    
    # Filter image   
    img_pixconn = np.zeros(img.shape)
    all_kernels = img_pad[idx_tt, idx_yy + 1, idx_xx + 1]    
    if conn == 1:
        all_kernels = all_kernels*conn1_selem
    if conn == 2:    
        all_kernels = all_kernels*conn2_selem
        
    all_kernels = np.sum(all_kernels, axis=(1,2))      
    for i in range(len(all_kernels)):
        img_pixconn[idx_t[i],idx_y[i],idx_x[i]] = all_kernels[i]  
        
    # Remove one dimension (if ndim = 2)    
    if ndim == 2:
        img_pixconn = img_pixconn.squeeze()

    return img_pixconn

#%% labconn

def labconn(img, labels=None, conn=2):
    
    conn1_selem = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    
    conn2_selem = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
    
    # Binarize input img
    img = img.astype('int')
    img[img > 0] = 1
    
    # Add one dimension (if ndim = 2)
    ndim = (img.ndim)        
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
        if labels is not None:
            labels = labels.reshape((1, labels.shape[0], labels.shape[1]))
    
    # Create labels if None
    if labels is None:
        labels = np.zeros(img.shape)        
        for i in range(img.shape[0]):
            labels[i,...] = label(np.invert(img[i,...]), connectivity=1)
 
    # Replace wat by NaNs on labels
    labels = labels.astype('float')
    labels[img == 1] = np.nan
    
    # Pad labels border with NaNs
    labels_pad = np.pad(labels, pad_width=((0, 0), (1, 1), (1, 1)),
                      mode='constant', constant_values=np.nan)
                  
    # Find wat non-zero coordinates
    idx = rwhere(img > 0, True) 
    idx_t = idx[0].squeeze().astype('int')
    idx_y = idx[1].squeeze().astype('int')
    idx_x = idx[2].squeeze().astype('int')
    
    # Define kernels
    kernel_t = np.zeros([3,3], dtype=int)
    kernel_x, kernel_y = np.meshgrid(
        np.arange(-1, 2),
        np.arange(-1, 2)
        )
    idx_tt = idx_t[:, None, None] + kernel_t
    idx_xx = idx_x[:, None, None] + kernel_x
    idx_yy = idx_y[:, None, None] + kernel_y
    
    # Filter image   
    img_labconn = np.zeros(labels.shape)
    all_kernels = labels_pad[idx_tt, idx_yy + 1, idx_xx + 1]    
    if conn == 1:
        all_kernels = all_kernels*conn1_selem
    if conn == 2:    
        all_kernels = all_kernels*conn2_selem
        
    # Get unique values (vectorized)
    all_kernels = all_kernels.reshape((all_kernels.shape[0], -1))
    all_kernels.sort(axis=1)  
    img_labconn[idx_t,idx_y,idx_x] = (np.diff(all_kernels) > 0).sum(axis=1)+1
        
    # for i in range(len(all_kernels)):
    #     kernel = all_kernels[i,...].ravel()
    #     kernel = kernel[~np.isnan(kernel)]
    #     img_labconn[idx_t[i],idx_y[i],idx_x[i]] = len(np.unique(kernel))
        
    # Remove one dimension (if ndim = 2)    
    if ndim == 2:
        img_labconn = img_labconn.squeeze()
        
    return img_labconn
