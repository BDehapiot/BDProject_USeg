#%% Imports -------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.restoration import rolling_ball
from skimage.measure import label, regionprops
from skimage.transform import downscale_local_mean
from skimage.filters import sato, threshold_li, gaussian
from skimage.segmentation import watershed, clear_border, expand_labels
from skimage.morphology import binary_dilation, remove_small_objects, square

# Custom
from skel import labconn
from nan import nanreplace

#%% Comments ------------------------------------------------------------------

'''
- stop using cell_info and bound_info as dataframe
'''

#%% Execute -------------------------------------------------------------------

# Paths
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8_lite.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
raw_name = 'Disc_Fixed_118hAEL_disc04_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Open data
raw_path = Path('data', raw_name)
raw = io.imread(raw_path)

# -----------------------------------------------------------------------------

# Parameters
binning = 2
ridge_size = 4
thresh_coeff = 0.4
small_cell_cutoff = 5
large_cell_cutoff = 10
remove_border_cells = False

# -----------------------------------------------------------------------------

def get_wat(
        raw, 
        binning, 
        ridge_size, 
        thresh_coeff, 
        small_cell_cutoff,
        large_cell_cutoff,
        parallel=False
        ):

    # Nested functions --------------------------------------------------------    

    def _get_wat(frame, img):
        
        # Resize
        rsize = downscale_local_mean(img, binning)
        
        # Subtract background
        rsize -= rolling_ball(
            gaussian(rsize, sigma=ridge_size // 2), radius=ridge_size * 2,
            )    
        
        # ridges filter 
        ridges = sato(
            rsize, sigmas=[ridge_size], mode='reflect', black_ridges=False,
            )
        
        # Get mask
        thresh = threshold_li(ridges, tolerance=1)
        mask = ridges > thresh * thresh_coeff
        mask = remove_small_objects(mask, min_size=np.sum(mask) * 0.01)
        markers = label(np.invert(mask), connectivity=1)
        
        # Filter cells (according to area)
        props = regionprops(markers)
        cell_info = pd.DataFrame(
            ([(frame, prop.label, prop.area, 0) for prop in props]),
            columns = ['frame', 'label', 'area', 'valid']
            )
        median = np.median(cell_info['area'])
        for i in range(len(cell_info)):
            
            # Detect & remove small cells
            if cell_info.loc[i,'area'] < median / small_cell_cutoff: 
                cell_info.loc[i,'valid'] = 1
                markers[markers==cell_info['label'][i]] = 0  
            
            # Detect large cells
            if cell_info.loc[i,'area'] > median * large_cell_cutoff: 
                cell_info.loc[i,'valid'] = 2                
                
        # Get watershed labels
        labels = watershed(
            ridges, 
            markers, 
            compactness=10 / binning, 
            watershed_line=True
            ) 

        # Remove large cells
        for idx in cell_info.loc[cell_info['valid'] == 2, 'label']:
            labels[labels==idx] = 0
            
        # Remove border cells
        if remove_border_cells:
            labels = clear_border(labels)
            
        # Update cell info
        props = regionprops(labels)
        cell_info = pd.DataFrame(
            ([(frame, prop.label, prop.area) for prop in props]),
            columns = ['frame', 'label', 'area']
            )

        # Get watershed wat
        wat = labels == 0
        temp = np.invert(wat)
        temp = binary_dilation(temp)
        wat = np.minimum(wat, temp)
        
        # Get vertices
        vertices = labconn(wat, conn=2) > 2
        
        # Get bounds & endpoints
        bounds = wat.copy()
        bounds[binary_dilation(vertices, square(3)) == 1] = 0
        endpoints = wat ^ bounds ^ vertices
        
        # Label bounds
        bound_labels = label(bounds, connectivity=2).astype('float')
        bound_labels = expand_labels(bound_labels, distance=1.5)
        bound_labels[wat == 0] = 0
               
        bound_labels = bound_labels.astype('int')
        small_bounds = wat ^ (bound_labels > 0) ^ vertices
        small_bound_labels = label(small_bounds, connectivity=2)
        small_bound_labels = small_bound_labels + np.max(bound_labels)
        small_bound_labels[small_bound_labels == np.max(bound_labels)] = 0
        bound_labels = bound_labels + small_bound_labels

        # Get bound info
        props = regionprops(
            bound_labels, 
            intensity_image=(gaussian(rsize, ridge_size))
            )

        # bound_info = pd.DataFrame(([
        #     (frame, prop.label, prop.coords, prop.intensity_mean) 
        #     for prop in props]),
        #     columns = ["frame", "label", "coords", "intensity"],
        #     )
        
        bound_info = [(
            frame, prop.label, prop.intensity_mean, 
            prop.coords
            ) 
            for prop in props
            ]
        
        return (
            rsize, ridges, mask, markers, labels, wat, vertices, bound_labels,
            cell_info, bound_info
            )
        
    # Main function -----------------------------------------------------------

    # Add one dimension (if ndim == 2)
    ndim = (raw.ndim)        
    if ndim == 2:
        raw = raw.reshape((1, raw.shape[0], raw.shape[1]))  
        
    # Adjust parameters acc. to binning
    ridge_size = ridge_size / binning
    
    if parallel:
    
        # Run parallel
        outputs = Parallel(n_jobs=-1)(
            delayed(_get_wat)(frame, img)
            for frame, img in enumerate(raw)
            )
        
    else:
        
        # Run serial
        outputs = [_get_wat(frame, img)
            for frame, img in enumerate(raw)
            ]
        
    # Append data dict.
    data = {
        'rsize': np.stack(
            [data[0] for data in outputs], axis=0).squeeze(),
        'ridges': np.stack(
            [data[1] for data in outputs], axis=0).squeeze(),
        'mask': np.stack(
            [data[2] for data in outputs], axis=0).squeeze(),
        'markers': np.stack(
            [data[3] for data in outputs], axis=0).squeeze(),
        'labels': np.stack(
            [data[4] for data in outputs], axis=0).squeeze(),
        'wat': np.stack(
            [data[5] for data in outputs], axis=0).squeeze(),
        'vertices': np.stack(
            [data[6] for data in outputs], axis=0).squeeze(),
        'bound_labels': np.stack(
            [data[7] for data in outputs], axis=0).squeeze(),
        # 'cell_info' : pd.concat(
        #     [(data[8]) for i, data in enumerate(outputs)]),
        # 'bound_info' : pd.concat(
        #     [(data[9]) for i, data in enumerate(outputs)]),      
        # 'bound_info' : [(data[9]) for i, data in enumerate(outputs)],  
        'bound_info' : [item for data in outputs for item in data[9]],
        }

    return data

# -----------------------------------------------------------------------------

print("get_wat:", end='')
t0 = time.time()

data = get_wat(
    raw, binning, 
    ridge_size, 
    thresh_coeff,
    small_cell_cutoff,
    large_cell_cutoff,
    parallel=True
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(data["rsize"], name="rsize")
# viewer.add_image(data["ridges"], name="ridges")
# viewer.add_image(data["mask"], name="mask")
# viewer.add_image(data["markers"], name="markers")

#%% Experiment ----------------------------------------------------------------

# -----------------------------------------------------------------------------

img = data["rsize"]
# mask = data["mask"]
mask = ~data["mask"]
sigma = 20

def masked_gaussian(img, sigma, mask):

    np.seterr(divide='ignore', invalid='ignore')

    img = img.astype(float)
    img[mask == 0] = 0
    img = gaussian(img, sigma=sigma, preserve_range=True)
    mask = mask.astype(int)
    mask = gaussian(mask, sigma=sigma, preserve_range=True)
    return img / mask

io.imsave(
    Path("data", "temp", raw_path.stem + "_mask_blur.tif"),
    mask_blur.astype("float32"), check_contrast=False,
    )

# -----------------------------------------------------------------------------

temp_mask = data["mask"]#[0]
rsize = data["rsize"]#[0]

# Get bg_blur (with a nan ignoring Gaussian blur)
temp1 = rsize.astype('float')
temp1[temp_mask == 0] = np.nan

temp2 = temp1.copy()
temp2[np.isnan(temp2)] = 0
temp2_blur = gaussian(temp2, sigma=ridge_size * 10)

temp3 = 0 * temp1.copy() + 1
temp3[np.isnan(temp3)] = 0
temp3_blur = gaussian(temp3, sigma=ridge_size * 10)

np.seterr(divide='ignore', invalid='ignore')
bg_blur = temp2_blur / temp3_blur

# Get rsize_blur (divided by bg_blur)
# bound_norm = gaussian(rsize, sigma=ridge_size//2)
# bound_norm = bound_norm / bg_blur * 1.5

bound_norm = rsize / bg_blur * 1.5


io.imsave(
    Path("data", "temp", raw_path.stem + "_bound_norm.tif"),
    bound_norm.astype("float32"), check_contrast=False,
    )

# from sklearn.mixture import GaussianMixture

# bound_info = data["bound_info"]
# bound_labels = data["bound_labels"].astype(float)
# intensities = np.stack([data[2] for data in bound_info]).reshape(-1, 1)
# frames = [data[0] for data in bound_info]
# coords = [data[3] for data in bound_info]

# gmm = GaussianMixture(n_components=2, random_state=0)
# gmm.fit(intensities)
# probabilities = gmm.predict_proba(intensities)[:, 0]

# for i, (frame, coord) in enumerate(zip(frames, coords)):
#     coord = (coord[:, 0], coord[:, 1])
#     bound_labels[coord] = intensities[i]

# io.imsave(
#     Path("data", "temp", raw_path.stem + "_bound_probs.tif"),
#     bound_labels.astype("float32"), check_contrast=False,
#     )


#%% Save ----------------------------------------------------------------------
  
io.imsave(
    Path("data", "temp", raw_path.stem + "_rsize.tif"),
    data["rsize"].astype("float32"), check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_ridges.tif"),
    data["ridges"].astype("float32"), check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_mask.tif"),
    data["mask"].astype("uint8") * 255, check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_markers.tif"),
    data["markers"].astype("uint16"), check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_labels.tif"),
    data["labels"].astype("uint16"), check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_wat.tif"),
    data["wat"].astype("uint8") * 255, check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_vertices.tif"),
    data["vertices"].astype("uint8") * 255, check_contrast=False,
    )

io.imsave(
    Path("data", "temp", raw_path.stem + "_bound_labels.tif"),
    data["bound_labels"].astype("uint16"), check_contrast=False,
    )

