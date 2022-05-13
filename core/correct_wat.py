#%% Imports

import time
import numpy as np
from pathlib import Path
from skimage.io import imread, imsave

#%% Initialize

# Create paths
root_path = Path(__file__).parents[1]
data_path = Path(root_path / 'data' / 'correct_wat')

for path in data_path.iterdir():
    
    if 'rsize_027.tif' in path.name:
        rsize_path = path
        print(rsize_path.resolve())
    
    if 'wat_027.tif' in path.name:
        wat_path = path
        print(wat_path.resolve())
        
    if 'inputs_027.tif' in path.name:
        inputs_path = path
        print(inputs_path.resolve())
       
    if 'labels_027.tif' in path.name:
        labels_path = path
        print(labels_path.resolve())
        
# open data
rsize = imread(rsize_path) 
wat = imread(wat_path) 
inputs = imread(inputs_path) 
labels = imread(labels_path) 

#%%

from skimage.morphology import label
from skimage.measure import regionprops_table

start = time.time()
print('xxx')

wat_labels = labels.copy()
wat_labels[wat != 0] = 0
input_labels = label(inputs)

props = regionprops_table(
    input_labels, wat_labels, 
    properties=('label', 'centroid', 'intensity_min')
    )

inputs_remove = inputs.copy()
inputs_add = np.zeros_like(inputs_remove)
for idx, ctrd_y, ctrd_x, int_min in zip(
        props['label'], 
        props['centroid-0'], 
        props['centroid-1'], 
        props['intensity_min']
        ):
    
    if int_min != 0:
        inputs_add[round(ctrd_y), round(ctrd_x)] = idx
        inputs_remove[input_labels == idx] = 0
                
# -----------------------------------------------------------------------------

from tools.conn import labconn

watnew = wat.copy()    
watnew[inputs_remove != 0] = 0
watnew = (labconn(watnew) > 1) * 255

# -----------------------------------------------------------------------------

# from scipy import ndimage

# for idx in np.unique(labels[inputs_add != 0]):
    
#     seeds = inputs_add.copy()
#     seeds[labels != idx] = 0
#     edm = ndimage.distance_transform_edt(np.invert(seeds))



end = time.time()
print(f'  {(end - start):5.5f} s')  
        
#%%

imsave(Path(data_path /'inputs_add.tif'), inputs_add, check_contrast=False)        
imsave(Path(data_path /'inputs_remove.tif'), inputs_remove, check_contrast=False)   

#%%
# import napari

# viewer = napari.Viewer()
# viewer.add_image(edm)

#%%

# from skimage.segmentation import watershed

# x, y = np.indices((80, 80))
# x1, y1, x2, y2 = 28, 28, 44, 52
# r1, r2 = 16, 20
# mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
# mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
# image = np.logical_or(mask_circle1, mask_circle2)

# from scipy import ndimage as ndi
# distance = ndi.distance_transform_edt(image)
# from skimage.feature import peak_local_max
# max_coords = peak_local_max(distance, labels=image,
#                              footprint=np.ones((3, 3)))
# local_maxima = np.zeros_like(image, dtype=bool)
# local_maxima[tuple(max_coords.T)] = True
# markers = ndi.label(local_maxima)[0]

# labels = watershed(-distance, markers, mask=image)
