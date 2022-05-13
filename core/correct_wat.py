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
        
    if 'ridges_027.tif' in path.name:
        ridges_path = path
        print(ridges_path.resolve())
        
# open data
rsize = imread(rsize_path) 
wat = imread(wat_path) 
inputs = imread(inputs_path) 
labels = imread(labels_path) 
ridges = imread(ridges_path) 

#%%

from skimage.morphology import label
from skimage.measure import regionprops_table
from tools.conn import labconn
from skimage.segmentation import watershed

#%%

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
                
watnew = wat.copy()    
watnew[inputs_remove != 0] = 0
watnew = labconn(watnew) > 1

unique = np.unique(labels[inputs_add != 0])
for idx in unique:
    
    mask = labels == idx
    temp_wat = watershed(ridges, inputs_add, mask=mask, watershed_line=True)
    temp_wat = temp_wat > 0
    temp_wat = temp_wat != mask
    watnew[temp_wat == True] = True

end = time.time()
print(f'  {(end - start):5.5f} s')  

#%%

imsave(Path(data_path /'inputs_add.tif'), inputs_add, check_contrast=False)        
imsave(Path(data_path /'inputs_remove.tif'), inputs_remove, check_contrast=False)   

#%%
import napari

viewer = napari.Viewer()
viewer.add_image(watnew)

#%%
