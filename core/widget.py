#%% Imports

import napari
import numpy as np
from random import randint
from skimage.io import imread
from pathlib import Path

#%% Initialize

# Create paths
root_path = Path(__file__).parents[1]
data_path = Path(root_path / 'data' / 'temp')
for path in data_path.iterdir():
    if 'rsize.tif' in path.name:
        rsize_path = path
        print(rsize_path.resolve())
    if 'wat.tif' in path.name:
        wat_path = path
        print(wat_path.resolve())

        
# open data
rsize = imread(rsize_path) 
wat = imread(wat_path) 

# random seed
t = randint(0, rsize.shape[0]-1)
    
#%% Initialize Viewer

def correct_bounds():

    class Viewer(napari.Viewer):
        pass

#%%

    viewer = Viewer()
    viewer.add_image(
        rsize[t,...], name=f'rsize t={t}',
        colormap='inferno',
        )
    viewer.add_image(
        wat[t,...], name=f'wat t={t}',
        colormap='gray',
        blending='additive',        
        )

#%%

    @viewer.bind_key('Enter')       
    def hide_wat(viewer):
        print('hello')
        yield
        print('goodbye')

    
#%%

correct_bounds()
    
