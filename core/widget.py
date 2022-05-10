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
rsize_full = imread(rsize_path) 
wat_full = imread(wat_path) 

# random seed
t = randint(0, rsize_full.shape[0]-1)
   
#%% Initialize Viewer

def correct_bounds(rsize, wat):

    class Viewer(napari.Viewer):
        pass
    
#%%

    user_inputs = np.zeros_like(wat)    

#%%

    viewer = Viewer()
    viewer.add_image(
        rsize, name='rsize',
        colormap='inferno',
        )
    viewer.add_image(
        wat, name='wat',
        colormap='gray',
        blending='additive',       
        )
    viewer.add_labels(
        user_inputs, 
        name='user_inputs'
        )

#%%

    @viewer.bind_key('p')       
    def hide_wat(viewer):
        viewer.layers['wat'].visible=False
        yield
        viewer.layers['wat'].visible=True

    @viewer.bind_key('Enter')       
    def apply_user_inputs(viewer):
        temp_inputs = viewer.layers['user_inputs']        
        wat_correct = viewer.layers['wat']
        wat_correct[temp_inputs != 0] = 0
        viewer.layers['wat'].data = wat_correct
        
        return wat_correct
        
#%%
    
    return user_inputs    
    
#%%

user_inputs  = correct_bounds(rsize_full[t,...], wat_full[t,...])

#%%

# test_inputs = user_inputs.copy()
# test_wat = wat.copy()
# test_wat[test_inputs != 0] = 0
# viewer = napari.Viewer()
# viewer.add_image(test_wat) 
