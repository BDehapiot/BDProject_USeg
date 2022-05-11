#%% Imports

import napari
import numpy as np
from skimage.io import imread
from pathlib import Path

from tools.conn import labconn

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
   
#%% Initialize Viewer
def correct_bounds():

    class Viewer(napari.Viewer):
        pass
    
    Viewer.frame = 0
    
    
#%%      
        

#%%

    viewer = Viewer()
    
    viewer.add_image(
        rsize[Viewer.frame,...], 
        name='rsize',
        colormap='inferno',
        )
    
    viewer.add_image(
        wat[Viewer.frame,...], 
        name='wat',
        colormap='gray',
        blending='additive',       
        )
    
    viewer.add_labels(
        np.zeros_like(wat[Viewer.frame,...]), 
        name='inputs',
        opacity=0.5,
        )
    
    viewer.layers['inputs'].brush_size = 10
    viewer.layers['inputs'].mode = 'PAINT'

#%%

    @viewer.bind_key('h')       
    def hide_wat(viewer):
        
        viewer.layers['wat'].visible=False        
        yield
        viewer.layers['wat'].visible=True

    @viewer.bind_key('Enter')       
    def apply_inputs(viewer):
               
        wat_corrected = wat.copy()
        wat_corrected[viewer.layers['inputs'].data != 0] = 0           
        viewer.layers['wat'].data = (labconn(wat_corrected) > 1)*255
        viewer.layers['inputs'].data = np.zeros_like(wat) 
        
    @viewer.bind_key('Backspace') 
    def undo_inputs(viewer):

        viewer.layers['wat'].data = wat
         
#%%

    return wat 
    
#%%

wat = correct_bounds()

viewer = napari.Viewer()
viewer.add_image(wat)

#%%

# wat_conn = labconn(wat)
# viewer = napari.Viewer()
# viewer.add_image((wat_conn > 1)*255)
