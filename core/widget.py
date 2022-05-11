#%% Imports

import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from skimage.io import imread

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
   
#%% Initialize
def correct_bounds(rsize, wat):
    
    watnew = wat.copy()
    inputs = np.zeros_like(wat)

    class Viewer(napari.Viewer):
        pass
     
    viewer = Viewer()
    
    viewer.add_image(
        rsize[0,...], 
        name='rsize',
        colormap='inferno',
        )
    
    viewer.add_image(
        watnew[0,...], 
        name='watnew',
        colormap='gray',
        blending='additive',       
        )  
    
    viewer.add_labels(
        inputs[0,...], 
        name='inputs',
        opacity=0.5,
        )
    
#%%

    @magicgui(
        
        auto_call = True,
               
        frame = {
            'widget_type': 'SpinBox', 
            'label': 'frame',
            'min': 0, 'max': wat.shape[0]-1,
            'visible': True,
            },
        
        )
    
    def display(
            frame: int,
            ):
        
        frame = display.frame.value       
        viewer.layers['rsize'].data = rsize[frame,...]
        viewer.layers['wat'].data = wat[frame,...]  
        
#%%

    viewer.window.add_dock_widget(display, area='right', name='widget') 
    
#%%

    @viewer.bind_key('Right')
    def next_frame(viewer):

        if display.frame.value < wat.shape[0]-1:
            display.frame.value += 1
            
    @viewer.bind_key('Left')
    def previous_frame(viewer):

        if display.frame.value > 0:
            display.frame.value -= 1
            
    # @viewer.bind_key('Enter')       
    # def apply_inputs(viewer):
               
    #     viewer.layers['watnew'].data[viewer.layers['inputs'].data != 0] = 0           
    #     viewer.layers['watnew'].data = (labconn(viewer.layers['watnew'].data) > 1)*255
    #     viewer.layers['inputs'].data = np.zeros_like(wat) 
        
             
#%%

wat = correct_bounds(rsize, wat)

#%%

# viewer = napari.Viewer()
# viewer.add_image(wat)
