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
def correct_wat(rsize, wat):
    
    watnew = wat.copy()
    inputs = np.zeros_like(wat)
    inputs_empty = inputs[0,...].copy()

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
        inputs_empty.copy(), 
        name='inputs',
        opacity=0.5,
        )
    
    viewer.layers['inputs'].brush_size = 6
    viewer.layers['inputs'].mode = 'PAINT'
    
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
        
        i = display.frame.value       
        viewer.layers['rsize'].data = rsize[i,...]
        viewer.layers['watnew'].data = watnew[i,...] 
        viewer.layers['inputs'].data = inputs_empty.copy() 
        
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
  
    @viewer.bind_key('Enter')       
    def apply_inputs(viewer):
        
        # update inputs
        i = display.frame.value 
        inputs[i,...][inputs[i,...] != 0] += 1        
        inputs[i,...] = np.maximum(
            inputs[i,...], viewer.layers['inputs'].data)
        
        # apply inputs
        watnew[i,...] = wat[i,...]
        watnew[i,...][inputs[i,...] != 0] = 0
        watnew[i,...] = (labconn(watnew[i,...]) > 1) * 255
        
        # update layers
        viewer.layers['inputs'].data = inputs_empty.copy()
        viewer.layers['watnew'].data = watnew[i,...]

    @viewer.bind_key('Backspace')
    def undo_last_input(viewer):
        
        # update inputs
        i = display.frame.value 
        inputs[i,...][inputs[i,...] != 0] -= 1  
        
        # apply inputs
        watnew[i,...] = wat[i,...]
        watnew[i,...][inputs[i,...] != 0] = 0
        watnew[i,...] = (labconn(watnew[i,...]) > 1) * 255
        
        # update layers
        viewer.layers['inputs'].data = inputs_empty.copy()
        viewer.layers['watnew'].data = watnew[i,...]

#%%
        
    return inputs            
             
#%%

inputs = correct_wat(rsize, wat)

#%%

# from skimage import io
# io.imsave(Path(data_path /'inputs_test.tif'), inputs[26,...].astype('uint8')*255, check_contrast=False)

#%%

# viewer = napari.Viewer()
# viewer.add_image(inputs[26,...])