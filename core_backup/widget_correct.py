#%% Imports

import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from skimage.io import imread, imsave

from tools.conn import labconn

#%% Functions

from skimage.morphology import label
from skimage.segmentation import watershed
from skimage.measure import regionprops_table

def apply_inputs(inputs, ridges, wat):
    
    input_labels = label(inputs != 0)
    labels = label(np.invert(wat), connectivity=1)
    wat_labels = labels.copy()
    wat_labels[wat != 0] = 0

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
    watnew = ((labconn(watnew) > 1)*255).astype('uint8')
    
    unique = np.unique(labels[inputs_add != 0])
    for idx in unique:        
        mask = labels == idx
        temp_wat = watershed(
            ridges, inputs_add, mask=mask, compactness=1, watershed_line=True
            )
        temp_wat = temp_wat > 0
        temp_wat = temp_wat != mask
        watnew[temp_wat == True] = 255
    
    return watnew


#%% Open data

# Create paths
root_path = Path(__file__).parents[1]
data_path = Path(root_path / 'data' / 'temp')

for path in data_path.iterdir():
    
    if 'uint8_rsize.tif' in path.name:
        rsize_path = path
        
    if 'uint8_ridges.tif' in path.name:
        ridges_path = path     
        
    if 'uint8_labels.tif' in path.name:
        labels_path = path 
    
    if 'uint8_wat.tif' in path.name:
        wat_path = path      

rsize = imread(rsize_path) 
ridges = imread(ridges_path) 
labels = imread(labels_path) 
wat = imread(wat_path) 

#%% Initialize

def correct_wat(rsize, ridges, wat):
  
    watnew = wat.copy()
    labnew = labels.copy()
    inputs = np.zeros_like(wat[0,...])
    
    class Viewer(napari.Viewer):
        pass
    
    # Create watlist
    Viewer.watlist = [
        [wat[i,...]] for i in range(wat.shape[0])
        ] 
         
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
        labnew[0,...], 
        name='labnew',
        opacity = 0.33,     
        )  
    
    viewer.add_labels(
        inputs.copy(), 
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
        viewer.layers['labnew'].data = labnew[i,...] 
        viewer.layers['inputs'].data = inputs.copy() 

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
    def add_inputs(viewer):
        
        # Extract info
        i = display.frame.value 
        
        # Apply inputs
        watnew[i,...] = apply_inputs(
            viewer.layers['inputs'].data, 
            ridges[i,...], watnew[i,...]
            )        
        labnew[i,...] = label(
            np.invert(watnew[i,...]), connectivity=1
            ) - 1
                
        # Update layers
        viewer.watlist[i].append(watnew[i,...].copy())
        viewer.layers['watnew'].data = watnew[i,...]
        viewer.layers['labnew'].data = labnew[i,...] 
        viewer.layers['inputs'].data = inputs.copy()
        
    @viewer.bind_key('Backspace')
    def remove_inputs(viewer):
        
        # Extract info
        i = display.frame.value 
        
        if len(viewer.watlist[i]) > 1:
            
            # Apply inputs
            watnew[i,...] = viewer.watlist[i][-2]          
            labnew[i,...] = label(
                np.invert(watnew[i,...]), connectivity=1
                ) - 1
            
            # Update layers
            viewer.watlist[i].pop()
            viewer.layers['watnew'].data = watnew[i,...]
            viewer.layers['labnew'].data = labnew[i,...] 
            viewer.layers['inputs'].data = inputs.copy()
            
#%%
        
    return inputs   
             
#%%

inputs = correct_wat(rsize, ridges, wat)

#%%

# from skimage import io
# io.imsave(Path(data_path /'inputs_test.tif'), inputs[26,...].astype('uint8')*255, check_contrast=False)

#%%

# viewer = napari.Viewer()
# viewer.add_image(inputs[26,...])
