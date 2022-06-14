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
    
    # print(labels.shape)
    # print(labels.dtype)
    # print(wat_labels.shape)
    # print(wat_labels.dtype)
    
    # imsave(Path(data_path /'labels.tif'), labels, check_contrast=False) 
    # imsave(Path(data_path /'wat_labels.tif'), wat_labels, check_contrast=False)    

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
        
        print(int_min)
        
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
        
    # imsave(Path(data_path /'inputs_add.tif'), inputs_add, check_contrast=False)        
    # imsave(Path(data_path /'inputs_remove.tif'), inputs_remove, check_contrast=False) 

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
    
    # viewer.add_image(
    #     labels[0,...], 
    #     name='labels',
    #     colormap='GrBu',
    #     opacity = 0.33,     
    #     )     
    
    viewer.add_labels(
        labels[0,...], 
        name='labels',
        opacity = 0.33,     
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
        viewer.layers['labels'].data = labels[i,...] 
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
    def add_inputs(viewer):
        
        # Update inputs
        i = display.frame.value 
        inputs[i,...][inputs[i,...] != 0] += 1        
        inputs[i,...] = np.maximum(
            inputs[i,...], viewer.layers['inputs'].data)
        
        # # Apply inputs
        # watnew[i,...] = wat[i,...]
        # watnew[i,...][inputs[i,...] != 0] = 0
        # watnew[i,...] = (labconn(watnew[i,...]) > 1) * 255
        
        # Apply inputs (new)
        watnew[i,...] = apply_inputs(
            inputs[i,...], ridges[i,...], watnew[i,...]
            )

        # Update layers
        viewer.layers['inputs'].data = inputs_empty.copy()
        viewer.layers['watnew'].data = watnew[i,...]
        viewer.layers['labels'].data = label(np.invert(watnew[i,...])) 

    @viewer.bind_key('Backspace')
    def remove_inputs(viewer):
        
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

inputs = correct_wat(rsize, ridges, wat)

#%%

# from skimage import io
# io.imsave(Path(data_path /'inputs_test.tif'), inputs[26,...].astype('uint8')*255, check_contrast=False)

#%%

# viewer = napari.Viewer()
# viewer.add_image(inputs[26,...])