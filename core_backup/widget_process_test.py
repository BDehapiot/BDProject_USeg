#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path
from magicgui import magicgui
from qtpy.QtWidgets import QWidget

#%%

class Display:
   
    def __init__(self, viewer : napari.Viewer):
        self.viewer = viewer
        
display = Display(napari.Viewer)   

#%%      

def useg():  

#%% Open data
    
    @magicgui(
        
        auto_call = True,
                       
        raw_path = {
            'widget_type': 'FileEdit', 
            'label': 'Select raw image or stack',
            'value': Path(__file__).parents[1] / 'data'
            },
       
        )
    
    def open_raw(
            raw_path: str,
            ):
        
        # Open raw
        raw = io.imread(raw_path) 
                
        # Add one dimension (if ndim = 2)
        ndim = (raw.ndim)        
        if ndim == 2:
            raw = raw.reshape((1, raw.shape[0], raw.shape[1]))
        
        # Display raw in viewer
        display.add_image(
            raw[raw.shape[0]//2,...], 
            name='raw',
            colormap='gray',
            contrast_limits=(
                np.quantile(raw, 0.001),
                np.quantile(raw, 0.999),
                ),
            )


#%%

    display.viewer.window.add_dock_widget(open_raw, area='right', name='widget')

#%%

useg()