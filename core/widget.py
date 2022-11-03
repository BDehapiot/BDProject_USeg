#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from magicgui import magicgui
from skimage.transform import resize

from functions import pre_processing, get_watershed, get_bounds, useg

#%% To do list ----------------------------------------------------------------

'''
- FileEdit to load images
- Progress bar during process, estimated time ?
- See, process and save intermediate steps (rsize, ridges, mask, labels...)
- add info about shortcuts with 'viewer.text_overlay'

'''

#%% Open data -----------------------------------------------------------------

# File name
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Open data
raw = io.imread(Path('../data/', raw_name))

# Add one dimension (if ndim = 2)
ndim = (raw.ndim)        
if ndim == 2:
    raw = raw.reshape((1, raw.shape[0], raw.shape[1]))

#%% Initialize ----------------------------------------------------------------

def widget(raw):
    
    class Viewer(napari.Viewer):
        pass
    
    viewer = Viewer()
    
    viewer.add_image(
        raw[raw.shape[0]//2,...], 
        name='raw',
        colormap='gray',
        contrast_limits=(
            np.quantile(raw, 0.001),
            np.quantile(raw, 0.999),
            ),
        )
    
#%% Preview -------------------------------------------------------------------
    
    @magicgui(
        
        auto_call = False,
        call_button="Preview",
                       
        frame = {
            'widget_type': 'SpinBox', 
            'label': 'frame',
            'min': 0, 'max': raw.shape[0]-1,
            'value': raw.shape[0]//2,
            },
        
        binning = {
            'widget_type': 'SpinBox', 
            'label': 'binning',
            'min': 1, 'max': 5, 'step': 1,
            'value': 2,
            },
        
        ridge_size = {
            'widget_type': 'FloatSpinBox', 
            'label': 'ridge size',
            'min': 0, 'max': 10, 'step': 0.5,
            'value': 3,
            },
        
        thresh_coeff = {
            'widget_type': 'FloatSpinBox', 
            'label': 'thresh. coeff.',
            'min': 0, 'step': 0.1,
            'value': 0.5,
            },

        small_cell_cutoff = {
            'widget_type': 'SpinBox', 
            'label': 'small cell cutoff',
            'min': 1, 'max': 20, 'step': 1,
            'value': 10,
            },

        large_cell_cutoff = {
            'widget_type': 'SpinBox', 
            'label': 'large cell cutoff',
            'min': 1, 'max': 20, 'step': 1,
            'value': 10,
            },           

        remove_border_cells = {
            'widget_type': 'CheckBox',
            'label': 'remove border cells',
            'value': False, 
            },        
                        
        )

    def preview(
            frame: int,
            binning: int,
            ridge_size: float,
            thresh_coeff: float,
            small_cell_cutoff: int,
            large_cell_cutoff: int,
            remove_border_cells: bool,           
            ):
        
        # Get info
        t = preview.frame.value         
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f'frame = {t}'
       
        # Get preview (one frame)          
        output_dict = useg(
            raw[t,...],
            preview.binning.value, 
            preview.ridge_size.value/preview.binning.value, 
            preview.thresh_coeff.value, 
            preview.small_cell_cutoff.value,
            preview.large_cell_cutoff.value,
            preview.remove_border_cells.value, 
            )
        
        # Update raw for display
        viewer.layers['raw'].data = raw[t,...]
        viewer.layers['raw'].opacity = 0.66
        
        # Resize wat for display
        wat = output_dict['wat']
        watsize = resize(wat, (            
            int(wat.shape[0]*binning), 
            int(wat.shape[1]*binning)),
            preserve_range=True, 
            ) 
        
        # Add wat for display (initialization)
        if not viewer.layers.__contains__('wat'): 
            
            viewer.add_image(
                watsize, 
                name='wat',
                colormap='red',
                contrast_limits=(0, 1),
                blending='additive',
                )
            
        # Update wat for display   
        else:
            viewer.layers['wat'].data = watsize
        
        
                                   
#%% Shortcuts -----------------------------------------------------------------

    @viewer.bind_key('Right')
    def next_frame(viewer):
                
        if preview.frame.value < raw.shape[0]-1:
            preview.frame.value += 1
            
    @viewer.bind_key('Left')
    def previous_frame(viewer):

        if preview.frame.value > 0:
            preview.frame.value -= 1   
            
    @viewer.bind_key('p', overwrite=True)
    def hide_wat(viewer):

        if viewer.layers.__contains__('wat'): 
            viewer.layers['wat'].visible = False
            yield
            viewer.layers['wat'].visible = True
            
#%% Add dock widget -----------------------------------------------------------
    
    viewer.window.add_dock_widget(
        [preview], 
        area='right', 
        name='widget'
        )

#%% Run -----------------------------------------------------------------------

widget(raw)
             