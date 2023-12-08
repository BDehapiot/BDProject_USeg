#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from magicgui import magicgui
from skimage.transform import resize

from functions import get_watershed

#%% To do list ----------------------------------------------------------------

'''
- FileEdit to load images
- Progress bar during process, estimated time ?
- add info about shortcuts with 'viewer.text_overlay'

'''

#%% Open data -----------------------------------------------------------------

# File name
# raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8.tif'
raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
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
        )

    # Add shortcut info
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
    'p = hide watershed line \n' +  
    'left/right arrows = change frame'
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
            'min': 1, 'max': 100000, 'step': 1,
            'value': 100,
            },

        large_cell_cutoff = {
            'widget_type': 'SpinBox', 
            'label': 'large cell cutoff',
            'min': 1, 'max': 100000, 'step': 1,
            'value': 20000,
            },           

        remove_border_cells = {
            'widget_type': 'CheckBox',
            'label': 'remove border cells',
            'value': False, 
            },       
        
        show_all_layers = {
            'widget_type': 'CheckBox',
            'label': 'show all layers',
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
            show_all_layers: bool,
            ):
        
        # Get info
        t = preview.frame.value         
       
        # Get watershed (one frame)          
        output_dict = get_watershed(
            raw[t,...],
            preview.binning.value, 
            preview.ridge_size.value, 
            preview.thresh_coeff.value, 
            preview.small_cell_cutoff.value,
            preview.large_cell_cutoff.value,
            preview.remove_border_cells.value, 
            )
        
        if not show_all_layers:
        
            # Update raw for display
            if not viewer.layers.__contains__('raw'): 
                
                viewer.layers.clear()
                viewer.grid.enabled = False
                
                viewer.add_image(
                    raw[t,...], 
                    name='raw',
                    colormap='gray',
                    opacity=0.66
                    )
                
            else:
                
                viewer.layers['raw'].data = raw[t,...]
                viewer.layers['raw'].opacity = 0.66   

            # # Add shapes for cell size info
            # small_cell_cutoff = preview.small_cell_cutoff.value/np.square(binning)
            # small_radius = np.sqrt(np.sqrt(small_cell_cutoff)/np.pi)
            # small_circle = np.array([[100, 100], [small_radius, small_radius]])  
            # large_cell_cutoff = preview.large_cell_cutoff.value/np.square(binning)
            # large_radius = np.sqrt(np.sqrt(large_cell_cutoff)/np.pi)
            # large_circle = np.array([[100, 100], [large_radius, large_radius]]) 
            # cirles = [small_circle, large_circle]
            # viewer.add_shapes(cirles, shape_type='ellipse')

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
                
        else:

            # Remove raw & wat for display
            if viewer.layers.__contains__('raw'): 
                viewer.layers.remove('raw') 
                if viewer.layers.__contains__('wat'): 
                    viewer.layers.remove('wat') 

            # Update all layer display
            if viewer.layers.__contains__('rsize'): 
                
                viewer.layers['rsize'].data = output_dict['rsize']
                viewer.layers['ridges'].data = output_dict['ridges']
                viewer.layers['mask'].data = output_dict['mask']
                viewer.layers['markers'].data = output_dict['markers']
                viewer.layers['labels'].data = output_dict['labels']
                viewer.layers['wat'].data = output_dict['wat']
                viewer.layers['bound_labels'].data = output_dict['bound_labels']
                viewer.layers['rsize_norm'].data = output_dict['rsize_norm']
                viewer.layers['bound_int'].data = output_dict['bound_int']
              
            # Add all layer display
            else:                    
                
                viewer.add_image(
                    output_dict['rsize'], name='rsize')
                viewer.add_image(
                    output_dict['ridges'], name='ridges')
                viewer.add_image(
                    output_dict['mask'], name='mask')
                viewer.add_labels(
                    output_dict['markers'], name='markers')
                viewer.add_labels(
                    output_dict['labels'], name='labels')
                viewer.add_image(
                    output_dict['wat'], name='wat')
                viewer.add_labels(
                    output_dict['bound_labels'], name='bound_labels')
                viewer.add_image(
                    output_dict['rsize_norm'], name='rsize_norm')
                viewer.add_image(
                    output_dict['bound_int'], name='bound_int')
            
            viewer.grid.enabled = True

#%% Callbacks
        
    @preview.frame.changed.connect
    def frame_callback():
        
        # Get info
        t = preview.frame.value
        
        # Update raw for display
        if not viewer.layers.__contains__('raw'): 
            
            viewer.layers.clear()
            viewer.grid.enabled = False
            
            viewer.add_image(
                raw[t,...], 
                name='raw',
                colormap='gray',
                opacity=1
                )
            
        else:
            
            viewer.layers['raw'].data = raw[t,...]
            viewer.layers['raw'].opacity = 1  
        
        # Remove wat for display
        if viewer.layers.__contains__('wat'): 
            viewer.layers.remove('wat') 
               
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
             