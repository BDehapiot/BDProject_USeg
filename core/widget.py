#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from magicgui import magicgui
from skimage.transform import resize

from get_wat import get_wat

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
raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# raw_name = 'Disc_Fixed_118hAEL_disc04_uint8_crop.tif'
# raw_name = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

# Open data
raw = io.imread(Path('../data/', raw_name))

# Add one dimension (if ndim = 2)
ndim = (raw.ndim)        
if ndim == 2:
    raw = raw.reshape((1, raw.shape[0], raw.shape[1]))

#%% Initialize ----------------------------------------------------------------

def display_wat(raw):
    
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
    
    @magicgui(
        
        auto_call = False,
                       
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
            'value': 5,
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
        
        preview = {
            'widget_type': 'CheckBox',
            'label': 'preview',
            'value': False, 
            },
        
        process = {
            'widget_type': 'PushButton',
            'label': 'process and save',
            }
                
        )

    def display(
            frame: int,
            binning: int,
            ridge_size: float,
            thresh_coeff: float,
            small_cell_cutoff: int,
            large_cell_cutoff: int,
            remove_border_cells: bool,           
            preview: bool,
            process: bool,
            ):
        
        # Get info
        t = display.frame.value         
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f'frame = {t}'
                
        if not preview:
     
            # Initialize raw display
            if not viewer.layers.__contains__('raw'): 
               
                viewer.add_image(
                    raw[t,...], 
                    name='raw',
                    colormap='gray',
                    contrast_limits=(
                        np.quantile(raw, 0.001),
                        np.quantile(raw, 0.999),
                        ),
                    )
                                
                viewer.reset_view()
            
            # Update raw display (preview off)
            else:
                viewer.layers['raw'].data = raw[t,...]
                viewer.layers['raw'].opacity = 1

            if viewer.layers.__contains__('wat'): 
                viewer.layers.remove('wat') 
                
        else:
            
            # Update raw display (preview on)
            viewer.layers['raw'].data = raw[t,...]
            viewer.layers['raw'].opacity = 0.66
            
            # Get wat (one frame)       
            output_dict = get_wat(
                raw[t,...], 
                binning, 
                ridge_size, 
                thresh_coeff, 
                small_cell_cutoff,
                large_cell_cutoff,
                remove_border_cells=remove_border_cells, 
                parallel=False
                )
            
            # Process wat for display
            wat = output_dict['wat']
            watsize = resize(wat, (            
                int(wat.shape[0]*binning), 
                int(wat.shape[1]*binning)),
                preserve_range=True, 
                )   
            
            # Initialize wat display
            if not viewer.layers.__contains__('wat'): 
                
                viewer.add_image(
                    watsize, 
                    name='wat',
                    colormap='red',
                    contrast_limits=(0, 1),
                    blending='additive',
                    )
                
            # Update wat display    
            else:
                viewer.layers['wat'].data = watsize
                
        @display.process.changed.connect
        def process_callback():
            
            start = time.time()
            print('get_wat')
            
            # Get wat (all frames)          
            output_dict = get_wat(
                raw, 
                binning, 
                ridge_size, 
                thresh_coeff, 
                small_cell_cutoff,
                large_cell_cutoff,
                remove_border_cells=remove_border_cells, 
                parallel=True
                )
            
            end = time.time()
            print(f'  {(end-start):5.3f} s')
            
            # Save wat
            io.imsave(
                Path('../data/', raw_name.replace('.tif', '_wat.tif')),
                output_dict['wat'].astype('uint8')*255,
                check_contrast=False,
                )

    viewer.window.add_dock_widget(display, area='right', name='widget')
    
#%% Shortcuts

    @viewer.bind_key('Right')
    def next_frame(viewer):
                
        if display.frame.value < raw.shape[0]-1:
            display.frame.value += 1
            
    @viewer.bind_key('Left')
    def previous_frame(viewer):

        if display.frame.value > 0:
            display.frame.value -= 1   
            
    @viewer.bind_key('p', overwrite=True)
    def hide_wat(viewer):

        if viewer.layers.__contains__('wat'): 
            viewer.layers['wat'].visible = False
            yield
            viewer.layers['wat'].visible = True
    
#%% Run -----------------------------------------------------------------------

display_wat(raw)
             