#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
from magicgui import magicgui
from skimage.transform import resize

from get_wat import get_wat

#%% Open data -----------------------------------------------------------------

# File name
raw_name = '13-12-06_40x_GBE_eCad_Ctrl_#19_uint8.tif'
# raw_name = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif'
# raw_name = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
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
        
        auto_call = True,
                       
        frame = {
            'widget_type': 'SpinBox', 
            'label': 'frame',
            'min': 0, 'max': raw.shape[0]-1,
            'value': raw.shape[0]//2,
            },
        
        binning = {
            'widget_type': 'SpinBox', 
            'label': 'binning',
            'min': 1, 'max': 4, 'step': 1,
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
            ):
        
        # Get info
        t = display.frame.value         
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f'frame = {t}'
                
        # Show raw image 
        if not preview:
            
            if viewer.layers.__contains__('raw'):               
                viewer.layers['raw'].data = raw[t,...]
                
            else:
                
                if viewer.layers.__contains__('rsize'): 
                    viewer.layers.remove('rsize') 
                
                if viewer.layers.__contains__('wat'): 
                    viewer.layers.remove('wat') 
                
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
                
        else:
            
            # Get wat preview            
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
            
            # Back size wat for display
            wat = output_dict['wat']
            watsize = resize(wat, (            
                int(wat.shape[0]*binning), 
                int(wat.shape[1]*binning)),
                preserve_range=True, 
                anti_aliasing=True,
                )    
            
            if viewer.layers.__contains__('wat'): 
                viewer.layers['wat'].data = watsize
                
            else:
                viewer.add_image(
                    watsize, 
                    name='wat',
                    colormap='red',
                    contrast_limits=(0, 1),
                    blending='additive',
                    )

    viewer.window.add_dock_widget(display, area='right', name='widget')
    
#%% Run -----------------------------------------------------------------------

display_wat(raw)
             