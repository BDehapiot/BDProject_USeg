#%% Imports
import napari
import numpy as np

from skimage import io
from pathlib import Path

from magicgui import magicgui

from functions import preprocess, watseg

#%% Parameters

# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_Lite_uint8.tif"
# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif"
raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif"
# raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite_uint8.tif"
# raw_name  = "17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif"
# raw_name  = "Disc_ex_vivo_118hAEL_disc2_uint8.tif"
# raw_name  = "Disc_Fixed_118hAEL_disc04_uint8.tif"

#%% Open data

# Create paths
root_path = Path(__file__).parents[1]
raw_path = Path(root_path / 'data' / raw_name)

# Open data
raw = io.imread(raw_path) 

# Add one dimension (if ndim = 2)
ndim = (raw.ndim)        
if ndim == 2:
    raw = raw.reshape((1, raw.shape[0], raw.shape[1]))

#%% Initialize

def get_wat(raw):
    
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
                 
#%%

    @magicgui(
        
        auto_call = True,
                       
        frame = {
            'widget_type': 'SpinBox', 
            'label': 'frame',
            'min': 0, 'max': raw.shape[0]-1,
            'value': raw.shape[0]//2,
            },
        
        rsize_factor = {
            'widget_type': 'SpinBox', 
            'label': 'resizing factor',
            'min': 1, 'max': 5, 'step': 1,
            'value': 1,
            },
        
        ridge_size = {
            'widget_type': 'FloatSpinBox', 
            'label': 'ridge size (pixels)',
            'min': 0, 'max': 10, 'step': 0.5,
            'value': 3,
            },
        
        thresh_coeff = {
            'widget_type': 'FloatSpinBox', 
            'label': 'thresh. coeff.',
            'min': 0, 'step': 0.1,
            'value': 1.0,
            },
        
        thresh_min_size = {
            'widget_type': 'SpinBox', 
            'label': 'min. object size',
            'min': 0, 'step': 1,
            'value': 1000,
            },
        
        preview = {
            'widget_type': 'CheckBox',
            'label': 'preview',
            'value': False, 
            },
        
        )
    
    def display(
            frame: int,
            rsize_factor:  float,
            ridge_size : float,
            thresh_coeff:  float,
            thresh_min_size: float,            
            preview: bool,
            ):
        
        # Get info
        i = display.frame.value         
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f'frame = {i}'
                
        # Show raw image ------------------------------------------------------
        
        if not preview:
            
            if viewer.layers.__contains__('raw'):               
                viewer.layers['raw'].data = raw[i,...]
                
            else:
                
                if viewer.layers.__contains__('rsize'): 
                    viewer.layers.remove('rsize') 
                if viewer.layers.__contains__('wat'): 
                    viewer.layers.remove('wat') 
                
                viewer.add_image(
                    raw[i,...], 
                    name='raw',
                    colormap='gray',
                    contrast_limits=(
                        np.quantile(raw, 0.001),
                        np.quantile(raw, 0.999),
                        ),
                    )
                                
                viewer.reset_view()

        else:
            
            # Get wat preview -------------------------------------------------
            
            rsize, ridges = preprocess(
                raw[i,...], 
                display.rsize_factor.value,
                display.ridge_size.value,
                parallel=False,
                )
            
            mask, markers, labels, wat = watseg(
                ridges, 
                thresh_coeff, 
                thresh_min_size*rsize_factor, 
                parallel=False,
                )
            
            # Update viewer ---------------------------------------------------
                                    
            if viewer.layers.__contains__('rsize'): 
                viewer.layers['rsize'].data = rsize 
                
            else:
                
                viewer.add_image(
                    rsize, 
                    name='rsize',
                    colormap='gray',
                    contrast_limits=(
                        np.quantile(rsize, 0.001),
                        np.quantile(rsize, 0.999),
                        ),
                    )
            
            if viewer.layers.__contains__('wat'): 
                viewer.layers['wat'].data = wat
                
            else:

                viewer.add_image(
                    wat, 
                    name='wat',
                    colormap='red',
                    contrast_limits=(0, 1),
                    blending='additive',
                    )
                
            if viewer.layers.__contains__('raw'):               
                viewer.layers.remove('raw')
                viewer.reset_view()
                
#%%

    viewer.window.add_dock_widget(display, area='right', name='widget') 
    
#%%

    @display.rsize_factor.changed.connect
    def rsize_factor_zoom_callback():
        print(viewer.camera.center)
        viewer.reset_view()
        
#%%

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
        
        # napari.window.Window(viewer, show=True)
        # viewer.layers.selection.active = viewer.layers['wat']

        if viewer.layers.__contains__('wat'): 
            
            viewer.layers['wat'].visible = False
            
            yield
            
            viewer.layers['wat'].visible = True
    

    
#%%

get_wat(raw)