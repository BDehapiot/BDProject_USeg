#%% Imports
import napari
import numpy as np

from skimage import io
from pathlib import Path

from magicgui import magicgui

from widget_functions import preprocess, watseg

#%% Open data

# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_Lite_uint8.tif"
# raw_name  = "13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_uint8.tif"
# raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif"
# raw_name  = "13-12-06_40x_GBE_eCad_Ctrl_#19_Lite_uint8.tif"
# raw_name  = "17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif"
# raw_name  = "18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif"
# raw_name  = "Disc_ex_vivo_118hAEL_disc2_uint8.tif"
# raw_name  = "Disc_Fixed_118hAEL_disc04_uint8.tif"

# Create paths
# root_path = Path(__file__).parents[1]
# raw_path = Path(root_path / 'data' / raw_name)

#%% Initialize

def useg():
    
    class Viewer(napari.Viewer):
        pass
    
    viewer = Viewer()
    
    viewer.my_attr = 10
    
    # viewer.newatrr = 10
    
#%% Open data
    
    # @magicgui(
        
    #     auto_call = True,
                       
    #     raw_path = {
    #         'widget_type': 'FileEdit', 
    #         'label': 'Select raw image or stack',
    #         'value': Path(__file__).parents[1] / 'data'
    #         },
       
    #     )
    
    # def open_raw(
    #         raw_path: str,
    #         ):
        
    #     # Open raw
    #     raw = io.imread(raw_path) 
        
    #     # Get variables
    #     viewer.shape = raw.shape
        
    #     # Add one dimension (if ndim = 2)
    #     ndim = (raw.ndim)        
    #     if ndim == 2:
    #         raw = raw.reshape((1, raw.shape[0], raw.shape[1]))
        
    #     # Display raw in viewer
    #     viewer.add_image(
    #         raw[raw.shape[0]//2,...], 
    #         name='raw',
    #         colormap='gray',
    #         contrast_limits=(
    #             np.quantile(raw, 0.001),
    #             np.quantile(raw, 0.999),
    #             ),
    #         )
    
#%%

    # @magicgui(
        
    #     auto_call = True,
                       
    #     frame = {
    #         'widget_type': 'SpinBox', 
    #         'label': 'frame',
    #         'min': 0, 'max': raw.shape[0]-1,
    #         'value': raw.shape[0]//2,
    #         },
        
    #     rsize_factor = {
    #         'widget_type': 'FloatSpinBox', 
    #         'label': 'resizing factor',
    #         'min': 0, 'max': 1, 'step': 0.1,
    #         'value': 0.5,
    #         },
        
    #     ridge_size = {
    #         'widget_type': 'FloatSpinBox', 
    #         'label': 'ridge size (pixels)',
    #         'min': 0, 'max': 20, 'step': 0.5,
    #         'value': 3,
    #         },
        
    #     thresh_coeff = {
    #         'widget_type': 'FloatSpinBox', 
    #         'label': 'thresh. coeff.',
    #         'min': 0, 'step': 0.1,
    #         'value': 1.0,
    #         },
        
    #     preview = {
    #         'widget_type': 'CheckBox',
    #         'label': 'preview',
    #         'value': False, 
    #         },
       
    #     )
    
    # def get_wat(
    #         raw_path: str,
    #         ):
        
    #     # Open raw
    #     raw = io.imread(raw_path) 
        
    #     # Add one dimension (if ndim = 2)
    #     ndim = (raw.ndim)        
    #     if ndim == 2:
    #         raw = raw.reshape((1, raw.shape[0], raw.shape[1]))
        
    #     # Display raw in viewer
    #     viewer.add_image(
    #         raw[raw.shape[0]//2,...], 
    #         name='raw',
    #         colormap='gray',
    #         contrast_limits=(
    #             np.quantile(raw, 0.001),
    #             np.quantile(raw, 0.999),
    #             ),
    #         )
        
    #     return 
                 
    
#%%

    # viewer.window.add_dock_widget(open_raw, area='right', name='widget') 
    
#%% Callbacks

#%% Keybindings
    
#%% 

useg()
