#%%

import time
from skimage import io

#%%

from functions import best_ridge_size
from tasks import process

#%% Parameters

''' 1) Open data '''

ROOT_PATH = '../data/'

# RAW_NAME = '13-12-06_40x_GBE_eCad_Ctrl_#19_Lite_uint8.tif'
RAW_NAME = '13-12-06_40x_GBE_eCad_Ctrl_#19_Lite2_uint8.tif'
# RAW_NAME = '13-03-06_40x_GBE_eCad(Carb)_Ctrl_#98_Lite_uint8.tif'
# RAW_NAME = '17-12-18_100x_DC_UtrCH_Ctrl_b3_uint8.tif'
# RAW_NAME = '18-03-12_100x_GBE_UtrCH_Ctrl_b3_uint8.tif'
# RAW_NAME = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif'
# RAW_NAME = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif'
# RAW_NAME = 'Disc_Fixed_118hAEL_disc04_uint8.tif'
# RAW_NAME = 'Disc_ex_vivo_118hAEL_disc2_uint8.tif'

''' 2) General options '''
RSIZE_FACTOR = 0.5 # must be <=1
TIME_WINDOW = 3 # must be >=1 and odd 

''' 3) Preprocess '''
RIDGE_SIZE = 'auto' 
RIDGE_SIZE_COEFF = 0.75

''' 4) Watershed '''
THRESH_COEFF = 0.5 
THRESH_MIN_SIZE = int(3000*RSIZE_FACTOR) 

''' 5) PIV '''
PIV = True
PIV_WIN_SIZE = int(96*RSIZE_FACTOR)

#%% Initialize

# Open data
raw = io.imread(ROOT_PATH + RAW_NAME) 

# Determine best ridge size
if RIDGE_SIZE == 'auto':
    RIDGE_SIZE = best_ridge_size(raw, RSIZE_FACTOR)    
    
# Adjust RIDGE_SIZE with RIDGE_SIZE_COEFF   
RIDGE_SIZE *= RIDGE_SIZE_COEFF 

# ............................................................................. 

if raw.ndim == 2:
    TIME_WINDOW = 1
       
if TIME_WINDOW == 1:
    PIV = False
    
if TIME_WINDOW < 1 or TIME_WINDOW % 2 == 0:
    raise ValueError("time_window must be odd and >= 1")


#%%

outputs, bound_int_display, bound_edm_int_display, bound_edm_sd_display = process(raw,
    RSIZE_FACTOR, TIME_WINDOW, RIDGE_SIZE, THRESH_COEFF, THRESH_MIN_SIZE,
    PIV, PIV_WIN_SIZE)


#%% Save data

start = time.time()
print('Save data')

io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_rsize.tif', outputs["rsize"].astype('float32'), check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_ridges.tif', outputs["ridges"].astype('float32'), check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_mask.tif', outputs["mask"].astype('uint8')*255, check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_markers.tif', outputs["markers"].astype('uint16'), check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_labels.tif', outputs["labels"].astype('uint16'), check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_wat.tif', outputs["wat"].astype('uint8')*255, check_contrast=False)
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_labels.tif', outputs["bound_labels"].astype('uint16'), check_contrast=False)
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_norm.tif', outputs["bound_norm"].astype('float32'), check_contrast=False)
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_edm.tif', outputs["bound_edm"].astype('float32'), check_contrast=False)

io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_int_display.tif', bound_int_display.astype('float32'), check_contrast=False)
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_edm_int_display.tif', bound_edm_int_display.astype('float32'), check_contrast=False)
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_bound_edm_sd_display.tif', bound_edm_sd_display.astype('float32'), check_contrast=False)

if PIV:
    
    io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_u.tif', outputs["u"].astype('float32'), check_contrast=False)
    io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_v.tif', outputs["v"].astype('float32'), check_contrast=False)
    io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_vector_field.tif', outputs["vector_field"].astype('uint8'), check_contrast=False)

end = time.time()
print(f'  {(end - start):5.3f} s')
