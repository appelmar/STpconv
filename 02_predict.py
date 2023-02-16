import os
import sys
import numpy as np
import rasterio as rio
import time
import tensorflow as tf

from skimage import io as skio
from skimage import util as skutil

# Import modules from lib/ directory
from lib.STpconvLayer import STpconv
from lib.STpconvUnet import STpconvUnet
from lib.DataGenerator import DataGenerator

## uncomment for computation time measurements
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)


print("Using TensorFlow version", tf.__version__)

DATA_PATH_IN = "data/validation"
DATA_PATH_OUT = "data/validation/predictions"

if os.path.exists(DATA_PATH_OUT):
    if len(os.listdir(DATA_PATH_OUT)) > 0:
        sys.exit("Output directory exists and is not empty")
else:
    os.makedirs(DATA_PATH_OUT)

model = STpconvUnet.load("models/S5P_CO_model", weights_name = "models/S5P_CO_model.h5")

model = STpconvUnet.load("model_architecture", weights_name = "out/epoch_50.h5")



def predict_single(img_path, mask_path, out_path):
    temp = skio.imread(img_path)
    temp = np.expand_dims(temp, axis=0) # append "sample" dimension
    temp = np.expand_dims(temp, axis=4) # append "channel" dimension
    temp[temp<0] = 0 
    X = temp 
 
    temp = skio.imread(mask_path)
    temp = np.expand_dims(temp, axis=0) # append "sample" dimension
    temp = np.expand_dims(temp, axis=4) # append "channel" dimension
    mask = temp 
    
    pred = model.predict([X,mask, mask]) # 2nd mask is not needed here, so we simply use the same mask
    
    # create output file and copy spatial reference from input
    img = rio.open(img_path)
    crs = None
    affine = None
    if not img.crs is None:
        crs = img.crs.to_string()
    if not img.transform is None:
        affine = img.transform
    new_dataset = rio.open(out_path, 'w',driver='GTiff', height=pred.shape[1],
                           width=pred.shape[2],count=pred.shape[3], dtype=pred.dtype, crs=crs, transform=affine)
    for it in range(model.nt):
        new_dataset.write(pred[0,:,:,it,0], it +1)
    new_dataset.close()
    img.close()
    
    

x_files = sorted(
    [
        os.path.join(DATA_PATH_IN, fname)
        for fname in os.listdir(DATA_PATH_IN)
        if fname.endswith(".tif") and fname.startswith("X_") 
    ]
)
mask_files = sorted(
    [
        os.path.join(DATA_PATH_IN, fname)
        for fname in os.listdir(DATA_PATH_IN)
        if fname.endswith(".tif") and fname.startswith("MASK_") 
    ]
)

assert len(mask_files) == len(x_files)


start = time.process_time()
for i in range(len(x_files)):
    out_path = os.path.join(DATA_PATH_OUT, "PRED_" + "_".join(os.path.basename(x_files[i]).split("_")[1:]))
    predict_single(x_files[i], mask_files[i], out_path)
t = time.process_time() - start   

print(t/len(mask_files))
print("DONE.")
                  
                