import os
import shutil
import numpy as np
import tensorflow as tf

# Import modules from lib directory
from lib.STpconvLayer import STpconv
from lib.STpconvUnet import STpconvUnet
from lib.DataGenerator import DataGenerator

print("Using TensorFlow version", tf.__version__)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Hyperparameters and other options
BATCH_SIZE = 2 
N_EPOCHS = 50 
OUT_PATH = os.getcwd()
DATA_PATH_TRAINING = "data/training"
DATA_PATH_VALIDATION = "data/validation"

N_CONV_LAYERS = 2
N_CONV_PER_BLOCK = 1
KERNEL_SIZES = [(3, 3, 3), (3, 3, 3)]
N_FILTERS    = [16, 16]
STRIDES = [(2, 2, 2),(2, 2, 2)]


trn_generator = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE)
val_generator = DataGenerator(DATA_PATH_VALIDATION, batch_size = BATCH_SIZE)

model = STpconvUnet(n_conv_layers = N_CONV_LAYERS, nx=128, ny=128, nt=16, kernel_sizes = KERNEL_SIZES, n_filters = N_FILTERS, learning_rate=0.01, n_conv_per_block=N_CONV_PER_BLOCK, strides = STRIDES)
model.summary()

# save model at the end of each epoch
checkpoint_dir = os.path.join(OUT_PATH, "out")
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
os.mkdir(checkpoint_dir)
checkpoint_filepath = os.path.join(checkpoint_dir, "epoch_{epoch:02d}.h5")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    save_freq = "epoch",
    save_best_only=False)

# save hyperparameters in a JSON file
model.save(os.path.join(OUT_PATH,"model_architecture"), save_weights = False)

# train model
history = model.model.fit(x=trn_generator, validation_data = val_generator, 
                          epochs = N_EPOCHS, callbacks=[model_checkpoint_callback])
