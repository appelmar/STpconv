# This file implements tf.keras.utils.Sequence to read
# spatiotemporal input blocks batch-wise when needed. This 
# is used while training models in order to reduce memory consumption.
#
# Please notice that negative values of pixels are interpreted as nodata values and will be 
# internally set to 0. For datasets with correct negative values, the loading procedure below
# must be adapted using a different nodata value. 
#

import numpy as np
import tensorflow as tf
import os
import random
from skimage import io as skio

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_dir, batch_size=32, dim=(128,128,16), n_channels=1, shuffle=True, nmax = -1, fill = 0):
        self.dim = dim
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.fill = fill
        
        self.x_files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif") and fname.startswith("X_") 
            ]
        )
        self.y_files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif") and fname.startswith("Y_") 
            ]
        )
        self.mask_files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif") and fname.startswith("MASK_") 
            ]
        )
        self.val_mask_files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif") and fname.startswith("VALMASK_") 
            ]
        )
        
        assert len(self.mask_files) == len(self.y_files) and len(self.mask_files) == len(self.x_files) and len(self.val_mask_files) == len(self.x_files)
        if nmax > 0 and nmax < len(self.mask_files):
            order = random.sample(range(len(self.mask_files)),nmax)
            self.mask_files  = [self.mask_files [i] for i in order]
            self.y_files   = [self.y_files [i] for i in order]
            self.x_files   = [self.x_files [i] for i in order]
            self.val_mask_files  = [self.val_mask_files [i] for i in order]
        
        
        self.n = len(self.mask_files)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        X = np.empty((0, self.dim[0], self.dim[1], self.dim[2], 1))
        y = np.empty((0, self.dim[0], self.dim[1], self.dim[2], 1))
        mask = np.empty((0, self.dim[0], self.dim[1], self.dim[2], 1))
        val_mask = np.empty((0, self.dim[0], self.dim[1], self.dim[2], 1))
       
        for i in range(self.batch_size):
            ii = index*self.batch_size + i
            
            temp = skio.imread(self.x_files[ii])
            temp = np.expand_dims(temp, axis=0) # append "sample" dimension
            temp = np.expand_dims(temp, axis=4) # append "channel" dimension
            temp[temp<0] = self.fill # set nodata values to 0 because masked pixels otherwise do contribute to the kernel sums
            X = np.concatenate((X, temp), axis = 0)
                
            temp = skio.imread(self.y_files[ii])
            temp = np.expand_dims(temp, axis=0) # append "sample" dimension
            temp = np.expand_dims(temp, axis=4) # append "channel" dimension
            temp[temp<0] = self.fill # set nodata values to 0 because masked pixels otherwise do contribute to the kernel sums
            y = np.concatenate((y, temp), axis = 0)
               
            temp = skio.imread(self.mask_files[ii])
            temp = np.expand_dims(temp, axis=0) # append "sample" dimension
            temp = np.expand_dims(temp, axis=4) # append "channel" dimension
            mask = np.concatenate((mask, temp), axis = 0)
            
            temp = skio.imread(self.val_mask_files[ii])
            temp = np.expand_dims(temp, axis=0) # append "sample" dimension
            temp = np.expand_dims(temp, axis=4) # append "channel" dimension
            val_mask = np.concatenate((val_mask, temp), axis = 0)
                  
            assert np.min(mask) >= 0 and  np.max(mask) <= 1
            assert np.min(val_mask) >= 0 and  np.max(val_mask) <= 1
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(y))
            
        return [X, mask, val_mask], y 

    def on_epoch_end(self):
        if self.shuffle == True:
            order = random.sample(range(self.n),self.n)
            self.x_files  = [self.x_files [i] for i in order]
            self.y_files  = [self.y_files [i] for i in order]
            self.mask_files  = [self.mask_files [i] for i in order]
            self.val_mask_files  = [self.val_mask_files [i] for i in order]
