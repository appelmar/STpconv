import os
import sys
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from lib.STpconvLayer import STpconv
import lib.Losses as losses


class STpconvUnet(object):

    def __init__(self, n_conv_layers = 4, nx=128, ny=128, nt=16, inference_only=False, net_name='default', 
                 kernel_sizes = None, n_filters = None, learning_rate = 0.0001, n_conv_per_block = 1, strides = None):
        
        # Settings
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.n_conv_layers = n_conv_layers
        self.inference_only = inference_only
        self.net_name = net_name
        self.learning_rate = learning_rate
        self.n_conv_per_block = n_conv_per_block
    
        assert n_conv_layers > 1, "n_conv_layers must be > 1"
        
        if not kernel_sizes is None:
            assert len(kernel_sizes) == n_conv_layers, "len(kernel_sizes) must equal n_conv_layers"
            
        if not n_filters is None:
            assert len(n_filters) == n_conv_layers, "len(n_filters) must equal n_conv_layers"
            
        if not strides is None:
            assert len(strides) == n_conv_layers, "len(strides) must equal n_conv_layers"
            
            
        if strides is None:
            # derive strides
            self.strides = []
            for i in range(self.n_conv_layers):
                s = [2,2,2]
                if self.nx / (2 ** (i+1)) < 1:
                    s[0] = 1
                if self.ny / (2 ** (i+1)) < 1:
                    s[1] = 1
                if self.nt / (2 ** (i+1)) < 1:
                    s[2] = 1
                self.strides.append(s)
        else:
            self.strides = strides
            
        # set kernel sizes if not provided    
        if not kernel_sizes is None:
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = []
            for i in range(self.n_conv_layers):
                s = (3,3,3)
               # if i == 0:
               #     s = [3,3,3]
                self.kernel_sizes.append(s)
        
        # set n_filters if not provided
        if not n_filters is None:
            self.n_filters = n_filters
        else:
            self.n_filters = []
            for i in range(self.n_conv_layers):
                s = 2 ** (i+5) # start with 32 and go up to 256
                s = min(s, 256)
                self.n_filters.append(s)
            
        
        self.model, inputs_mask, validation_mask = self.build_pconv_unet()
        self.compile_pconv_unet(self.model, inputs_mask, validation_mask)            
        
   
    def build_pconv_unet(self):      

        # INPUTS
        inputs_img = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='inputs_img')
        inputs_mask = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='inputs_mask')
        input_validation_mask = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='validation_mask')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, strides):
            if (self.n_conv_per_block > 1):
                conv, mask = STpconv(filters, kernel_size, strides=(1,1,1), activation = tf.keras.layers.LeakyReLU(alpha=0.1))([img_in, mask_in])
                if (self.n_conv_per_block > 2):
                    for i in range(self.n_conv_per_block - 2):
                        conv, mask = STpconv(filters, kernel_size, strides=(1,1,1),activation = tf.keras.layers.LeakyReLU(alpha=0.1))([conv, mask])
                conv, mask = STpconv(filters, kernel_size, strides=strides, activation = tf.keras.layers.LeakyReLU(alpha=0.1))([conv, mask])         
            else:
                conv, mask = STpconv(filters, kernel_size, strides=strides, activation = tf.keras.layers.LeakyReLU(alpha=0.1))([img_in, mask_in])
           
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        prev_img = inputs_img
        prev_mask = inputs_mask
        e_conv = []
        e_mask = []
        for i in range(self.n_conv_layers):
            prev_img, prev_mask = encoder_layer(prev_img, prev_mask, self.n_filters[i], self.kernel_sizes[i], self.strides[i])
            e_conv.append(prev_img)
            e_mask.append(prev_mask)
   
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, size):
            up_img = tf.keras.layers.UpSampling3D(size=size)(img_in)
            up_mask = tf.keras.layers.UpSampling3D(size=size)(mask_in)
            concat_img = tf.keras.layers.Concatenate(axis=4)([e_conv,up_img])
            concat_mask = tf.keras.layers.Concatenate(axis=4)([e_mask,up_mask])
            
            if (self.n_conv_per_block > 1):
                conv, mask = STpconv(filters, kernel_size)([concat_img, concat_mask])
                for i in range(self.n_conv_per_block - 1):
                    conv, mask = STpconv(filters, kernel_size, activation = tf.keras.layers.LeakyReLU(alpha=0.1))([conv, mask])   
            else:
                conv, mask = STpconv(filters, kernel_size, activation = tf.keras.layers.LeakyReLU(alpha=0.1))([concat_img, concat_mask])
                
            return conv, mask
            
            
        d_conv = []
        d_mask = []
        prev_img = e_conv[self.n_conv_layers - 1]
        prev_mask =  e_mask[self.n_conv_layers - 1]
        for i in reversed(range(self.n_conv_layers)):       
            if i == 0:
                prev_img, prev_mask = decoder_layer(prev_img, prev_mask, inputs_img, 
                                                    inputs_mask, 1, (3,3,3), self.strides[i]) # decoder layers use 3x3x3 kernels only
            else:
                prev_img, prev_mask = decoder_layer(prev_img, prev_mask, e_conv[i-1], 
                                                     e_mask[i-1], self.n_filters[i-1], (3,3,3), self.strides[i]) # decoder layers use 3x3x3 kernels only
             
            d_conv.append(prev_img)
            d_mask.append(prev_mask)
             
           
        #outputs = STpconv(1, kernel_size=(1,1,1), strides=(1,1,1), activation = "linear")([d_conv[len(d_conv)-1],  d_mask[len(d_conv)-1]]) 
        outputs = tf.keras.layers.Conv3D(1, 1, activation = "linear")(d_conv[len(d_conv)-1]) 
        
        
        
        # Setup the model inputs / outputs
        model = tf.keras.Model(inputs=[inputs_img, inputs_mask, input_validation_mask], outputs=outputs)

        return model, inputs_mask, input_validation_mask   

    def compile_pconv_unet(self, model, inputs_mask, validation_mask):
        model.compile(
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = self.learning_rate),
            loss=self.loss_total(inputs_mask, validation_mask),
            metrics=[losses.mae_gaps(inputs_mask, validation_mask), losses.rmse_gaps(inputs_mask, validation_mask),  losses.r2_gaps(inputs_mask, validation_mask)]
        )

    def loss_total(self, mask, validation_mask):
        """
        Creates a loss function
        """
        def loss(y_true, y_pred):
            return losses.mae_gaps(mask, validation_mask)(y_true, y_pred) 
        loss.__name__ = 'loss_total'
        
        return loss
        
    def save(self, name, save_weights = True):
        if save_weights:
            self.model.save_weights(name)
        data = {}
        data['nx'] =  self.nx
        data['ny'] =  self.ny
        data['nt'] =  self.nt
        data['n_conv_layers'] =  self.n_conv_layers
        data['inference_only'] =  self.inference_only
        data['net_name'] =  self.net_name
        data['kernel_sizes'] = self.kernel_sizes
        data['n_filters'] = self.n_filters
        data['learning_rate'] = self.learning_rate
        data['n_conv_per_block'] = self.n_conv_per_block
        data['strides'] = self.strides
        
        
        with open(name + ".json", 'w') as fp:
            json.dump(data, fp)
        
        
    @staticmethod
    def load(name, weights_name = "", inference_only=True):
        with open(name + ".json", 'r') as fp:
            data = json.load(fp)
        if not 'n_conv_per_block' in data: 
            n_conv_per_block = 1
        else:
            n_conv_per_block =  data['n_conv_per_block']
            
        if not 'strides' in data: 
            strides = None
        else:
            strides =  [tuple(x) for x in data["strides"]]     
        
        if not 'kernel_sizes' in data: 
            kernel_sizes = None
        else:
            kernel_sizes =  [tuple(x) for x in data["kernel_sizes"]]   
      
        m = STpconvUnet(n_conv_layers = data["n_conv_layers"], nx=data["nx"], ny=data["ny"], 
                        nt=data["nt"], inference_only=inference_only, net_name=data["net_name"],
                        kernel_sizes = kernel_sizes, n_filters = data["n_filters"], 
                        learning_rate = data["learning_rate"],n_conv_per_block = n_conv_per_block, strides = strides)
        
        if not weights_name:
            load_status = m.model.load_weights(name)     
        else:
            load_status = m.model.load_weights(weights_name)     
              
        return m;
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())


    
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
