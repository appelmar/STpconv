import tensorflow as tf

class STpconv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = (3,3,3), strides = (1,1,1), use_bias = True, activation = None):
      super(STpconv, self).__init__() # add name and other args?
      
      self.filters = filters
      self.kernel_size = kernel_size
      self.strides = strides
      self.use_bias = use_bias
      
      self.in_activation = activation
      if activation is None:
        self.activation = None
      elif isinstance(activation, str):
        self.activation = tf.keras.activations.deserialize(activation)
      elif callable(activation):
        self.activation = activation
      elif isinstance(activation, dict):
        self.activation = tf.keras.layers.deserialize(activation)
      else:
        raise ValueError("Failed to interpret activation arg, expected None, string or instance of tf.keras.layers.Layer.")
        
      
        
    def get_config(self):
     
      if isinstance(self.activation, tf.keras.layers.Layer):
        act = self.activation.get_config()
      elif callable(self.activation):
        act = tf.keras.activations.serialize(self.activation)
        
      return {"filters": self.filters, 
              "kernel_size": self.kernel_size,
              "strides": self.strides,
              "use_bias": self.use_bias,
              "activation" : act} # TODO: test

    def build(self, input_shape):       
      channel_axis = -1 # assume channel_last data format
      if len(input_shape) != 2:
        raise ValueError("Invalid input shape, expected length-two list containing image and mask.")
      if input_shape[0][channel_axis] is None:
        raise ValueError("The channel dimension of image inputs should be defined. Found `None`")
      if input_shape[1][channel_axis] is None:
        raise ValueError("The channel dimension of mask inputs should be defined. Found `None`.")
      # if input_shape[0].shape != input_shape[1].shape:
      #   raise ValueError("Invalid input shape, expected identical shapes of image and mask inputs.")
      #     
      self.input_dim = input_shape[0][channel_axis]
      
      # Image kernel
      kernel_shape = self.kernel_size + (self.input_dim, self.filters)
      self.kernel = self.add_weight(shape=kernel_shape,
                                    initializer=tf.keras.initializers.GlorotUniform(),
                                    name='STpconv_kernel')
      # Mask kernel
      self.kernel_mask = tf.ones(kernel_shape)


      # Calculate padding sizes per dimension
      p0 = int((self.kernel_size[0]-1)/2)
      p1 = int((self.kernel_size[1]-1)/2)
      p2 = int((self.kernel_size[2]-1)/2)
      self.paddings = [[0, 0], [p0, p0],  [p1, p1],  [p2, p2],  [0, 0]] 
        
      
      if self.use_bias:
        self.bias = self.add_weight(shape=(self.filters,), initializer=tf.keras.initializers.zeros(), name='bias')
      else:
        self.bias = None
      self.built = True

    def call(self, inputs):
      # Both image and mask must be supplied
      if type(inputs) is not list or len(inputs) != 2:
        raise Exception('STpconv must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

      # Padding (=zero padding) done explicitly so that padding becomes part of the masked partial convolution
      imgs  = tf.pad(inputs[0], self.paddings, "CONSTANT", constant_values=0) # zero padding
      masks = tf.pad(inputs[1], self.paddings, "CONSTANT", constant_values=0) # zero padding
      
      # images = K.spatial_3d_padding(inputs[0], self.pconv_padding, self.data_format)
      # masks = K.spatial_3d_padding(inputs[1], self.pconv_padding, self.data_format)
      mask_output = tf.nn.conv3d(masks, self.kernel_mask, strides = (1,) + self.strides + (1,), padding = "VALID") # TODO: data format?
      img_output = tf.nn.conv3d(masks * imgs, self.kernel, strides = (1,) + self.strides + (1,), padding = "VALID") # TODO: data format?

     
      # Calculate the mask ratio on each pixel in the output mask
      n = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
      mask_ratio = n / (mask_output + 1e-8) # avoid devision by zero

      # Clip output to be between 0 and 1
      mask_output = tf.clip_by_value(mask_output, 0, 1)

      # Remove ratio values where there are holes
      mask_ratio = mask_ratio * mask_output

      # Normalize iamge output
      img_output = img_output * mask_ratio

      # Apply bias only to the image (if chosen to do so)
      if self.use_bias:
        img_output = tf.nn.bias_add(img_output, self.bias) # data format?
        
      # Cal activation function
      if not self.activation is None:
        img_output = self.activation(img_output)
          
      return [img_output, mask_output]
    
