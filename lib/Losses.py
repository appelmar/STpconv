from tensorflow.keras import backend as K


def mae_all(mask, validation_mask):
    """
    Computes MAE for all values
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(mask) + K.flatten(validation_mask)
        y_true = y_true[K.greater(m, 0)]
        y_pred = y_pred[K.greater(m, 0)]
        return K.mean(K.abs(y_true-y_pred))
        #return K.sqrt(K.mean(K.square(y_true-y_pred)))
    loss.__name__ = 'mae_all'
    
    return loss


def mae_gaps(mask, validation_mask):
    """
    Computes MAE for gap values only
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        # TODO
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(validation_mask)
        y_true = y_true[K.equal(m, 1)]
        y_pred = y_pred[K.equal(m, 1)]
        return K.mean(K.abs(y_true-y_pred))
        #return K.sqrt(K.mean(K.square(y_true-y_pred)))
    loss.__name__ = 'mae_gaps'
    
    return loss




def rmse_all(mask, validation_mask):
    """
    Computes RMSE for all values
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(mask) + K.flatten(validation_mask)
        y_true = y_true[K.greater(m, 0)]
        y_pred = y_pred[K.greater(m, 0)]
        return K.sqrt(K.mean(K.square(y_true-y_pred)))
    loss.__name__ = 'rmse_all'
    
    return loss

def rmse_gaps(mask, validation_mask):
    """
    Computes RMSE for masked values only
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        # TODO
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(validation_mask)
        y_true = y_true[K.equal(m, 1)]
        y_pred = y_pred[K.equal(m, 1)]
        #return K.mean(K.abs(y_true-y_pred))
        return K.sqrt(K.mean(K.square(y_true-y_pred)))
    loss.__name__ = 'rmse_gaps'
    
    return loss

def mse_all(mask, validation_mask):
    """
    Computes MSE for all values
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(mask) + K.flatten(validation_mask)
        y_true = y_true[K.greater(m, 0)]
        y_pred = y_pred[K.greater(m, 0)]
        return K.mean(K.square(y_true-y_pred))
    loss.__name__ = 'mse_all'
    
    return loss


def mse_gaps(mask, validation_mask):
    """
    Computes MSE for masked values only
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        # TODO
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(validation_mask)
        y_true = y_true[K.equal(m, 1)]
        y_pred = y_pred[K.equal(m, 1)]
        return K.mean(K.square(y_true-y_pred))
    loss.__name__ = 'mse_gaps'
    
    return loss



def r2_gaps(mask, validation_mask):
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        m = K.flatten(validation_mask)
        y_true = y_true[K.equal(m, 1)]
        y_pred = y_pred[K.equal(m, 1)]
        return 1-(K.sum(K.square(y_true - y_pred)))/(K.sum(K.square(y_true-K.mean(y_true))))
    loss.__name__ = 'r2_gaps'
    
    return loss


def r2_all(mask, validation_mask):
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        m = K.flatten(mask) + K.flatten(validation_mask)
        y_true = y_true[K.greater(m, 0)]
        y_pred = y_pred[K.greater(m, 0)]

        return 1-(K.sum(K.square(y_true - y_pred)))/(K.sum(K.square(y_true-K.mean(y_true))))
    loss.__name__ = 'r2_all'
    
    return loss


def tv_loss(mask, validation_mask):
    """
    Computes total variation loss for masked values only
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        # TODO
        y_comp = mask * y_true + (1-mask) * y_pred
        
        # Create dilated hole region using a 3x3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, 3, mask.shape[4], mask.shape[4]))
        dilated_mask = K.conv3d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:,:], P[:,:-1,:,:,:])
        b = self.l1(P[:,:,1:,:,:], P[:,:,:-1,:,:])   
        c = self.l1(P[:,:,:,1:,:], P[:,:,:,:-1,:]) 
        return a+b+c
    
    loss.__name__ = 'tv_loss'
    
    return loss
