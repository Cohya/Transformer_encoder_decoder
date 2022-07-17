import tensorflow as tf 
import numpy as np 
import torch 
import torch.nn.functional as F

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
    
def create_look_ahead_mask(size): ## <- this is the best one 
   # size = t
   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # 1 - 
   return mask  # (seq_len, seq_len)

def create_look_Only_back_mask(size): # the token can see only the previuse one, tokens are the "t"
   # size = t 
   mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0) -1  #- 1 # - tf.eye(size)
   return tf.math.abs(mask)  # (seq_len, seq_len)

# t = 3
# # dot = torch.randn((1,t,t))
# weights = tf.random.normal(shape = (1,t,t))


# # mask_(dot)
# m1 = create_look_ahead_mask(t)
# m = create_look_Only_back_mask(t)#create_look_ahead_mask(t)

# w1  = weights  +  (m1 * -1e9)
# w = weights  + (m *-1e9)

# w1 = tf.nn.softmax(w1, axis = -1)
# w = tf.nn.softmax(w, axis = -1)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def get_angle(pos, i, d_model, n = 100):
    # position of the vec, index, d_model==k (dims of encoding)
    # n = 10000 in the paper "Attention is all you need"
    angle_rate = 1/np.power(n,(2*(i//2)) / np.float32(d_model))
    return pos * angle_rate

def positional_endcoding(positions, d_model):
    # x.shape = (number of samples, t, k)
    angle_rads = get_angle(np.arange(positions)[:,np.newaxis],
                           np.arange(d_model)[np.newaxis ,:],
                           d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    out = tf.cast(pos_encoding, dtype=tf.float32)
    return out


def getSeriesData(n = 1000):
    n_part = 100 #int(n*0.1) #10% for validation 
    seriesGeneral = np.sin((0.1 * np.arange(n)) **2 ) # x(t) = sin(w*t**2)
    validation_series = seriesGeneral[-n_part:]
    series = seriesGeneral[:-n_part]
    return series, validation_series

def prepare_data_for_encoder_decoder_transformer(series, validation_series, T = 5 ):
    # T is the dimention of the look back steps 
    T  = 5
    # D = 1
    
    # X_transformer = [x_encoder, x_decoder]
    X_encoder = []
    X_decoder = []
    Y = []
    for t in range(len(series) - T-1):
        x_encoder = series[t:t+T]
        x_decoder = series[(t+1):t+T+1] #series[(t+T-1):t+T+1]
        y = series[(t+2):(t+T+2)]
        X_encoder.append(np.reshape(x_encoder, newshape=(T,1)))
        X_decoder.append(np.reshape(x_decoder, newshape = (T,1))) # was 2
        Y.append(np.reshape(y, newshape = (T, 1)))
    
    N = len(X_decoder) # the numebr of samples
    n = int(N *0.8)
    

    X_encoder = np.array(X_encoder, dtype = np.float32)
    X_decoder = np.array(X_decoder, dtype = np.float32)
    Y  = np.array(Y, dtype = np.float32)
    
    X_decoder_train = X_decoder[:n,...]
    X_encoder_train = X_encoder[:n,...]
    Y_train = Y[:n,...]
    
    X_general_train = [X_encoder_train, X_decoder_train]
    
    X_test_general = [X_encoder[n:,...], X_decoder[n:,...]]
    Y_test = Y[n:,...]
    
    X_encoder_validation = []
    X_decoder_validation = []
    Y_Validation = []
    
    for t in range(len(validation_series) - T - 1):
        x_encoder = validation_series[t:t+T]
        x_decoder = validation_series[(t+1):t+T+1] #series[(t+T-1):t+T+1]
        y = validation_series[(t+2):(t+T+2)]
        X_encoder_validation.append(np.reshape(x_encoder, newshape=(T,1)))
        X_decoder_validation.append(np.reshape(x_decoder, newshape = (T,1))) # was 2
        Y_Validation.append(np.reshape(y, newshape = (T, 1)))
    
    X_encoder_validation= np.array(X_encoder_validation, dtype = np.float32)
    X_decoder_validation  = np.array(X_decoder_validation , dtype = np.float32)
    Y_Validation  = np.array(Y_Validation, dtype = np.float32)
    
    X_general_validation = [X_encoder_validation,X_decoder_validation]
    
    return X_general_train, Y_train, X_test_general, Y_test, X_general_validation, Y_Validation

# position = 4
# d_model =4

# d = positional_endcoding(position, d_model)

 