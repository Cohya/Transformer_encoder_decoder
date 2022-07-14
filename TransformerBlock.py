import tensorflow as tf 
from TrasnformerBuildingBlocks import  (SelfMultiHeadAttention, 
                                        LayerNormalization,ANN, 
                                        SelfMultiHeadAttentionKeras, AnnKears)


class TransformerDecoderBlock(object):
    def __init__(self, k, heads, hiddenInAnn = 4, dropoutRate = 0.0, mask_fn = None):
        
        self.k = k
        self.heads = heads 
        self.hiddenInAnn = hiddenInAnn
        self.dropoutRate = dropoutRate
        self.mask_fn = mask_fn
        
        self.attention1 = SelfMultiHeadAttention( k = k, 
                                                heads=heads, 
                                                mask_fn = mask_fn)
        # mostly the mask is look_ahead_mask 
        self.attention2 = SelfMultiHeadAttention( k = k, 
                                                heads = heads,
                                                mask_fn = None) # here it can be padding mask 
        
        
        self.normLayer1 = LayerNormalization(k = k)
        self.normLayer2 = LayerNormalization(k = k)
        self.normLayer3 = LayerNormalization(k = k)
        
        M2 = k * hiddenInAnn
        self.ff = ANN(k, K= k, hidden_layer_sizes=[[M2, tf.nn.relu]])
        
        ## collect trainable params 
        self.trainable_params = []
        
        self.trainable_params += self.attention1.trainable_params
        self.trainable_params += self.attention2.trainable_params
        self.trainable_params += self.normLayer1.trainable_params
        self.trainable_params += self.normLayer2.trainable_params
        self.trainable_params += self.normLayer3.trainable_params
        self.trainable_params += self.ff.trainable_params
        
    def forward(self,x, x_decoder, is_training = False):
        
        attended = self.attention1.forward(va = x,ke = x,qu = x )
        # print("attended:", attended[0])
        if is_training:
            attended  = tf.nn.dropout(attended , rate=self.dropoutRate)
            
        norm1 = self.normLayer1.forward(attended + x)
        # print(norm1[0])
        attended2 = self.attention2.forward(va = x_decoder,ke = x_decoder,qu = norm1)
        
        if is_training:
            attended2 = tf.nn.dropout(attended2, rate = self.dropoutRate)
        
        norm2 = self.normLayer2.forward(attended2 + norm1)
        
        feedforward = self.ff.forward(norm2)
        
        if is_training:
            feedforward = tf.nn.dropout(feedforward, rate = self.dropoutRate)
        
        norm3 = self.normLayer3.forward(feedforward + norm2)
        
        return norm3
        
        
class TransformerDecoderBlockKeras(object):
    def __init__(self, k, heads, hiddenInAnn = 4, dropoutRate = 0.0, mask_fn = None):
        
        self.k = k
        self.heads = heads 
        self.hiddenInAnn = hiddenInAnn
        self.dropoutRate = dropoutRate
        self.mask_fn = mask_fn
        
        self.attention1 = SelfMultiHeadAttentionKeras( k = k, 
                                                heads=heads,
                                                mask_fn = mask_fn)
        # mostly the mask is look_ahead_mask 
        self.attention2 = SelfMultiHeadAttentionKeras( k = k, 
                                                      heads = heads,
                                                mask_fn = None) # here it can be padding mask 
        
        
        self.normLayer1 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        self.normLayer2 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        self.normLayer3 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        
        M2 = k * hiddenInAnn
        self.ff = AnnKears(dims=k, K = k ,hidden_layer_sizes=[[M2,tf.nn.relu]])

        
        x = tf.random.normal(shape = (2,4,k))
        self.forward(x,x, is_training=True)
        
        ## collect trainable params 
        self.trainable_params = []
        
        self.trainable_params += self.attention1.trainable_params
        self.trainable_params += self.attention2.trainable_params
        self.trainable_params += self.normLayer1.trainable_variables
        self.trainable_params += self.normLayer2.trainable_variables
        self.trainable_params += self.normLayer3.trainable_variables
        self.trainable_params += self.ff.trainable_params
        
    def forward(self,x, x_decoder, is_training = False):
        
        attended = self.attention1.forward(va = x,ke = x,qu =x)
        # print("attended:", attended[0])
        if is_training:
            attended  = tf.nn.dropout(attended , rate=self.dropoutRate)
        

        norm1 = self.normLayer1.apply(attended + x)
        # print("Norm:", norm1[0])
        attended2 = self.attention2.forward(va = x_decoder,ke = x_decoder,qu = norm1)
        
        if is_training:
            attended2 = tf.nn.dropout(attended2, rate = self.dropoutRate)
        
        norm2 = self.normLayer2.apply(attended2 + norm1)
        
        feedforward = self.ff.forward(norm2)
        
        if is_training:
            feedforward = tf.nn.dropout(feedforward, rate = self.dropoutRate)
        
        norm3 = self.normLayer3.apply(feedforward + norm2)
        
        return norm3     
        
        
class TransformerEncoderBlock():
    def __init__(self, k, heads, hiddenInAnn = 4, dropoutRate = 0.0, mask_fn = None):
        # k == features dimentions
        # heads = the number of heads in the multihead self attention
        self.dropoutRate = dropoutRate
        self.k = k
        self.mask_fn = mask_fn
        
        self.attention =  SelfMultiHeadAttention(k = k, heads = heads,
                                                 mask_fn=mask_fn)
        
        self.normLayer1 = LayerNormalization(k = k,eps=1e-6) # use much faster: tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        self.normLayer2 = LayerNormalization(k = k, eps=1e-6) # tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        
        M2 = hiddenInAnn * k # mostly bigger than the element dims
        self.ff = ANN(k, K = k , hidden_layer_sizes=[[M2,tf.nn.relu]])
        
        
        ## collect trainable params
        self.trainable_params = self.attention.trainable_params
        
        ## collect from norm layer gamma and beta (if you use faster keras normalization layer use trainable_variables)
        self.trainable_params += self.normLayer1.trainable_params
        self.trainable_params += self.normLayer2.trainable_params
        
        
        self.trainable_params += self.ff.trainable_params
    #@tf.function 
    def forward(self, x, is_training = False):
        attended = self.attention.forward(va=x, ke=x, qu=x)
        if is_training:
            attended = tf.nn.dropout(attended, rate = self.dropoutRate)
            
        norm1 = self.normLayer1.forward(attended + x)
        feedforward = self.ff.forward(norm1)        
        #### Drop out should be add here 
        if is_training:
            feedforward = tf.nn.dropout(feedforward, rate = self.dropoutRate)
            #x = tf.nn.experimental.stateless_dropout(x, rate = self.dropoutRate) <-- for tensorflow 2.7 and above
       
        norm2 = self.normLayer2.forward(feedforward + norm1 ) 
        return norm2


class TransformerEncoderBlockKeras():
    def __init__(self, k, heads, hiddenInAnn = 4, dropoutRate = 0.0, mask_fn = None):
        # k == features dimentions
        # heads = the number of heads in the multihead self attention 
        self.dropoutRate = dropoutRate
        self.k = k
        self.mask_fn = mask_fn
        
        self.attention =  SelfMultiHeadAttentionKeras(k = k, heads = heads,
                                                      mask_fn=mask_fn)
        
        self.normLayer1 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        self.normLayer2 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
        
        M2 = hiddenInAnn * k # mostly bigger than the element dims
        self.ff = AnnKears(dims=k, K = k ,hidden_layer_sizes=[[M2,tf.nn.relu]])
        
        ## run for initializtion
        x = tf.random.normal(shape = (2,4,k))
        self.forward(x, is_training=True)
        ## collect trainable params
        self.trainable_params = self.attention.trainable_params
        
        self.trainable_params += self.normLayer1.trainable_variables
        self.trainable_params += self.normLayer2.trainable_variables
        
        
        self.trainable_params += self.ff.trainable_params
    # @tf.function    
    def forward(self, x, is_training = False):
        attended = self.attention.forward(va=x, ke=x, qu=x)
        if is_training:
            attended = tf.nn.dropout(attended, rate = self.dropoutRate)
            
        norm1 = self.normLayer1.apply(attended + x)
        feedforward = self.ff.forward(norm1)
        
        if is_training:
            feedforward = tf.nn.dropout(feedforward, rate = self.dropoutRate)
             #x = tf.nn.experimental.stateless_dropout(x, rate = self.dropoutRate) <-- for tensorflow 2.7 and above
        norm2 = self.normLayer2.apply(feedforward + norm1) 

        return norm2 


# ############## Transformer Block Keras Vs My Transformer from scratch 
# x = tf.random.normal(shape = (2,4,3))  
# n,t,k = x.shape     
# trns = TransformerEncoderBlock(k, 8)  
# trnsKerass = TransformerEncoderBlockKeras(k, 8)  


# for w1, w2 in zip(trns.trainable_params, trnsKerass.trainable_params):
#     print(w1.shape, w2.shape)
#     w1.assign(w2)
# import time 
# t0 = time.time()    
# y = trns.forward(x, is_training=False)  
# t1 = time.time() 
# y2 = trnsKerass.forward(x, is_training=False)  
# t2 = time.time()  
# print("diff:", tf.reduce_mean(tf.math.abs(y-y2)), "ForwardTime-Keras:", t2-t1, "FrowardTimeFromScratch:", t1-t0)   


# # ############ With mask 
# from utils import create_look_ahead_mask
# mask = create_look_ahead_mask

# x = tf.random.normal(shape = (2,4,3))  
# n,t,k = x.shape     
# trns = TransformerEncoderBlock(k, heads=8, mask_fn = mask)  
# trnsKerass = TransformerEncoderBlockKeras(k, 8, mask_fn=mask)  


# for w1, w2 in zip(trns.trainable_params, trnsKerass.trainable_params):
#     print(w1.shape, w2.shape)
#     w1.assign(w2)
# import time 
# t0 = time.time()    
# y = trns.forward(x, is_training=True)  
# t1 = time.time() 
# y2 = trnsKerass.forward(x, is_training=True)  
# t2 = time.time()  
# print("diff:", tf.reduce_mean(y-y2), "ForwardTime-Keras:", t2-t1, "FrowardTimeFromScratch:", t1-t0)   

############# Check Decoder Block keras vs Me 
# from utils import create_look_ahead_mask
# k = 3
# x = tf.random.normal(shape = (3,5,3))
# mask_fn = create_look_ahead_mask
# decoderMe = TransformerDecoderBlock(k, heads = 8, hiddenInAnn = 4, 
#                                     dropoutRate = 0.0, mask_fn = mask_fn)
# decoderKeras = TransformerDecoderBlockKeras(k, heads = 8, hiddenInAnn = 4, 
#                                     dropoutRate = 0.0, mask_fn = mask_fn)

# for w, w2 in zip(decoderKeras.trainable_params, decoderMe.trainable_params):
#     # print(w.shape, w2.shape)
#     w.assign(w2)
#     # print(tf.reduce_mean(w - w2))
    
# yKeras = decoderKeras.forward(x,x, is_training=False)
# yMe = decoderMe.forward(x,x, is_training=False)

# print(tf.reduce_mean(tf.math.abs(yKeras - yMe)))
# TransformerDecoderBlock(object):
#     def __init_(self, k, heads, hiddenInAnn = 4, dropoutRate = 0.0, mask_fn = None):