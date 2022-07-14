from TransformerBlock import (TransformerDecoderBlock, TransformerEncoderBlockKeras,
                                TransformerDecoderBlockKeras, TransformerEncoderBlock)

from utils import positional_endcoding,create_look_ahead_mask
import tensorflow as tf 

class EncoderKeras():
    def __init__(self, num_layers, k, num_of_heads, dff = 4, dropoutRate = 0.0):
        self.k = k 
        self.num_of_heads = num_of_heads
        self.dff = dff # hidden layer number = dff * k
        self.dropoutRate = dropoutRate
        
        self.layers = []
        
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderBlockKeras(k = k,
                                                            heads = num_of_heads,
                                                            hiddenInAnn = dff,
                                                            dropoutRate = dropoutRate, 
                                                            mask_fn = None))
            
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
        # self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
         
         
    def forward(self, x, is_training = False):
        ## We assumed, that x is already encoded to numbers 
        # # x should have a shape of (n,t,k) 
        # t  = tf.shape(x)[1]
        
        # #positional embadding
        # x += self.pos_enbadding[:,:t,:]
        # print(self.pos_enbadding[:,:t,:],self.pos_enbadding[:,:t,:].shape)
        if is_training:
            x = tf.nn.dropout(x, rate = self.dropoutRate)
            
        for layer in self.layers:
            x = layer.forward(x, is_training)
            
        return x 
    
class Encoder():
    def __init__(self, num_layers, k, num_of_heads, dff = 4, dropoutRate = 0.0):
        self.k = k 
        self.num_of_heads = num_of_heads
        self.dff = dff # hidden layer number = dff * k
        self.dropoutRate = dropoutRate
        
        self.layers = []
        
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderBlock(k = k,
                                                            heads = num_of_heads,
                                                            hiddenInAnn = dff,
                                                            dropoutRate = dropoutRate, 
                                                            mask_fn = None))
            
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
        # self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
         
         
    def forward(self, x, is_training = False):
        ## We assumed, that x is already encoded to numbers 
        # x should have a shape of (n,t,k) 
        # t  = tf.shape(x)[1]
        
        # #positional embadding
        # x += self.pos_enbadding[:,:t,:]
        # # print(self.pos_enbadding[:,:t,:],self.pos_enbadding[:,:t,:].shape)
        if is_training:
            x = tf.nn.dropout(x, rate = self.dropoutRate)
            
        for layer in self.layers:
            x = layer.forward(x, is_training)
            
        return x 
 
class DecoderKeras():
    def __init__(self, num_layers, k, num_of_heads, dff = 4, dropoutRate = 0.0, 
                 mask_fn = create_look_ahead_mask):
        
        self.k = k # this is d_model 
        self.num_layers = num_layers
        self.num_of_heads = num_of_heads
        self.mask_fn = mask_fn
        self.dropoutRate = dropoutRate
        # Collect all layers
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerDecoderBlockKeras(k = k,
                                                 heads=num_of_heads,
                                                 hiddenInAnn = dff, 
                                                 dropoutRate = dropoutRate,
                                                 mask_fn = mask_fn)
            self.layers.append(layer)
            
        ## collect all trainable params 
        self.trainable_params =[]
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
        
        # self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
        
        
    def forward(self, x, x_encoder, is_training = False):
        # we assumed that x is already embedded (means numbers)
        # x should have a shape of (n,t,k) 
        # t  = tf.shape(x)[1]
        
        # #positional embadding
        # x += self.pos_enbadding[:,:t,:]
        
        if is_training:
            x = tf.nn.dropout(x, rate = self.dropoutRate)
            
        for layer in self.layers:
            x = layer.forward(x, x_encoder, is_training=is_training)
        
        return x 

class Decoder():
    def __init__(self, num_layers, k, num_of_heads, dff = 4, dropoutRate = 0.0, 
                 mask_fn = create_look_ahead_mask):
        
        self.k = k # this is d_model 
        self.num_layers = num_layers
        self.num_of_heads = num_of_heads
        self.mask_fn = mask_fn
        self.dropoutRate = dropoutRate
        # Collect all layers
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerDecoderBlock(k = k,
                                                 heads=num_of_heads,
                                                 hiddenInAnn = dff, 
                                                 dropoutRate = dropoutRate,
                                                 mask_fn = mask_fn)
            self.layers.append(layer)
            
        ## collect all trainable params 
        self.trainable_params =[]
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
        
        # self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
        
        
    def forward(self, x, x_encoder, is_training = False):
        # we assumed that x is already embedded (means numbers)
        # x should have a shape of (n,t,k) 
        # t  = tf.shape(x)[1]
        
        # #positional embadding
        # x += self.pos_enbadding[:,:t,:]
        
        if is_training:
            x = tf.nn.dropout(x, rate = self.dropoutRate)
            
        for layer in self.layers:
            x = layer.forward(x, x_encoder, is_training=is_training)
        
        return x 
        
            
        
#################3 check Encoder!
# x = tf.random.normal(shape = (10,4,4))           
        
# encoderkeras = EncoderKeras(1,4,8)      
# encoder = Encoder(1,4,8)  
# y = encoder.forward(x)

# for w, w2 in zip(encoderkeras.trainable_params, encoder.trainable_params):
#     w.assign(w2)
    
# x = tf.random.normal(shape = (1,4,4))
# y1 = encoder.forward(x)
# y2 = encoderkeras.forward(x)

# print("diff:", tf.reduce_mean(tf.math.abs(y1 - y2)))
#########33 check Decoder
# decoderKeras = DecoderKeras(num_layers = 4, k=4, num_of_heads=8, dff = 4, dropoutRate = 0.0, 
#                   mask_fn = create_look_ahead_mask)

# decoder= Decoder(num_layers=4, k=4, num_of_heads=8, dff = 4, dropoutRate = 0.0, 
#                   mask_fn = create_look_ahead_mask)

# for w, w2 in zip(decoderKeras.trainable_params, decoder.trainable_params):
#     w.assign(w2)
    
# x = tf.random.normal(shape = (1,4,4))
# y1 = decoder.forward(x, x)
# y2 = decoderKeras.forward(x,x)

# print("diff:", tf.reduce_mean(tf.math.abs(y1 - y2)))
    




