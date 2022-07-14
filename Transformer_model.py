
import tensorflow as tf 
from Encoder_Decoder_Transformer import Encoder, EncoderKeras, Decoder, DecoderKeras
from utils import create_look_ahead_mask, positional_endcoding
from TrasnformerBuildingBlocks import DenseLayer

Max_Tokens = 100 # number of look back 

class Transformer_Encoder_Decoder_Keras():
    def __init__(self, num_layer_encoder, num_layer_decoder, dff, d_model, targetDims ,
                 num_of_heads = 8, dropoutRate=0.0, activate_embedding = False, d_model_embed = 0):
        """
        # d_model == k is the numebr of dims of each token 
        # dff -> dff*k =the number of hidden units in the ANN architecture in the Encoder/decoder block
     
        # targetDims -> the output dims of the transformer 
        # num_heads --> number of heads in the multi-selfAttention step 
        # num_layer_encoder, num_layer_decoder -> number of layers we use in the encoder and decoder, respectively 
        """
        self.dropoutRate = dropoutRate
        self.k = d_model
        self.num_layer_encoder = num_layer_encoder
      
        self.encoder = EncoderKeras(num_layers = num_layer_encoder,
                               k = d_model,
                               num_of_heads = num_of_heads,
                               dff = dff,
                               dropoutRate = dropoutRate)
        
        if activate_embedding:
            self.activate_embedding = activate_embedding
            self.layerEmbedding = tf.keras.layers.Dense(d_model)
            # self.layerEmbeddingDecoder = tf.keras.layers.Dense(d_model)
            ## activate the layers 
            x = tf.random.normal(shape = (3, 3, d_model_embed))
            self.layerEmbedding.apply(x)
            
            
        self.decoder = DecoderKeras(num_layers = num_layer_decoder,
                                    k = d_model,
                                    num_of_heads = num_of_heads,
                                    dff = dff,
                                    dropoutRate = dropoutRate, 
                                    mask_fn = create_look_ahead_mask)
        #Final layer 
        self.final_layer = tf.keras.layers.Dense(targetDims)
        ## activate final layer 
        x = tf.random.normal(shape = (3,3,d_model))
        self.final_layer.apply(x)
        
        ##  collect trainable params 
        self.trainable_params = []
        self.trainable_params += self.layerEmbedding.trainable_variables
        self.trainable_params += self.encoder.trainable_params
        self.trainable_params += self.decoder.trainable_params
        self.trainable_params += self.final_layer.trainable_variables
        
        
        self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
        
    def forward(self, inputs, is_training = False):
        x_encoder, x_decoder = inputs
        
        if self.activate_embedding:
            x_encoder = self.layerEmbedding.apply(x_encoder)
            x_decoder = self.layerEmbedding.apply(x_decoder)
        # x should have a shape of (n,t,k) 
        t_encoder  = tf.shape(x_encoder)[1]
        t_decoder = tf.shape(x_decoder)[1]
        
        #positional embadding
        x_encoder += self.pos_enbadding[:,:t_encoder,:]
        x_decoder += self.pos_enbadding[:,:t_decoder,:]
        
        encoder_output = self.encoder.forward(x_encoder, is_training = is_training)
        
        
        decoder_out = self.decoder.forward(x_decoder, encoder_output,
                                           is_training = is_training) # (n,t,k)
        
        # print(decoder_out.shape)
        final_output = self.final_layer.apply(decoder_out) # (n, t, target_dims)
        # print(final_output.shape)
        return final_output #,decoder_out, encoder_output
        
       
class Transformer_Encoder_Decoder():
    def __init__(self, num_layer_encoder, num_layer_decoder, dff, d_model, targetDims ,
                 num_of_heads = 8, dropoutRate=0.0, activate_embedding = False, d_model_embed = 0):
        """
        # d_model == k is the numebr of dims of each token 
        # dff -> dff*k =the number of hidden units in the ANN architecture in the Encoder/decoder block
     
        # targetDims -> the output dims of the transformer 
        # num_heads --> number of heads in the multi-selfAttention step 
        # num_layer_encoder, num_layer_decoder -> number of layers we use in the encoder and decoder, respectively 
        """
        self.dropoutRate = dropoutRate
        self.k = d_model
        self.num_layer_encoder = num_layer_encoder
        
        self.encoder = Encoder(num_layers = num_layer_encoder,
                               k = d_model,
                               num_of_heads = num_of_heads,
                               dff = dff,
                               dropoutRate = dropoutRate)
        
        if activate_embedding:
            self.activate_embedding = activate_embedding
            self.layerEmbedding = DenseLayer(M1 = d_model_embed, M2 = d_model)
            # self.layerEmbeddingDecoder = tf.keras.layers.Dense(d_model)
            
            
        self.decoder = Decoder(num_layers = num_layer_decoder,
                                    k = d_model,
                                    num_of_heads = num_of_heads,
                                    dff = dff,
                                    dropoutRate = dropoutRate, 
                                    mask_fn = create_look_ahead_mask)
        #Final layer 
        self.final_layer = DenseLayer(M1 = d_model , M2 = targetDims)
        
        ##  collect trainable params 
        self.trainable_params = []
        self.trainable_params += self.layerEmbedding.trainable_params
        self.trainable_params += self.encoder.trainable_params
        self.trainable_params += self.decoder.trainable_params
        self.trainable_params += self.final_layer.trainable_params
        
        
        self.pos_enbadding = positional_endcoding(Max_Tokens, self.k)
        
    def forward(self, inputs, is_training = False):
        x_encoder, x_decoder = inputs
        
        if self.activate_embedding:
            x_encoder = self.layerEmbedding.forward(x_encoder)
            x_decoder = self.layerEmbedding.forward(x_decoder)
        # x should have a shape of (n,t,k) 
        t_encoder  = tf.shape(x_encoder)[1]
        t_decoder = tf.shape(x_decoder)[1]
        
        #positional embadding
        x_encoder += self.pos_enbadding[:,:t_encoder,:]
        x_decoder += self.pos_enbadding[:,:t_decoder,:]
        
        encoder_output = self.encoder.forward(x_encoder, is_training = is_training)
        
        
        decoder_out = self.decoder.forward(x_decoder, encoder_output,
                                         is_training = is_training) # (n,t,k)
        
        final_output = self.final_layer.forward(decoder_out) # (n, t, target_dims)
        
        return final_output #,decoder_out, encoder_output
  
## Check the transformerbased my layers ve Keras
# k = 1 
# d_model = 10    
# x = tf.random.normal(shape = (10, 3,k))

# transformerK = Transformer_Encoder_Decoder_Keras(num_layer_encoder = 1,
#                                                 num_layer_decoder = 1,
#                                                 dff = 4, d_model= d_model,
#                                                 targetDims = k ,
#                                                 num_of_heads = 8,
#                                                 dropoutRate=0.0,
#                                                 activate_embedding = True,
#                                                 d_model_embed = k)
# transformer = Transformer_Encoder_Decoder(num_layer_encoder = 1,
#                                                 num_layer_decoder = 1,
#                                                 dff = 4, d_model= d_model,
#                                                 targetDims = k ,
#                                                 num_of_heads = 8,
#                                                 dropoutRate=0.0,
#                                                 activate_embedding = True,
#                                                 d_model_embed = k)

# for w1, w2 in zip(transformerK.trainable_params, transformer.trainable_params):
#     w1.assign(w2)
#     print(w1.shape, w2.shape, tf.reduce_sum(w1 - w2).numpy())
    
    
    
# X = [x,x]
# # y, ydecoder, yencoder = transformer.forward(X)
# # yK, yKdecoder, yKencoder = transformerK.forward(X)
# # print("diff General:", tf.reduce_mean(tf.math.abs(y - yK)),
# #       "diff Decoder:", tf.reduce_mean(tf.math.abs(ydecoder - yKdecoder)),
# #       "diff Encoder:", tf.reduce_mean(tf.math.abs(yencoder - yKencoder)))


# y= transformer.forward(X)
# yK = transformerK.forward(X)
# print("diff General:", tf.reduce_mean(tf.math.abs(y - yK)))


