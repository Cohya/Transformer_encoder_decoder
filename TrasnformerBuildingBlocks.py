
import tensorflow as tf 
# import numpy as np 


class DenseLayer():
    def __init__(self, M1, M2, f = tf.identity,use_bais = True, id_ = 0):
        W0 = tf.random.normal(shape = (M1, M2),mean=0.0, stddev=1.0)  * tf.math.sqrt(2./float(M1))
        self.W = tf.Variable(initial_value = W0, name = str(id_))
        if use_bais:
            self.b = tf.Variable(initial_value = tf.zeros((M2, )))
            self.trainable_params = [self.W, self.b]
            
        else:
            self.trainable_params = [self.W]
        
        self.f = f
        self.use_bais = use_bais
   
    def forward(self, x):
        Z = tf.matmul(x, self.W) 
        if self.use_bais:
            Z += self.b
            
        return self.f(Z)

class ANN(object):
    def __init__(self, dims, K, hidden_layer_sizes):
        #  K, hidden_layer_sizes = [[M2,f], ....]
        # num_channels <-- the depth of the input 
        self.K = K # the number of output nodes of the net (number of actions)
        
        
            
       
        # let's calculate the size of the input after fletten 
        M1 = dims
        ## collect all the fully connected layers
        self.dense_layers = []
        # # architecture hiiden_layer_sizes = [[M2, f]]
        counter =0 
        for M2, f in hidden_layer_sizes:
            layer = DenseLayer(M1 , M2, f= f,use_bais=True, id_ = counter)
            self.dense_layers.append(layer)
            M1 = M2
            counter += 1
            
        # Now let's creat the last layer 
        layer = DenseLayer(M1, self.K, f = tf.identity, use_bais=True) # linear
        self.dense_layers.append(layer)
        
        # collect the trainable params 
        self.trainable_params = []
        
        for layer in self.dense_layers:
            self.trainable_params += layer.trainable_params
            
    def forward(self, Z):
        # print(Z.shape)
        # Z = tf.float32(Z)
        for layer in self.dense_layers:
            Z = layer.forward(Z)
            
        return Z
    
    
class SelfMultiHeadAttention(object):#SelfAttention
    def __init__(self, k, heads = 8, mask_fn = None):
        # k - dims of the input (features) 
        # head - number of heads we use 
        self.k, self.heads = k, heads
        self.mask_fn = mask_fn
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        
        self.tokeys = DenseLayer(M1 = k, M2 = k * heads, use_bais= False, id_ = 0)
        self.toqueries = DenseLayer(M1 = k, M2 = k * heads, use_bais= False, id_ = 1)
        self.tovalues = DenseLayer(M1 = k, M2 = k * heads, use_bais= False, id_ = 2)
        
        # Now we would like to unifies the outputs 
        # of different heads into a single k - vector 
        
        self.unifyheads = DenseLayer(M1 = heads * k, M2 = k, use_bais=True)
        
        self.trainable_params = self.tokeys.trainable_params
        self.trainable_params += self.toqueries.trainable_params
        self.trainable_params += self.tovalues.trainable_params
        self.trainable_params += self.unifyheads.trainable_params
        
    def forward(self, va,ke,qu):
        """ x = (batch, time, features)"""
        assert len(va.shape) == 3
        
        b, t, k = va.shape
        h = self.heads
        
        queries = self.toqueries.forward(qu)
        queries = tf.reshape(queries, shape = (b,t,h,k)) #  (b, t,h*k) --> (b,t,h,k) each head with its own vector 
        
        keys = self.tokeys.forward(ke)
        keys = tf.reshape(keys, shape = (b,t,h,k))

        values = self.tovalues.forward(va)
        values = tf.reshape(values, shape = (b,t,h,k))
        
        keys = tf.transpose(keys, perm = (0,2,1,3)) # (b,h,t,k)
        queries = tf.transpose(queries, perm = (0,2,1,3))
        values = tf.transpose(values, perm = (0,2,1,3))
        
        keys = tf.reshape(keys, shape = (b*h, t,k))
        
        queries = tf.reshape(queries, shape = (b*h, t,k))
        values = tf.reshape(values, shape = (b*h, t,k))
        
        # normelization 
        queries = queries / (k**(1/4)) 
        keys = keys / (k**(1/4)) 
        
        weights = tf.matmul(queries,tf.transpose(keys, perm = (0,2,1))) # (bh, t, t)
      
        # Mask
        if self.mask_fn: # mask out the lower half of the dot matrix,including the diagonal
            # print("in Mask")
            # teh mask can be padding mask 
            m = self.mask_fn(t) 
            weights  += (m * -1e9)  #"""shold be multiply"""
            
        weights = tf.nn.softmax(weights, axis = -1) #axis = 2 is the same normelized along the columns
        
            
            
        out = tf.matmul(weights, values) # (bh, t, k)
        # out = tf.transpose(out, perm = (0,2,1)) # (bh, k,t) <--- why ???
        out = tf.reshape(out, shape = (b,t,h*k))
        return self.unifyheads.forward(out)
 
    


class LayerNormalization():
    def __init__(self, k, eps = 1e-6):
        self.k = k
        self.eps = eps
        self.gamma = tf.Variable(initial_value = tf.ones(shape = k),name = "gamma")
        self.beta = tf.Variable(initial_value = tf.zeros(shape = k), name = 'beta')
        self.trainable_params = [self.gamma, self.beta]
        
    def forward(self, x):
        n,t,k = x.shape
        
        for i in range(n):#len(x)
            x_i = x[i]
            ni,_ = x_i.shape
            for j in range(ni): #len(x_i)
                if j == 0: 
                    x_ij = x_i[j]
                    x_norm_ij = self.gamma * (x_ij - tf.reduce_mean(x_ij))/(tf.math.reduce_std(x_ij) + self.eps) + self.beta
                    sample=  x_norm_ij
                    sample = tf.expand_dims(sample, axis = 0)
                else:
                    x_ij = x_i[j]
                    x_norm_ij = self.gamma * (x_ij - tf.reduce_mean(x_ij))/(tf.math.reduce_std(x_ij) + self.eps) + self.beta
                    x_norm_ij = tf.expand_dims(x_norm_ij, axis = 0)
                    sample = tf.concat((sample, x_norm_ij), axis = 0)
                    
            if i ==0:
                x_norm2 = tf.expand_dims(sample, axis = 0)
            else:
                x_norm2_i = tf.expand_dims(sample, axis = 0)
                x_norm2 = tf.concat((x_norm2, x_norm2_i), axis = 0 )
        return x_norm2
    

class  SelfMultiHeadAttentionKeras(object):
   
    def __init__(self, k, heads = 8, mask_fn = None):
        # k - dims of the input (features) 
        # head - number of heads we use 
        self.k, self.heads = k, heads
        self.mask_fn = mask_fn
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        M1 = k
        
        self.tokeys = tf.keras.layers.Dense(units = k * heads, use_bias= False)
        self.toqueries = tf.keras.layers.Dense(units = k * heads, use_bias= False)
        self.tovalues = tf.keras.layers.Dense(units = k * heads, use_bias= False)
        ## activate layers
        x = tf.random.normal(shape = (1,k, M1))
        self.tokeys.apply(x)
        self.toqueries.apply(x)
        self.tovalues.apply(x) 
        
        
        # Now we would like to unifies the outputs 
        # of different heads into a single k - vector 
        M1 = heads * k
        self.unifyheads = tf.keras.layers.Dense(units = k, use_bias = True)
        x = tf.random.normal(shape = (1,k, M1))
        self.unifyheads.apply(x)
        
        self.trainable_params = self.tokeys.trainable_variables
        self.trainable_params += self.toqueries.trainable_variables
        self.trainable_params += self.tovalues.trainable_variables
        self.trainable_params += self.unifyheads.trainable_variables
        
    def forward(self,va,ke,qu):
         # call(self, v, k, q) <--- for better approach 
        """ x = (batch, time, features)"""
        assert len(va.shape) == 3
        
        b, t, k = va.shape
        h = self.heads
        
        queries = self.toqueries.call(qu)
        queries = tf.reshape(queries, shape = (b,t,h,k)) #  (b, t,h*k) --> (b,t,h,k) each head with its own vector 
        
        keys = self.tokeys.call(ke)
        keys = tf.reshape(keys, shape = (b,t,h,k))

        values = self.tovalues.call(va)
        values = tf.reshape(values, shape = (b,t,h,k))
        
        keys = tf.transpose(keys, perm = (0,2,1,3)) # (b,h,t,k)
        queries = tf.transpose(queries, perm = (0,2,1,3))
        values = tf.transpose(values, perm = (0,2,1,3))
        
        keys = tf.reshape(keys, shape = (b*h, t,k))
        
        queries = tf.reshape(queries, shape = (b*h, t,k))
        values = tf.reshape(values, shape = (b*h, t,k))
        
        # normelization 
        queries = queries / (k**(1/4)) 
        keys = keys / (k**(1/4)) 
        
        weights = tf.matmul(queries,tf.transpose(keys, perm = (0,2,1))) # (bh, t, t)
        
        ##Mask 
        if self.mask_fn :
            # print("in Mask")
            m = self.mask_fn(t)
            weights += (m * -1e9)
        
        
        weights = tf.nn.softmax(weights, axis = -1) #axis = 2 is the same normelized along the columns
        out = tf.matmul(weights, values) # (bh, t, k)
        # out = tf.transpose(out, perm = (0,2,1)) # (bh, k,t) <--- why ???
        out = tf.reshape(out, shape = (b,t,h*k))
        return self.unifyheads.call(out)    

class AnnKears():
    def __init__(self, dims, K, hidden_layer_sizes):
        self.dims = dims
        self.K = K 
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense_layers = []
        
        M1 = dims ### the featres dimentions 
        for M2, f in hidden_layer_sizes:
            layer = tf.keras.layers.Dense(units = M2, activation=f, use_bias=True)
            ## activat layer step 
            self.dense_layers.append(layer)
            M1 = M2
            
        ## last layer 
        layer = tf.keras.layers.Dense(units = K, activation=tf.identity, use_bias=True)
        
        self.dense_layers.append(layer)
        
        ## actuavate layer
        x = tf.random.normal(shape = (1, 2,dims))
        self.forward(x)
        
        self.trainable_params = []
        for layer in self.dense_layers:
            self.trainable_params += layer.trainable_variables
            
    def forward(self,x):
        Z = x
        for layer in self.dense_layers:
            Z = layer.apply(Z)
        
        return Z
 
### ANN vs ANNkeras check similarity            
# x = tf.random.normal(shape = (1,4,3))           
# n, t, k = x.shape
# M2 = 4

# ff = ANN(k, K = k , hidden_layer_sizes=[[M2,tf.nn.relu]])  
# ffKeras = AnnKears(dims=k, K = k ,hidden_layer_sizes=[[M2,tf.nn.relu]] )
       
            
# for w1, w2 in zip(ff.trainable_params, ffKeras.trainable_params):
#     print(w1.shape, w2.shape)
#     w1.assign(w2)
    
# print("diff:", tf.reduce_mean(ff.forward(x) - ffKeras.forward(x)))
        
#####################################################################
####Check if the keras is the same as what I prepared Multiheadattention
# k = 3
# x = tf.random.normal(shape = (3, 4,k))
# mhaKeras = SelfMultiHeadAttentionKeras(k = k, heads=  8)
# mha = SelfMultiHeadAttention(k=k, heads=8)

# for w1, wk in zip(mha.trainable_params, mhaKeras.trainable_params):
#     print(w1.shape, wk.shape)
#     wk.assign(w1)
    
# del_i = mha.forward(x,x,x) - mhaKeras.forward(x,x,x)
# print(tf.reduce_mean(del_i))
##############################################################3

### Check you layer Normalization vs Keras Layer
# x = tf.random.normal(shape = (2,4,3))
# n,t,k = x.shape
# layer = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
# x_norm= layer.apply(x)
# layer.weights[0].assign([9,9,9])
# layer.weights[1].assign([9,9,9])
# x_norm= layer.apply(x)
# layer2 = LayerNormalization(k)
# layer2.trainable_params[0].assign([9,9,9])#gamma
# layer2.trainable_params[1].assign([9,9,9])#beta
# x_norm2 = layer2.forward(x)
# print("diff:", tf.reduce_mean(x_norm2 - x_norm))

# layer.weights[0].assign([1,0.5,1])
# layer.weights[1].assign([1,0.5,1])
# x_norm= layer.apply(x)

# x_mean = tf.reduce_mean(x, axis = -1)
# x_std = tf.math.reduce_std(x, axis = -1)

# k = 3

# gamma = tf.Variable(initial_value = tf.ones(shape = k))
# gamma.assign(layer.weights[0])
# beta = tf.Variable(initial_value = tf.zeros(shape = k))
# beta.assign(layer.weights[1])
# for i in range(len(x)):
#     x_i = x[i]
#     for j in range(len(x_i)):
#         if j == 0: 
#             x_ij = x_i[j]
#             x_norm_ij = gamma * (x_ij - tf.reduce_mean(x_ij))/(tf.math.reduce_std(x_ij) + 1e-6) + beta
#             sample=  x_norm_ij
#             sample = tf.expand_dims(sample, axis = 0)
#         else:
#             x_ij = x_i[j]
#             x_norm_ij = gamma * (x_ij - tf.reduce_mean(x_ij))/(tf.math.reduce_std(x_ij) + 1e-6) + beta
#             x_norm_ij = tf.expand_dims(x_norm_ij, axis = 0)
#             sample = tf.concat((sample, x_norm_ij), axis = 0)
            
#     if i ==0:
#         x_norm2 = tf.expand_dims(sample, axis = 0)
#     else:
#         x_norm2_i = tf.expand_dims(sample, axis = 0)
#         x_norm2 = tf.concat((x_norm2, x_norm2_i), axis = 0 )

# print("diff:", tf.reduce_sum(x_norm - x_norm2)) 
# meanX = tf.reduce_mean(x, axis = -1)
# stdX = tf.math.reduce_std(x, axis = -1)
# count  = 0
# for i in range(len(x)):
#     x_i = tf.expand_dims(x[i], axis = 0)
#     if count == 0:
#         x_normi = (x_i - meanX[i]) / stdX[i]
#     else:
#         x_normi2 = (x_i - meanX[i]) / (stdX[i] + 1e-6)
#         x_normi = tf.concat((x_normi, x_normi2), axis = 0)
    
#     count += 1
# for i in range(len(x)):
#     i= 0 
#     x_i = x[i]
#     t, k = x_i.shape
#     sum_x_i = tf.reduce_sum(x_i)
#     meanX = sum_x_i/(t*k)
#     std = tf.math.sqrt(tf.reduce_sum((x_i - meanX)**2) / (t*k))
    
    

   
        

# x = tf.random.normal(shape = (2,4,3))
# layer = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
# layer2 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6)
# x_norm= layer.apply(x)


# x_mean = tf.reduce_mean(x, axis = -1)
# x_std = tf.math.reduce_std(x, axis = -1)


# # x_stss =  sum([(x[j] - x_mean) ** 2 for j in range(9)]) / 9
# meanX0 = [tf.reduce_mean(xi) for xi in x]
# stdX0 = [tf.math.reduce_std(xi) for xi in x]

# meanX = tf.reduce_mean(x, axis = (1,2))
# stdX = tf.math.reduce_std(x, axis = (1,2))
# count  = 0
# for i in range(len(x)):
#     x_i = tf.expand_dims(x[i], axis = 0)
#     if count == 0:
#         x_normi = (x_i - meanX[i]) / stdX[i]
#     else:
#         x_normi2 = (x_i - meanX[i]) / (stdX[i] + 1e-6)
#         x_normi = tf.concat((x_normi, x_normi2), axis = 0)
    
#     count += 1





