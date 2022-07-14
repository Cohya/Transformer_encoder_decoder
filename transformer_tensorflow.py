
import tensorflow as tf 

class Dense():
    def __init__(self, M1, M2, activation = tf.identity,use_bais = False, id_ = 0):
        W0 = tf.random.normal(shape = (M1, M2),mean=0.0, stddev=1.0)  * tf.math.sqrt(2./float(M1))
        self.W = tf.Variable(initial_value = W0, name = str(id_))
        if use_bais:
            self.b = tf.Variable(initial_value = tf.zeros((M2, )))
            self.trainable_params = [self.W, self.b]
            
        else:
            self.trainable_params = [self.W]
        
        self.f = activation
        self.use_bais = use_bais
        
    def forward(self, x):
        Z = tf.matmul(x, self.W) 
        if self.use_bais:
            Z += self.b
            
        return self.f(Z)
    
    
class MultiHeadAttention(object):
    def __init__(self, k, heads = 8):
        # k - dims of the input (features) 
        # head - number of heads we use 
        self.k, self.heads = k, heads
        
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        
        self.tokeys = Dense(M1 = k, M2 = k * heads, use_bais= False, id_ = 0)
        self.toqueries = Dense(M1 = k, M2 = k * heads, use_bais= False, id_ = 1)
        self.tovalues = Dense(M1 = k, M2 = k * heads, use_bais= False, id_ = 2)
        
        # Now we would like to unifies the outputs 
        # of different heads into a single k - vector 
        
        self.unifyheads = Dense(M1 = heads * k, M2 = k)
        
        self.trainable_params = self.tokeys.trainable_params
        self.trainable_params += self.toqueries.trainable_params
        self.trainable_params += self.tovalues.trainable_params
        self.trainable_params += self.unifyheads.trainable_params
        
    def forward(self, x):
        """ x = (batch, time, features)"""
        assert len(x.shape) == 3
        
        b, t, k = x.shape
        h = self.heads
        
        queries = self.toqueries.forward(x)
        queries = tf.reshape(queries, shape = (b,t,h,k)) #  (b, t,h*k) --> (b,t,h,k) each head with its own vector 
        
        keys = self.tokeys.forward(x)
        keys = tf.reshape(keys, shape = (b,t,h,k))

        values = self.tovalues.forward(x)
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
        weights = tf.nn.softmax(weights, axis = 2) # normelized along the columns
        out = tf.matmul(weights, values) # (bh, t, k)
        # out = tf.transpose(out, perm = (0,2,1)) # (bh, k,t) <--- why ???
        out = tf.reshape(out, shape = (b,t,h*k))
        return self.unifyheads.forward(out), out
 

## Just make sure that we wrote as we should do 

class SingleSelfAttention():
    def __init__(self, k): 
        
        self.k = k
        
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        
        self.tokeys = Dense(M1 = k, M2 = k , use_bais= False, id_ = 0)
        self.toqueries = Dense(M1 = k, M2 = k, use_bais= False, id_ = 1)
        self.tovalues = Dense(M1 = k, M2 = k , use_bais= False, id_ = 2)
    
        self.trainable_params = self.tokeys.trainable_params
        self.trainable_params += self.toqueries.trainable_params
        self.trainable_params += self.tovalues.trainable_params
        
    def forward(self,x):
        queries = self.toqueries.forward(x)/ (self.k**(1/4))
        keys = self.tokeys.forward(x)/ (self.k**(1/4))
        values = self.tovalues.forward(x)
        
        w = tf.matmul(queries, tf.transpose(keys, perm = (0,2,1)))
        w = tf.nn.softmax(w, axis = 2)
        
        head = tf.matmul(w, values)
        
        return head

class MultiHeadAttentionWithLoop():
    def __init__(self, k, heads):
        
        self.k , self.h = k, heads
        self.headLayers = []
        for _ in range(heads):
            layer = SingleSelfAttention(k)
            self.headLayers.append(layer)
            
        self.trainable_params = []
        for layer in self.headLayers:
            self.trainable_params += layer.trainable_params
            
        self.linearUnify = Dense(M1 = heads * k,M2 = k)
        
        self.trainable_params += self.linearUnify.trainable_params
    def forward(self, x):
        b, t, k = x.shape
        count = 0
        for layer in self.headLayers:
            if count == 0 :
                headsConcat = layer.forward(x)
                headsConcat2 = layer.forward(x)
            else:
                head = layer.forward(x)
                headsConcat = tf.concat((headsConcat, head), axis = 0)
                headsConcat2 = tf.concat((headsConcat2, head), axis = 2)
            count += 1
            # print(headsConcat.shape)
        h = count
        # headsConcat = tf.transpose(headsConcat, perm = (0,2,1)) # (bh, k,t)
        headsConcat = tf.reshape(headsConcat, shape = (b,t,h*k))
        print("headsConcat2:", headsConcat2)
        print("headsConcat ", headsConcat )
        output = self.linearUnify.forward(headsConcat)
        # print(output.shape)
        return output, headsConcat
            
            
            
        
        
        
        
k = 3       
       
x = tf.random.normal(shape = (1,4,3))
mhaBetter = MultiHeadAttention(k = 3, heads = 2)
mha = MultiHeadAttentionWithLoop(k = 3,heads = 2)


## let's make the weights the same
count  = 0
for i in range(len(mha.headLayers)):
    layer = mha.headLayers[i]
    if count  == 0:
        Wtokeys = layer.tokeys.W #Dense(M1 = k, M2 = k , use_bais= False, id_ = 0)
        Wtoqueries = layer.toqueries.W #Dense(M1 = k, M2 = k, use_bais= False, id_ = 1)
        Wtovalues = layer.tovalues.W
        
    else:
        Wtokeys = tf.concat((Wtokeys, layer.tokeys.W), axis = 1)
        Wtoqueries = tf.concat((Wtoqueries, layer.toqueries.W), axis = 1)
        Wtovalues = tf.concat((Wtovalues, layer.tovalues.W), axis = 1)
        
    count += 1
        
Wunify = mha.trainable_params[-1] 

mhaBetter.trainable_params[0].assign(Wtokeys)
mhaBetter.trainable_params[1].assign(Wtoqueries)
mhaBetter.trainable_params[2].assign(Wtovalues)
mhaBetter.trainable_params[3].assign(Wunify)

print(tf.reduce_sum(mhaBetter.tokeys.W - Wtokeys),
tf.reduce_sum(mhaBetter.toqueries.W - Wtoqueries),
tf.reduce_sum(mhaBetter.tovalues.W - Wtovalues),
tf.reduce_sum(mhaBetter.unifyheads.W -  Wunify))




y,h = mhaBetter.forward(x)       
y2,h2 = mha.forward(x)     

print(tf.reduce_sum(y - y2))   
        
## check why is it differnet
b,t,h,k = 1, 4,2,3
quries = mhaBetter.toqueries.forward(x)        
quries  = tf.reshape(quries , shape = (b,t,h,k))        
quries = tf.transpose(quries, perm = (0,2,1,3)) 


count = 0
queries2_vec =[]
for layer in mha.headLayers:
    if count == 0 :
        quries2 = layer.toqueries.forward(x)
        queries2_vec.append(quries2)
        
    else:
        head = layer.toqueries.forward(x)
        queries2_vec.append(head)
        quries2 = tf.concat((quries2, head), axis = 2)
        
    count += 1
    # print(headsConcat.shape)
    
quries2  = tf.reshape(quries2 , shape = (b,t,h,k))           
quries2 = tf.transpose(quries2, perm = (0,2,1,3))        
print(quries - quries2)  
        
values = mhaBetter.tovalues.forward(x)        
values   = tf.reshape(values , shape = (b,t,h,k))           
values = tf.transpose(values , perm = (0,2,1,3))         

count = 0
for layer in mha.headLayers:
    if count == 0 :
        values2 = layer.tovalues.forward(x)
        
    else:
        head = layer.tovalues.forward(x)
        values2  = tf.concat((values2 , head), axis = 2)
        
    count += 1       
        
values2   = tf.reshape(values2 , shape = (b,t,h,k))           
values2 = tf.transpose(values2 , perm = (0,2,1,3))        
   
print(values - values2 )       
        
     
keys = mhaBetter.tokeys.forward(x)        
keys  = tf.reshape(keys , shape = (b,t,h,k))           
keys= tf.transpose(keys , perm = (0,2,1,3))     
       
keys2_vec =[]
count = 0
for layer in mha.headLayers:
    if count == 0 :
        keys2 = layer.tokeys.forward(x)
        keys2_vec.append(keys2)
        
    else:
        head = layer.tokeys.forward(x)
        keys2_vec.append(head)
        keys2 = tf.concat((keys2 , head), axis = 2)
        
    count += 1       
     
keys2  = tf.reshape(keys2 , shape = (b,t,h,k))           
keys2= tf.transpose(keys2 , perm = (0,2,1,3))    

print(keys- keys2 )  
     
keys = tf.reshape(keys, shape = (b*h, t,k))/ (k**(1/4))
quries = tf.reshape(quries, shape = (b*h, t,k))/ (k**(1/4))
values = tf.reshape(values, shape = (b*h, t,k))

w = tf.matmul(quries, tf.transpose(keys, perm = (0,2,1)))
w = tf.nn.softmax(w, axis = 2)       

quries2 = tf.reshape(quries2, shape = (b*h, t,k)) / (k ** (1/4))
values2 = tf.reshape(values2, shape = (b*h, t,k))
keys2 = tf.reshape(keys2, shape = (b*h, t,k))/ (k**(1/4))
w2 = tf.matmul(quries, tf.transpose(keys2, perm = (0,2,1)))
w2 = tf.nn.softmax(w2, axis = 2) 



out = tf.matmul(w, values) # (bh, t, k)
out = tf.transpose(out, perm = (0,2,1)) # (bh, k,t)
out = tf.reshape(out, shape = (b,t,h*k))

out2 = tf.matmul(w2, values2) # (bh, t, k)
out2 = tf.transpose(out2, perm = (0,2,1)) # (bh, k,t)
out2 = tf.reshape(out2, shape = (b,t,h*k))