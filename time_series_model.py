import tensorflow as tf 
from Transformer_model import Transformer_Encoder_Decoder_Keras,Transformer_Encoder_Decoder
from sklearn.utils import shuffle
import numpy as np 
import matplotlib.pyplot as plt 
from utils import getSeriesData, prepare_data_for_encoder_decoder_transformer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# how the learning rate decrease
# lrate = d_model(-0.5) * min(step_num^(-0.5), step_num*warmup_steps(-1.5))

class Model():
    def __init__(self, nnet , learning_rate = 0.001):
        self.nnet = nnet
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, 
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        
        self.loss_func = tf.keras.losses.Huber()
        
        
    def forward(self, x, is_training = False):
        # x.shape = [(n, t, k)
        y_hat = self.nnet.forward(x, is_training = is_training)
        
        return y_hat 
    
    def cost(self, x, target, is_training = True):
        
        y_hat = self.forward(x, is_training = is_training)
        
        # only the last prediction 
        cost_i = self.loss_func(y_true = target[:,-1,0],
                                y_pred = y_hat[:,-1,0] )
        
        return cost_i
    
    def update_weights(self, x, target, is_training=True):
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            cost_i = self.cost(x, target, is_training = is_training)
            
        gradients = tape.gradient(cost_i, self.nnet.trainable_params)
        self.optimizer.apply_gradients(zip(gradients, self.nnet.trainable_params))
    
        return cost_i
            
        
    def train(self,x_train, y_train, x_test, y_test, batch_sz = 32, epochs = 1, verbos = True):
        ## make shure that x_train is an numpy array for the shuffle function 
        X_encoder, X_decoder = x_train
        n = len(X_encoder)
        n_batchs = n // batch_sz
        print(n_batchs)
        cost_vec = []
        cost_test_vec = []
        initVec = []
        for epoch in range(epochs):
            # print("sdfsdf")
            X_enc, X_dec , Y= shuffle(X_encoder, X_decoder, y_train)
            
            for j in range(n_batchs):
                x_enc_batch = X_enc[j * batch_sz : (j+1) * batch_sz, ...]
                x_dec_batch = X_dec[j * batch_sz : (j+1) * batch_sz, ...]
                ybatch = Y[j*batch_sz:(j+1) * batch_sz, ...]
                
                xbatch = [x_enc_batch, x_dec_batch]
                
                cost_i = self.update_weights(xbatch, ybatch, is_training = True)
                
                cost_vec.append(cost_i)
                
                if j  == 0:
                    initVec.append(len(cost_vec) - 1)
                    cost_test = self.cost(x_test, y_test, is_training=False)
                    cost_test_vec.append(cost_test)
                    if verbos:
                        print("Epoch: %i, j: %i, Cost: %.6f, Cost_test: %.6f" % (epoch, j, cost_i,cost_test ))
        
        if verbos:
            plt.figure(101)
            plt.plot(cost_vec, label = 'loss train')
            plt.plot(initVec,cost_test_vec, label = 'loss test ')
            plt.legend(frameon=False)
            
        return cost_vec, cost_test_vec
    
    def predict(self, x):
        y_hat = self.forward(x, is_training=False)
        return y_hat #[0,-1,0].numpy()
                    
        



# nnet = Transformer_Encoder_Decoder(num_layer_encoder = 2, num_layer_decoder = 2,
#                                          dff = 4, d_model = 10, targetDims = 1,
#                                          num_of_heads = 8, dropoutRate=0.0, 
#                                          activate_embedding = True, d_model_embed =  1)

nnet = Transformer_Encoder_Decoder_Keras(num_layer_encoder = 2, num_layer_decoder = 2,
                                         dff = 4, d_model = 10, targetDims = 1,
                                         num_of_heads = 8, dropoutRate=0.0, 
                                         activate_embedding = True, d_model_embed =  1)
# dff = 4 means we have 4*d_model hidden neurons in the feed forward layer 
learning_rate = 0.0002#CustomSchedule(d_model = 10)

model = Model(nnet = nnet, learning_rate=learning_rate)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '16' # was 12

# make the original data 

series, validation_series = getSeriesData(n = 1500)
plt.figure(22)
plt.plot(series[-100:], 'k', label = 'Train data')
plt.plot(np.arange(len(validation_series)) + len(series[-100:]),validation_series, 
         'r',label = 'Validation data')
plt.legend(frameon=False, loc = 2)
plt.ylim([-1.5,1.5])
plt.xlabel('Time')
plt.ylabel('Value')
## build the dataset for Encoder decoder Transformer  
# let's see if we can use T past values to predict the next value
T  = 5
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '12' # was 12

(X_general_train, Y_train, 
 X_test_general, Y_test, 
 X_general_validation, Y_Validation) = prepare_data_for_encoder_decoder_transformer(series,
                                                                                    validation_series, T = 5 )


cost_vec, cost_test_vec = model.train(x_train=X_general_train, y_train = Y_train, x_test=X_test_general, y_test=Y_test,batch_sz=32,
            epochs=50)


y_hat = model.predict(X_general_validation) # X_general_validation = [X_encoder, Xdecoder]

y_hat_new = y_hat[:,-1,0].numpy()

    



## creating self prediction 
X_0_encoder = np.expand_dims(X_general_validation[0][0],axis = 0).astype(np.float32)
X_0_decoder = np.expand_dims(X_general_validation[1][0], axis = 0).astype(np.float32)
y_self_prediction = []
for i in range(100):
    
    y_pred = model.predict([X_0_encoder, X_0_decoder])
    y_pred = y_pred[0,-1,0].numpy()
    x_encoder = []
    x_decoder = []
    
    for i in range(1, T):
        a = X_0_encoder[0,i,0]
        b = X_0_decoder[0,i,0]
        x_encoder.append(a)
        x_decoder.append(b)
        
    x_encoder.append(b)
    x_decoder.append(y_pred)
    X_0_encoder = np.expand_dims(np.array([x_encoder]), axis = 2)
    X_0_decoder = np.expand_dims(np.array([x_decoder]), axis = 2)
    
    y_self_prediction.append(y_pred)



plt.figure(2)
plt.plot(validation_series[6:], label = r'$Y$')
plt.plot(y_hat_new , label = r'$\hat{Y} (Single-shot)$')
plt.legend(frameon = False)
plt.xlabel('# Steps')
plt.ylabel('Value')
plt.ylim([-1.5,2.2])


plt.figure(3)
plt.plot(validation_series[6:], label = r'$Y$')
plt.plot(y_self_prediction[:-6], label = r'$\hat{Y} (Autoregressive) $')
plt.legend(frameon = False)
plt.ylim([-1.5,2.2])

plt.figure(4)
error = np.abs(y_self_prediction[:-6] - validation_series[6:])
error2 = np.abs(y_hat_new - validation_series[6:])
plt.plot(error, c = 'k')
plt.xlabel('# Steps')
plt.ylabel('|Error Value|')

#####################################
fig, axes = plt.subplots(3,1)
axes[0].plot(validation_series[6:],c ='r', label = r'$Y$')
axes[0].plot(y_hat_new , label = r'$\hat{Y} \ (Single-shot)$')
axes[0].legend(frameon = False, loc = 2)
axes[0].set_xlabel('# Steps')
axes[0].set_ylabel('Value')
axes[0].set_ylim([-1.5,4])


axes[1].plot(validation_series[6:],c ='r', label = r'$Y$')
axes[1].plot(y_self_prediction[:-6], label = r'$\hat{Y} (Autoregressive) $')
axes[1].legend(frameon = False, loc = 2)
axes[1].set_ylim([-1.5,4])
axes[1].set_xlabel('# Steps')
axes[1].set_ylabel('Value')


axes[2].plot(error2,  '--k', label = 'Single-shot')
axes[2].plot(error, c = 'k', label = 'Autoregressive')
axes[2].legend(frameon = False, loc = 2)
axes[2].set_xlabel('# Steps')
axes[2].set_ylabel('|Error Value|')
axes[2].set_ylim([-1.5,4])