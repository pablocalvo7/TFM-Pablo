######################################################
########## IMPORT PACKAGES ##########
import tensorflow as tf
from tensorflow.python.framework.ops import internal_convert_to_tensor_or_composite
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## HYPERPARAMETERS ##########
hyp_test = True #If we are testing the hyperparameters, so we can vary them freely

nneurons = 10
nhidden = 2
epochs = 80
minib_size = 10
eta = 0.01
########## HYPERPARAMETERS ##########
######################################################


######################################################
########## DATA ##########
filex = '/Users/user/Desktop/TFM/6. Simple functions/data/x_F1_2values_12.csv'
filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F1_2values_12.csv'

Ne = 2 #Number of x values --> F_j(x1,...xN)
perm_2values = True
inverse_problem = True
F1 = True

#For the file and plot's title
if(perm_2values):
    res_name = '12'
else:
    res_name = 'nores'
if(F1):
    func_name = 'F1'

ntrain = 10000
nvalidation = 5000
ndata = ntrain + nvalidation
x,F = scm.read_data(filex,filey)

#NORMALIZATION
F_norm = scm.normalization_function_1(F,Ne)

if(inverse_problem):
    xtr = F_norm[0:ntrain,:]
    xva = F_norm[ntrain:ntrain + nvalidation, :]
    ytr = x[0:ntrain,:]
    yva = x[ntrain:ntrain + nvalidation, :]
else:
    ytr = F_norm[0:ntrain,:]
    yva = F_norm[ntrain:ntrain + nvalidation, :]
    xtr = x[0:ntrain,:]
    xva = x[ntrain:ntrain + nvalidation, :]

input_neurons  = xtr.shape[1]
output_neurons = ytr.shape[1]
########## DATA ##########
######################################################


######################################################
########## NEURAL NETWORK ##########
kernel_init1 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/np.sqrt(input_neurons), seed=None)
kernel_init2 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/np.sqrt(nneurons), seed=None)
bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

model = Sequential()
model.add(Input(shape=(input_neurons,)))
for j in range(nhidden):
    model.add(Dense(nneurons, activation='sigmoid',kernel_initializer=kernel_init1, bias_initializer=bias_init))
#output layer with linear activation
model.add(Dense(output_neurons,kernel_initializer=kernel_init2, bias_initializer=bias_init))
model.summary()

# Compile and fit
name_opt='Adam'
loss_function = 'mse'
opt = tf.keras.optimizers.Adam(eta)
model.compile(optimizer=opt, loss=loss_function)

# print model layers
model.summary()
r = model.fit(xtr, ytr, batch_size = minib_size, epochs=epochs,
              validation_data=(xva, yva))
score_va = model.evaluate(xva, yva, verbose=0)
score_tr = model.evaluate(xtr, ytr, verbose=0)

#Save the model
directory = '/Users/user/Desktop/TFM/6. Simple functions/models/'
if(inverse_problem):
    if(hyp_test):
        file_name = 'Model(hyp_test)_'+func_name+'_'+str(Ne)+'values_'+res_name
    else:
        file_name = 'Model_'+func_name+'_'+str(Ne)+'values_'+res_name
else:
    if(hyp_test):
        file_name = 'Model(hyp_test)_direct_'+func_name+'_'+str(Ne)+'values_'+res_name
    else:
        file_name = 'Model_direct_'+func_name+'_'+str(Ne)+'values_'+res_name
model.save(directory+file_name)

#Save Description
file_hyper=open(directory+file_name+'/hyperparameters.txt',"w")

description='nneurons='+str(nneurons)+'\n'+'nhidden='+str(nhidden)+'\n'+'n_train='+str(ntrain)+'\n'+'n_val='+str(nvalidation)+'\n'+'epochs='+str(epochs)+'\n'+'mbs='+str(minib_size)+'\n'+'eta='+str(eta)+'\n'+'loss='+str(loss_function)+'\n'+'optimizer='+name_opt

file_hyper.write(description)

file_hyper.close()

# Plot the loss and save the history
file_hist = directory+file_name+'/history.txt'
file_graph = directory+file_name+'/pp.png'
n_epochs = np.arange(len(r.history['loss']))
scm.save_history(r,file_hist)

scm.GraphData_history([[n_epochs, r.history['loss']],
               [n_epochs,  r.history['val_loss']]],['r', 'b'], 
              ['Train', 'Validation'],func_name+', '+str(Ne)+' atoms, restriction: '+res_name,file_graph, Axx='Epochs', Axy='Loss')

########## NEURAL NETWORK ##########
######################################################