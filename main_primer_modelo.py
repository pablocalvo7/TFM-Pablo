######################################################
########## IMPORT PACKAGES ##########
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subroutines_chain_model as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## HYPERPARAMETERS ##########
nneurons = 50
nhidden = 1
epochs = 120
minib_size = 10
eta = 0.01
########## HYPERPARAMETERS ##########
######################################################


######################################################
########## DATA ##########
filex = '/Users/user/Desktop/more_data_HR/alt_6atoms/ENERGIES_6atoms_HR_invres_alt_1.csv'
filey = '/Users/user/Desktop/more_data_HR/alt_6atoms/EIGENVALUES_6atoms_HR_invres_alt_1.csv'

J=1
Ne = 6 #Number of atoms
rigid = True #Type of hamiltonian. True: rigid; False: periodic

ntrain = 10000
nvalidation = 5000
ndata = ntrain + nvalidation
energies,eigenvalues=scm.read_data(filex,filey)

#NORMALIZATION
x,y = scm.normalization(energies, eigenvalues, Ne, J, rigid)

xtr = x[0:ntrain,:]
xva = x[ntrain:ntrain + nvalidation, :]
ytr = y[0:ntrain,:]
yva = y[ntrain:ntrain + nvalidation, :]

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
    model.add(Dense(nneurons, activation='relu',kernel_initializer=kernel_init1, bias_initializer=bias_init))
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
file_name_save_NN = '/Users/user/Desktop/HR_models_more_data/Model_6atoms_HR_invres_alt_1'
model.save(file_name_save_NN)

#Save Description
file_hyper=open(file_name_save_NN+'/hyperparameters.txt',"w")

description='nneurons='+str(nneurons)+'\n'+'nhidden='+str(nhidden)+'\n'+'n_train='+str(ntrain)+'\n'+'n_val='+str(nvalidation)+'\n'+'epochs='+str(epochs)+'\n'+'mbs='+str(minib_size)+'\n'+'eta='+str(eta)+'\n'+'loss='+str(loss_function)+'\n'+'optimizer='+name_opt

file_hyper.write(description)

file_hyper.close()

# Plot the loss and save the history
file_hist = file_name_save_NN+'/history.txt'
file_graph = file_name_save_NN+'/pp.png'
n_epochs = np.arange(len(r.history['loss']))
scm.save_history(r,file_hist)

scm.GraphData_history([[n_epochs, r.history['loss']],
               [n_epochs,  r.history['val_loss']]],['r', 'b'], 
              ['Train', 'Validation'],'HR, 6 atoms, inv. restriction (alt. 1)',file_graph, Axx='Epochs', Axy='Loss')

########## NEURAL NETWORK ##########
######################################################