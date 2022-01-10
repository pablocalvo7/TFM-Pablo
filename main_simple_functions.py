######################################################
########## IMPORT PACKAGES ##########
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.python.framework.ops import internal_convert_to_tensor_or_composite
from tensorflow.python.keras.saving.save import load_model
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, LogCosh
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subroutines as scm
import math
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## HYPERPARAMETERS ##########
hyp_test = False #If we are testing the hyperparameters, so we can vary them freely

nneurons = 30
nhidden = 1
epochs = 20
minib_size = 10
eta = 0.01
########## HYPERPARAMETERS ##########
######################################################


######################################################
########## DATA ##########
make_gap = False
gap = 0.1
plot_prediction = True

if(make_gap):
    filex = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/x_F1_3values_3res_gap'+str(gap)+'.csv'
    filey = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/F_F1_3values_3res_gap'+str(gap)+'.csv'
else:
    filex = '/Users/user/Desktop/models_symmetry_loss/data/x_F_square_1values_nores.csv'
    filey = '/Users/user/Desktop/models_symmetry_loss/data/F_F_square_1values_nores.csv'

Ne = 1 #Number of x values --> F_j(x1,...xN)
number_res = 0 #2,3,4,..., number of sorted values starting from the beginning
inverse_problem = True
F1 = False

#For the file and plot's title
if(number_res==0):
    res_name = 'nores'
else:
    res_name = str(number_res)+'res'
if(F1):
    func_name = 'F1'

func_name = 'F_square_sinwe'
res_name = 'nores'

ntrain = 15000
nvalidation = 5000
ndata = ntrain + nvalidation
x,F = scm.read_data(filex,filey)

#NORMALIZATION
#F_norm = scm.normalization_function_1(F,Ne)
#x_norm = scm.normalization_xx_range_test(x)
F_norm = F

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
initialize_weights_normal = False
model_without_train = True
load_other_weights = True

if(initialize_weights_normal):
    kernel_init1 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/np.sqrt(input_neurons), seed=None)
    kernel_init2 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/np.sqrt(nneurons), seed=None)
    bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

#model = Sequential()
#model.add(Input(shape=(input_neurons,)))
#for j in range(nhidden):
 #   model.add(Dense(nneurons, activation='sigmoid',kernel_initializer=kernel_init1, bias_initializer=bias_init))
#output layer with linear activation
#model.add(Dense(output_neurons,kernel_initializer=kernel_init2, bias_initializer=bias_init))
#model.summary()
old_model_path = '/Users/user/Desktop/models_symmetry_loss/Model_direct_F_sin2pix_1values_nores'
model = models.load_model(old_model_path)
model_untrained = models.load_model(old_model_path)

#if(model_without_train):
    #model_untrained = Sequential()
   # model_untrained.add(Input(shape=(input_neurons,)))
    #for j in range(nhidden):
     #   model_untrained.add(Dense(nneurons, activation='sigmoid',kernel_initializer=kernel_init1, bias_initializer=bias_init))
    #output layer with linear activation
    #model_untrained.add(Dense(output_neurons,kernel_initializer=kernel_init2, bias_initializer=bias_init))
   # model_untrained.summary()


# Compile and fit
name_opt='Adam'
loss_function = 'mse_sym'
opt = tf.keras.optimizers.Adam(eta)
model.compile(optimizer=opt, loss=scm.symmetry_loss_square)

# print model layers
model.summary()
r = model.fit(xtr, ytr, batch_size = minib_size, epochs=epochs,
              validation_data=(xva, yva))
#score_va = model.evaluate(xva, yva, verbose=0)
#score_tr = model.evaluate(xtr, ytr, verbose=0)

#Save the model
if(make_gap):
    directory = '/Users/user/Desktop/TFM/6. Simple functions/models/gap/'
else:
    directory = '/Users/user/Desktop/models_symmetry_loss/'
if(inverse_problem):
    if(hyp_test):
        if(make_gap):
            file_name = 'Model(hyp_test)_'+func_name+'_'+str(Ne)+'values_'+res_name+'_gap'+str(gap)
        else:
            file_name = 'Model(hyp_test)_'+func_name+'_'+str(Ne)+'values_'+res_name
    else:
        if(make_gap):
            file_name = 'Model_'+func_name+'_'+str(Ne)+'values_'+res_name+'_gap'+str(gap)
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

Title = func_name+', '+str(Ne)+' values, restriction: '+res_name
scm.GraphData_history([[n_epochs, r.history['loss']],
               [n_epochs,  r.history['val_loss']]],['r', 'b'], 
              ['Train', 'Validation'],Title ,file_graph, Axx='Epochs', Axy='Loss')


########## NEURAL NETWORK ##########
######################################################


######################################################
########## PLOT PREDICTION ##########
if(plot_prediction):
    
    x_plot = []
    y_plot = []
    ndata = 1000
    delta = 1/ndata
    for i in range(ndata):
        x_ = i*delta
        y_ = math.sqrt(x_)
        x_plot.append(x_)
        y_plot.append(y_)

    x_plot = np.array(x_plot)
    y_plot = np.array(y_plot)
    y_pred = model.predict(x_plot)
    if(model_without_train):
        y_pred_untrained = model_untrained.predict(x_plot)


    directory = '/Users/user/Desktop/models_symmetry_loss/plots/'
    file_name = 'F_square_sin_weigths.png'
    Title = func_name+', '+str(Ne)+' values'

    if(model_without_train):
        datalist = [[x_plot,y_plot],[x_plot,y_pred],[x_plot,y_pred_untrained]]
        labellist = ['desired','NN sym_cost','NN sym_cost untrained']
        Axx = r'$x^2$'
        Axy = r'$x$'
        scm.GraphData_history(datalist, ['r', 'b', 'k'] ,labellist, Title, directory+file_name, Axx, Axy)
    else:
        datalist = [[x_plot,y_plot],[x_plot,y_pred]]
        labellist = ['desired','NN']
        Axx = r'$x$'
        Axy = r'$f(x)$'
        scm.GraphData_history(datalist, ['r', 'b'] ,labellist, Title, directory+file_name, Axx, Axy) 


########## PLOT PREDICTION ##########
######################################################
