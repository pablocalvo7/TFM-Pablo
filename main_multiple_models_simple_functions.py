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
import subroutines as scm

########## IMPORT PACKAGES ##########
######################################################


######################################################
########## HYPERPARAMETERS (FIXED OR LIST) ##########
nneurons = 25
#nhidden = 1
epochs = 100
minib_size = 10
eta = 0.01

nhidden_list = [1,2,3,4]
#nneurons_list=[5,10,15,20,25,30,35,40,45,50]
#eta_list = [0.0001,0.001,0.01,0.1,1]
########## HYPERPARAMETERS (FIXED OR LIST) ##########
######################################################


######################################################
########## DATA ##########
filex = '/Users/user/Desktop/TFM/6. Simple functions/data/range_test/x_F1_2values_12.csv'
filey = '/Users/user/Desktop/TFM/6. Simple functions/data/range_test/F_F1_2values_12.csv'

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

#fixed or list
ntrain = 17000
nvalidation = 1000
#ntrain_list = [1000,3000,5000,7000,9000,11000,13000,15000,17000,19000]
ndata = ntrain + nvalidation
x,F = scm.read_data(filex,filey)

#NORMALIZATION
F_norm = scm.normalization_function_1(F,Ne)

if(inverse_problem):
    input_neurons  = F_norm.shape[1]
    output_neurons = x.shape[1]
else:
    input_neurons  = x.shape[1]
    output_neurons = F_norm.shape[1]
########## DATA ##########
######################################################

######################################################
########## MODEL COMPLEXITY ##########
param_free=nhidden_list
param_free_current=[]
results_train=[]
results_val=[]
Nparams=1
if(inverse_problem):
    main_folder = 'Model_'+func_name+'_'+str(Ne)+'values_'+res_name+'_multiple/'
else:
    main_folder = 'Model_direct_'+func_name+'_'+str(Ne)+'values_'+res_name+'_multiple/'
carpeta = 'nhidden/'
axis_name = 'Hidden layers'
########## MODEL COMPLEXITY ##########
######################################################


######################################################
########## NEURAL NETWORKS ##########
for nhidden in param_free:
    
    free=nhidden # For the file name
    param_free_current.append(free)

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

    # Compile
    name_opt='Adam'
    loss_function = 'mse'
    opt = tf.keras.optimizers.Adam(eta)
    model.compile(optimizer=opt, loss=loss_function)
    model.summary()

    #EarlyStopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

    #Fit
    r = model.fit(xtr, ytr, batch_size = minib_size, epochs=epochs,
                validation_data=(xva, yva))
    score_va = model.evaluate(xva, yva, verbose=0)
    score_tr = model.evaluate(xtr, ytr, verbose=0)
    ep_ES=len(r.history['val_loss'])

    #Save the model
    directory = '/Users/user/Desktop/TFM/6. Simple functions/models/'
    file_name = main_folder+carpeta+str(free)
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

    Title = func_name+', '+str(Ne)+' atoms, restriction: '+res_name
    scm.GraphData_history([[n_epochs, r.history['loss']],
                [n_epochs,  r.history['val_loss']]],['r', 'b'], 
                ['Train', 'Validation'],Title,file_graph, Axx='Epochs', Axy='Loss')

    #Update model complexity curve
    
    results_train.append(score_tr)
    results_val.append(score_va)

    results_train_array=np.array(results_train)
    results_train_array=results_train_array.reshape((Nparams,1))
    results_val_array=np.array(results_val)
    results_val_array=results_val_array.reshape((Nparams,1))
    

    graph_MC = directory+main_folder+carpeta+'/MC.png'
    text_MC = directory+main_folder+carpeta+'/MC_data.txt'
    scm.GraphData_history([[param_free_current, results_train_array[:,0]],
                   [param_free_current,  results_val_array[:,0]]],['r', 'b'], 
                  ['Train', 'Validation'],'Model Complexity',graph_MC, Axx=axis_name, Axy='Cost')
    
    f = open(text_MC,"w")
    for j in range(Nparams): 
        row= str(param_free_current[j])+' '+str(results_val_array[j,0])+' '+' '+str(results_train_array[j,0])+' '+''+'\n'
        f.write(row)
        
    f.close()
    
    Nparams=Nparams+1

########## NEURAL NETWORKS ##########
######################################################