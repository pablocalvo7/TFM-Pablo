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
nneurons = 50
nhidden = 1
epochs = 80
minib_size = 10
eta = 0.01
########## HYPERPARAMETERS ##########
######################################################


######################################################
########## DATA ##########
grid = True

if(grid):
    filex = '/Users/user/Desktop/HP_symmetry_loss/data/(grid)ENERGIES_2atoms_HP_invres.csv'
    filey = '/Users/user/Desktop/HP_symmetry_loss/data/(grid)EIGENVALUES_2atoms_HP_invres.csv'

    filex_val = '/Users/user/Desktop/HP_symmetry_loss/data/ENERGIES_2atoms_HP_invres.csv'
    filey_val = '/Users/user/Desktop/HP_symmetry_loss/data/EIGENVALUES_2atoms_HP_invres.csv'
else:
    filex = '/Users/user/Desktop/barrer_datos_2/data/ENERGIES_5atoms_HP_sweep_5.csv'
    filey = '/Users/user/Desktop/barrer_datos_2/data/EIGENVALUES_5atoms_HP_sweep_5.csv'

J=1 #nearest neighbours hopping term
Ne = 2 #Number of atoms
rigid = False #Type of hamiltonian. True: rigid; False: periodic
inverse_problem = True

#for the title
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

#For the file's title
res_name = 'invres'

ntrain = 40000
nvalidation = 10000
ndata = ntrain + nvalidation
energies,eigenvalues=scm.read_data(filex,filey)
if(grid):
    energies_val,eigenvalues_val = scm.read_data(filex_val,filey_val)

#NORMALIZATION
if(grid):
    xtr,ytr = scm.normalization(energies, eigenvalues, Ne, J, rigid)
    xva,yva = scm.normalization(energies_val, eigenvalues_val, Ne, J, rigid)
else:
    x,y = scm.normalization(energies, eigenvalues, Ne, J, rigid)
    if(inverse_problem):
        xtr = x[0:ntrain,:]
        xva = x[ntrain:ntrain + nvalidation, :]
        ytr = y[0:ntrain,:]
        yva = y[ntrain:ntrain + nvalidation, :]
    else:
        ytr = x[0:ntrain,:]
        yva = x[ntrain:ntrain + nvalidation, :]
        xtr = y[0:ntrain,:]
        xva = y[ntrain:ntrain + nvalidation, :]

input_neurons  = xtr.shape[1]
output_neurons = ytr.shape[1]

print("shape ytr: ",np.shape(ytr))
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
#score_va = model.evaluate(xva, yva, verbose=0)
#score_tr = model.evaluate(xtr, ytr, verbose=0)

#Save the model
directory = '/Users/user/Desktop/HP_symmetry_loss/models/'
if(inverse_problem):
    file_name = 'Model_'+str(Ne)+'atoms_'+H_name+'_'+res_name
else:
    file_name = 'Model_direct_'+str(Ne)+'atoms_'+H_name+'_'+res_name
if(grid):
    file_name = '(grid)Model_'+str(Ne)+'atoms_'+H_name+'_'+res_name

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

Title = H_name+', '+str(Ne)+' atoms, '+res_name
scm.GraphData_history([[n_epochs, r.history['loss']],
               [n_epochs,  r.history['val_loss']]],['r', 'b'], 
              ['Train', 'Validation'],Title,file_graph, Axx='Epochs', Axy='Loss')

########## NEURAL NETWORK ##########
######################################################


######################################################
########## PLOT PREDICTIONS ##########
plot_predictions = False

if(plot_predictions):
    train = True #whether we plot training data or validation data
    plot_separated_energies = True
    plot_differences = True
    if(train): #for the file's title
        data_name = 'train'
    else:
        data_name = 'val'


    #from which position of "y" we plot (depending on train or validation plotting)

    if(train):
        des_y=ytr
        NN_y = model.predict(xtr)
    else:
        des_y=yva
        NN_y = model.predict(xva)

    if(plot_separated_energies):
        for i in range(Ne):

            filename_data='/Users/user/Desktop/HP_symmetry_loss/predictions/energy'+str(i+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'_'+res_name+'.png'
            Title = H_name+', '+str(Ne)+' atoms'

            y_QM = des_y[:,i]
            y_NN = NN_y[:,i]

            datalist = [[y_QM,y_NN]]
            labellist = ['NN=QM','NN']
            Axx = r'$\varepsilon_'+str(i+1)+'$ (QM)'
            Axy = r'$\varepsilon_'+str(i+1)+'$ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)

    if(plot_differences):
        for i in range(Ne):
            for j in range(i+1,Ne):
                filename_data='/Users/user/Desktop/HP_symmetry_loss/predictions/energy'+str(i+1)+'-energy'+str(j+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'_'+res_name+'.png'
                Title = H_name+', '+str(Ne)+' atoms'

                y_QM = abs(des_y[:,i]-des_y[:,j])
                y_NN = abs(NN_y[:,i]-NN_y[:,j])
                datalist = [[y_QM,y_NN]]
                labellist = ['NN=QM','NN']
                Axx = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (QM)'
                Axy = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (NN)'

                scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)

########## PLOT PREDICTIONS ##########
######################################################