######################################################
########## IMPORT PACKAGES ##########
from numpy.lib.scimath import sqrt
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## IMPORT MODELS AND DATA ##########

#MODELS
#model_nores=models.load_model('/Users/user/Desktop/HP_models_more_data/Model_3atoms_HR_nores')
model=models.load_model('/Users/user/Desktop/HP_symmetry_loss/models/(grid)Model_2atoms_HP_nores_sym',
custom_objects={'symmetry_loss':scm.symmetry_loss})
#model_res1=models.load_model('/Users/user/Desktop/HP_models_more_data/Model_3atoms_HP_cycres')
#model_res2=models.load_model('/Users/user/Desktop/HP_models_more_data/inv_plus_cyc/Model_4atoms_HP_allres')

filex = '/Users/user/Desktop/HP_symmetry_loss/data/(grid)ENERGIES_2atoms_HP_nores.csv'
filey = '/Users/user/Desktop/HP_symmetry_loss/data/(grid)EIGENVALUES_2atoms_HP_nores.csv'
energies,eigenvalues = scm.read_data(filex,filey)

J=1
rigid = True #Type of hamiltonian. True: rigid; False: periodic
Ne = 2 #Number of atoms
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

ntrain = 40000
nval = 10000

#NORMALIZATION
x,y = scm.normalization(energies, eigenvalues, Ne, J, rigid)

y_res = []
for i in range(ntrain):
    if(y[i,0]<y[i,1]):
        new = [y[i,1],y[i,0]]
    else:
        new = y[i,:]
    y_res.append(new)

y_res = np.array(y_res)


########## IMPORT MODELS AND DATA ##########
######################################################


######################################################
########## PREDICT AND PLOT ##########
train = True #whether we plot training data or validation data
plot_separated_energies = True
plot_differences = True
if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

res_name = 'nores_sym'


#PREDICTIONS
#NN_nores_y = model_nores.predict(x)
NN_y = model.predict(x)
#NN_res1_y = model_res1.predict(x)
#NN_res2_y = model_res2.predict(x)

#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
    ndata = ntrain
else:
    from_num = ntrain
    ndata = nval

if(plot_separated_energies):
    for i in range(Ne):

        filename_data='/Users/user/Desktop/HP_symmetry_loss/predictions/energy'+str(i+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'_'+res_name+'.png'
        Title = H_name+', '+str(Ne)+' atoms'

        ndata=1000
        y_QM = y_res[from_num:from_num+ndata,i]
        y_NN = NN_y[from_num:from_num+ndata,i]

        datalist = [[y_QM,y_NN]]
        labellist = ['NN=QM','NN inv. restricted']
        Axx = r'$\varepsilon_'+str(i+1)+'$ (QM)'
        Axy = r'$\varepsilon_'+str(i+1)+'$ (NN)'

        scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)

if(plot_differences):
    for i in range(Ne):
        for j in range(i+1,Ne):
            filename_data='/Users/user/Desktop/HP_symmetry_loss/predictions/energy'+str(i+1)+'-energy'+str(j+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'_'+res_name+'.png'
            Title = H_name+', '+str(Ne)+' atoms'

            y_QM = abs(y_res[from_num:from_num+ndata,i]-y_res[from_num:from_num+ndata,j])
            y_NN = abs(NN_y[from_num:from_num+ndata,i]-NN_y[from_num:from_num+ndata,j])
            datalist = [[y_QM,y_NN]]
            labellist = ['NN=QM','NN inv. restricted']
            Axx = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (QM)'
            Axy = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)


########## PREDICT AND PLOT ##########
######################################################