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
import subroutines_chain_model as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## IMPORT MODELS AND DATA ##########

#MODELS
#model_nores=models.load_model('/Users/user/Desktop/HP_models_more_data/Model_3atoms_HR_nores')
#model_invres=models.load_model('/Users/user/Desktop/HP_models_more_data/Model_3atoms_HR_invres')
#model_res1=models.load_model('/Users/user/Desktop/HP_models_more_data/Model_3atoms_HP_cycres')
model_res2=models.load_model('/Users/user/Desktop/HP_models_more_data/inv_plus_cyc/Model_4atoms_HP_allres')

filex = '/Users/user/Desktop/more_data_HP/inv_plus_cyc/ENERGIES_4atoms_HP_allres.csv'
filey = '/Users/user/Desktop/more_data_HP/inv_plus_cyc/EIGENVALUES_4atoms_HP_allres.csv'
energies,eigenvalues = scm.read_data(filex,filey)

J=1
rigid = False #Type of hamiltonian. True: rigid; False: periodic
Ne = 4 #Number of atoms
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

#NORMALIZATION
x,y = scm.normalization(energies, eigenvalues, Ne, J, rigid)

########## IMPORT MODELS AND DATA ##########
######################################################


######################################################
########## PREDICT AND PLOT ##########
train = False #whether we plot training data or validation data
plot_separated_energies = True
plot_differences = False
if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

ntrain = 10000
nval = 5000

#PREDICTIONS
#NN_nores_y = model_nores.predict(x)
#NN_invres_y = model_invres.predict(x)
#NN_res1_y = model_res1.predict(x)
NN_res2_y = model_res2.predict(x)

#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
else:
    from_num = ntrain

if(plot_separated_energies):
    for i in range(Ne):

        filename_data='/Users/user/Desktop/prediction_HP_more_data/inv_plus_cyc/energy'+str(i+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'.png'
        Title = H_name+', '+str(Ne)+' atoms'

        ndata=1000
        y_QM = y[from_num:from_num+ndata,i]
        y_NN = NN_res2_y[from_num:from_num+ndata,i]

        datalist = [[y_QM,y_NN]]
        labellist = ['NN=QM','NN all restricted']
        Axx = r'$\varepsilon_'+str(i+1)+'$ (QM)'
        Axy = r'$\varepsilon_'+str(i+1)+'$ (NN)'

        scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)

if(plot_differences):
    for i in range(Ne):
        for j in range(i+1,Ne):
            filename_data='/Users/user/Desktop/3 atoms/energy'+str(i+1)+'-energy'+str(j+1)+'_'+H_name+'_'+str(Ne)+'atoms_'+data_name+'.png'
            Title = H_name+', '+str(Ne)+' atoms'

            ndata = 1000
            y_QM = abs(y[from_num:from_num+ndata,i]-y[from_num:from_num+ndata,j])
            y_NN = abs(NN_invres_y[from_num:from_num+ndata,i]-NN_invres_y[from_num:from_num+ndata,j])
            datalist = [[y_QM,y_NN]]
            labellist = ['NN=QM','NN inv. restricted']
            Axx = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (QM)'
            Axy = r'$|\varepsilon_'+str(i+1)+r' - \varepsilon_'+str(j+1)+' |$ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, filename_data, Axx, Axy)


########## PREDICT AND PLOT ##########
######################################################