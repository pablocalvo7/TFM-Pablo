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
#model_nores = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_2values_nores')
#model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_2values_12')
model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model(hyp_test)_F1_2values_12')
#model_direct = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_direct_F1_2values_nores')


filex = '/Users/user/Desktop/TFM/6. Simple functions/data/x_F1_2values_12.csv'
filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F1_2values_12.csv'
xx,F = scm.read_data(filex,filey)

Ne = 2 #Number of x values --> F_j(x1,...xN)
F1 = True
inverse_problem = True
hyp_test = True

#For the file and plot's title
if(F1):
    func_name = 'F1'

#NORMALIZATION
F_norm = scm.normalization_function_1(F,Ne)

if(inverse_problem):
    x=F_norm
    y=xx
else:
    x=xx
    y=F_norm

########## IMPORT MODELS AND DATA ##########
######################################################


######################################################
########## PREDICT AND PLOT ##########
train = True #whether we plot training data or validation data
plot_separated_values = False
plot_differences = True

if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

ntrain = 10000
nval = 5000

#PREDICTIONS
#NN_nores_y = model_nores.predict(x)
NN_12_y = model_12.predict(x)
#NN_direct_y = model_direct.predict(x)


#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
else:
    from_num = ntrain

if(plot_separated_values):
    for i in range(Ne):
        directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/2 values/'
        if(inverse_problem):
            if(hyp_test):
                file_name='(hyp_test)x'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                file_name='x'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
        else:
            if(hyp_test):
                file_name='direct(hyp_test)_F'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                file_name='direct_F'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
        Title = func_name+', '+str(Ne)+' values'

        ndata=1000
        y_desired = y[from_num:from_num+ndata,i]
        #y_NN_nores = NN_nores_y[from_num:from_num+ndata,i]
        y_NN_12 = NN_12_y[from_num:from_num+ndata,i]
        #y_NN_direct = NN_direct_y[from_num:from_num+ndata,i]

        datalist = [[y_desired,y_NN_12]]
        labellist = ['NN=desired','NN 1,2 restriction']
        if(inverse_problem):
            Axx = r'$x_'+str(i+1)+'$ (desired)'
            Axy = r'$x_'+str(i+1)+'$ (NN)'
        else:
            Axx = r'$F_'+str(i+1)+'$ (desired)'
            Axy = r'$F_'+str(i+1)+'$ (NN)'

        scm.GraphData_prediction(datalist, labellist, Title, directory+file_name, Axx, Axy)

if(plot_differences):
    for i in range(Ne):
        for j in range(i+1,Ne):
            directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/2 values/'
            if(hyp_test):
                file_name = '(hyptest)x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                file_name = 'x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            Title = func_name+', '+str(Ne)+' values'

            ndata = 1000
            y_desired = abs(y[from_num:from_num+ndata,i]-y[from_num:from_num+ndata,j])
            y_NN_nores = abs(NN_nores_y[from_num:from_num+ndata,i]-NN_nores_y[from_num:from_num+ndata,j])
            y_NN_12 = abs(NN_12_y[from_num:from_num+ndata,i]-NN_12_y[from_num:from_num+ndata,j])
            datalist = [[y_desired,y_NN_nores],[y_desired,y_NN_12]]
            labellist = ['NN=desired','NN no restricted','NN 1,2 restricted']
            Axx = r'$|x_'+str(i+1)+r' - x_'+str(j+1)+' |$ (desired)'
            Axy = r'$|x_'+str(i+1)+r' - x_'+str(j+1)+' |$ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, directory+file_name, Axx, Axy)


########## PREDICT AND PLOT ##########
######################################################