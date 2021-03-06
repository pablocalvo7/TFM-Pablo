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
alpha = False
make_gap = False
gap = 0.01

#MODELS
if(make_gap):
    model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/gap/Model_F1_3values_3res_gap'+str(gap))
else:
    #model_nores = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_3values_nores')
    model_2res = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model(hyp_test)_direct_F_square_root_1values_positive_res')
    #model_3res = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_3values_3res')
    #model_direct = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_direct_F1_2values_nores')


if(make_gap):
    filex = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/x_F1_3values_3res_gap'+str(gap)+'.csv'
    filey = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/F_F1_3values_3res_gap'+str(gap)+'.csv'
else:
    filex = '/Users/user/Desktop/TFM/6. Simple functions/data/x_F_square_1values_positive_res.csv'
    filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F_square_1values_positive_res.csv'

Nsamples = 20000
xx,F = scm.read_data(filex,filey)

Ne = 1 #Number of x values --> F_j(x1,...xN)
F1 = False
inverse_problem = True
hyp_test = True #wheter the model is made for testing hyperparameters or not

#For the file and plot's title
if(F1):
    func_name = 'F1'

func_name = 'F_square_uniform_values'
res_name = 'positive_res'

#NORMALIZATION
#F_norm = scm.normalization_function_1(F,Ne)
#xx_norm = scm.normalization_xx_range_test(xx)
F_norm = F


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
plot_separated_values = True
plot_differences = False
plot_sum = False

if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

ntrain = 10000
nval = 3000

#PREDICTIONS
#NN_nores_y = model_nores.predict(x)
NN_2res_y = model_2res.predict(x)
#NN_3res_y = model_3res.predict(x)


#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
else:
    from_num = ntrain



if(plot_separated_values):
    for i in range(Ne):
        if(make_gap):
            directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/2 values/gap/'
        else:
            directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/1 values/'
        if(inverse_problem):
            if(hyp_test):
                if(make_gap):
                    file_name='(hyp_test)x'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'_gap'+str(gap)+'.png'
                else:
                    file_name='(hyp_test)x'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                if(make_gap):
                    file_name='x'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'_gap'+str(gap)+'.png'
                else:
                    if(alpha):
                        file_name='alpha'+str(i+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
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
        y_NN_2res = NN_2res_y[from_num:from_num+ndata,i]
        #y_NN_3res = NN_3res_y[from_num:from_num+ndata,i]

        datalist = [[y_desired,y_NN_2res]]
        labellist = ['NN=desired','NN (+) restriction']
        if(inverse_problem):
            if(alpha):
                Axx = r'$\alpha_'+str(i+1)+'$ (desired)'
                Axy = r'$\alpha_'+str(i+1)+'$ (NN)'
            else:
                Axx = r'$x_'+str(i+1)+'$ (desired)'
                Axy = r'$x_'+str(i+1)+'$ (NN)'
        else:
            Axx = r'$F_'+str(i+1)+'$ (desired)'
            Axy = r'$F_'+str(i+1)+'$ (NN)'

        scm.GraphData_prediction(datalist, labellist, Title, directory+file_name, Axx, Axy)

if(plot_differences):
    for i in range(Ne):
        for j in range(i+1,Ne):
            if(make_gap):
                directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/3 values/gap/'
            else:
                directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/2 values/'
            if(hyp_test):
                if(make_gap):
                    file_name = '(hyptest)x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'_gap'+str(gap)+'.png'
                else:
                    file_name = '(hyptest)x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                if(make_gap):
                    file_name = 'x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'_gap'+str(gap)+'.png'
                else:
                    file_name = 'x'+str(i+1)+'-x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            Title = func_name+', '+str(Ne)+' values'

            ndata = 17000
            y_desired = abs(y[from_num:from_num+ndata,i]-y[from_num:from_num+ndata,j])
            #y_NN_nores = abs(NN_nores_y[from_num:from_num+ndata,i]-NN_nores_y[from_num:from_num+ndata,j])
            y_NN_2res = abs(NN_2res_y[from_num:from_num+ndata,i]-NN_2res_y[from_num:from_num+ndata,j])
            #y_NN_3res = abs(NN_3res_y[from_num:from_num+ndata,i]-NN_3res_y[from_num:from_num+ndata,j])

            datalist = [[y_desired,y_NN_2res]]
            labellist = ['NN=desired','NN 2 restricted']
            Axx = r'$|x_'+str(i+1)+r' - x_'+str(j+1)+' |$ (desired)'
            Axy = r'$|x_'+str(i+1)+r' - x_'+str(j+1)+' |$ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, directory+file_name, Axx, Axy)

if(plot_sum):
    for i in range(Ne):
        for j in range(i+1,Ne):
            directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/2 values/'
            if(hyp_test):
                file_name = '(hyptest)x'+str(i+1)+'+x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            else:
                file_name = 'x'+str(i+1)+'+x'+str(j+1)+'_'+func_name+'_'+str(Ne)+'values_'+data_name+'.png'
            Title = func_name+', '+str(Ne)+' values'

            ndata = 1000
            y_desired = y[from_num:from_num+ndata,i]+y[from_num:from_num+ndata,j]
            #y_NN_nores = NN_nores_y[from_num:from_num+ndata,i]+NN_nores_y[from_num:from_num+ndata,j]
            y_NN_12 = NN_12_y[from_num:from_num+ndata,i]+NN_12_y[from_num:from_num+ndata,j]
            datalist = [[y_desired,y_NN_12]]
            labellist = ['NN=desired','NN no restricted','NN 1,2 restricted']
            Axx = r'$x_'+str(i+1)+r' + x_'+str(j+1)+' $ (desired)'
            Axy = r'$x_'+str(i+1)+r' + x_'+str(j+1)+' $ (NN)'

            scm.GraphData_prediction(datalist, labellist, Title, directory+file_name, Axx, Axy)


########## PREDICT AND PLOT ##########
######################################################