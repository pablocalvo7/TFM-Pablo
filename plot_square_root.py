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
########## IMPORT MODEL ##########

#MODELS
model_sym = models.load_model('/Users/user/Desktop/model_symmetry_loss/Model_F_square_1values_nores',
custom_objects={'symmetry_loss_square' : scm.symmetry_loss_square})


Ne = 1 #Number of x values --> F_j(x1,...xN)
inverse_model = True

#For the file and plot's title
func_name = 'F_square'
res_name = 'nores'

########## IMPORT MODEL ##########
######################################################


######################################################
########## PREDICT AND PLOT ##########

x_plot = []
y_plot = []
ndata = 1000
delta = 1/ndata
for i in range(ndata):
    x_ = i*delta
    if(inverse_model):
        y_ =np.sqrt(x_)
    else:
        y_=x_ * x_
    x_plot.append(x_)
    y_plot.append(y_)

x_plot = np.array(x_plot)
y_plot = np.array(y_plot)
y_pred = model_sym.predict(x_plot)


directory = '/Users/user/Desktop/model_symmetry_loss/plots/'
if(inverse_model):
    file_name = 'Fsquare.png'
else:
    file_name = 'square.png'
Title = func_name+', '+str(Ne)+' values'


datalist = [[x_plot,y_plot],[x_plot,y_pred]]
labellist = ['desired','NN sym_cost']
Axx = r'$x^2$'
if(inverse_model):
    Axy = r'$x$'


scm.GraphData_history(datalist, ['r', 'b'] ,labellist, Title, directory+file_name, Axx, Axy)




########## PREDICT AND PLOT ##########
######################################################