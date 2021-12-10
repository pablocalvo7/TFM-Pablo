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
model_2res = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model(hyp_test)_F_square_1values_positive_res')


Ne = 1 #Number of x values --> F_j(x1,...xN)


#For the file and plot's title
func_name = 'F_square'
res_name = 'positive_res'

########## IMPORT MODEL ##########
######################################################


######################################################
########## PREDICT AND PLOT ##########

x_plot = []
y_plot = []
ndata = 1000
delta = 1/ndata
for i in range(10*ndata):
    x_ = i*delta
    y_ =np.sqrt(x_)
    x_plot.append(x_)
    y_plot.append(y_)

x_plot = np.array(x_plot)
y_plot = np.array(y_plot)
y_pred = model_2res.predict(x_plot)


directory = '/Users/user/Desktop/TFM/6. Simple functions/predictions/1 values/'
file_name = 'square_root.png'
Title = func_name+', '+str(Ne)+' values'


datalist = [[x_plot,y_plot],[x_plot,y_pred]]
labellist = ['desired','NN (+) restriction']
Axx = r'$x$'
Axy = r'$\sqrt{x}$'


scm.GraphData_history(datalist, ['r', 'b'] ,labellist, Title, directory+file_name, Axx, Axy)




########## PREDICT AND PLOT ##########
######################################################