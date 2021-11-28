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
model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_2values_nores')
#model_direct = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_direct_F1_2values_nores')


filex = '/Users/user/Desktop/TFM/6. Simple functions/data/x_F1_2values_nores.csv'
filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F1_2values_nores.csv'
xx,F = scm.read_data(filex,filey)

Ne = 2 #Number of x values --> F_j(x1,...xN)
F1 = True
inverse_problem = True
perm_2values = False
hyp_test = False #wheter the model is made for testing hyperparameters or not

#For the file and plot's title
if(perm_2values):
    res_name = '12'
else:
    res_name = 'nores'
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
########## EVALUATE AND PLOT ##########
train = True #whether we plot training data or validation data
nbins = 40 #number of bins for the plot
delta = 1/float(nbins) #interval x1, x2

if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

ntrain = 5000
nval =1000

#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
else:
    from_num = ntrain

ndata=5000
x = x[from_num:from_num+ndata,:]
y = y[from_num:from_num+ndata,:]

#DIVIDE DATA IN GROUPS (i,j)

#1. Create empty matrix
data_x_groups = np.ndarray((nbins,nbins,ndata,Ne))
data_y_groups = np.ndarray((nbins,nbins,ndata,Ne))
number_of_data = np.zeros((nbins,nbins))
cost_matrix = np.zeros((nbins,nbins))

#2. Fill the matrix data_groups
for i in range(ndata):
    k=int(y[i,1]/delta)
    l=int(y[i,0]/delta)
    num=int(number_of_data[k,l])
    data_x_groups[k,l,num] = x[i,:]
    data_y_groups[k,l,num] = y[i,:]
    number_of_data[k,l]=number_of_data[k,l]+1

print("WE PRINT GROUPS OF DATA:")
for i in range(nbins):
    for j in range(nbins):
        print("------> i,j = ",i,j)
        num=int(number_of_data[i,j])
        print(data_y_groups[i,j,0:num])

for i in range(nbins):
    for j in range(nbins):
        if(number_of_data[i,j]==0):
            cost_matrix[i,j]=0
            print("No data in element (",i,",",j,"): ")
        else:
            num = int(number_of_data[i,j])
            score = model_12.evaluate(data_x_groups[i,j,0:num,:],data_y_groups[i,j,0:num,:])
            print("Score element (",i,",",j,"): ",score)
            cost_matrix[i,j]=score

#PLOT
cost_matrix=np.flip(cost_matrix,axis=0)
directory = '/Users/user/Desktop/'
file_name = 'cost_distribution_'+func_name+'_'+res_name+'_'+data_name+'.png'
Title = "Cost distribution, "+res_name+" restriction"

fig = plt.figure(figsize=(7,7))
plt.imshow(cost_matrix,extent=[0,1,0,1])
plt.title("Cost distribution",fontsize=30,fontname= 'Gill Sans')
plt.xlabel(r'$x_1$',fontsize=25,fontname='Gill Sans')
plt.ylabel(r'$x_2$',fontsize=25,fontname='Gill Sans')
plt.colorbar()
plt.savefig(directory+file_name, bbox_inches='tight')
plt.show()


########## PREDICT AND PLOT ##########
######################################################