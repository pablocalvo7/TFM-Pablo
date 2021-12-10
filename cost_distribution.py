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
make_gap = False
gap = 0.1

alpha = True

#MODELS
if(make_gap):
    model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/gap/Model_F1_2values_12_gap'+str(gap))
else:
    #model_nores = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_2values_nores')
    #model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_F1_2values_12')
    model_12 = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/alpha/Model_F1_2values_2res')
    #model_direct = models.load_model('/Users/user/Desktop/TFM/6. Simple functions/models/Model_direct_F1_2values_nores')

if(make_gap):
    filex = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/x_F1_2values_12_gap'+str(gap)+'.csv'
    filey = '/Users/user/Desktop/TFM/6. Simple functions/data/gap/F_F1_2values_12_gap'+str(gap)+'.csv'
else:
    filex = '/Users/user/Desktop/TFM/6. Simple functions/data/alpha_F1_2values_12.csv'
    filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F1_2values_12.csv'

Nsamples = 20000
xx,F = scm.read_data(filex,filey)

if(alpha):
    x_new=[]
    for i in range(Nsamples):
        x_1 = xx[i,0] - (0.5*xx[i,1])
        x_2 = xx[i,0] + (0.5*xx[i,1])
        list_x = [x_1,x_2]
        x_new.append(list_x)
    
    x_new = np.array(x_new)

Ne = 2 #Number of x values --> F_j(x1,...xN)
F1 = True
inverse_problem = True
number_res = 2
hyp_test = False #wheter the model is made for testing hyperparameters or not


#For the file and plot's title
if(number_res==0):
    res_name = 'nores'
else:
    res_name = str(number_res)+'res'
if(F1):
    func_name = 'F1'

#NORMALIZATION
F_norm = scm.normalization_function_1(F,Ne)

if(inverse_problem):
    x=F_norm
    y=xx
    if(alpha):
        y_new=x_new
else:
    x=xx
    y=F_norm


########## IMPORT MODELS AND DATA ##########
######################################################


######################################################
########## EVALUATE AND PLOT ##########
train = True #whether we plot training data or validation data
nbins = 50 #number of bins for the plot
delta = 1/float(nbins) #interval x1, x2
evaluate_output = True
#for plot's file name and title
if(evaluate_output):
    ev_name = 'output'
else:
    ev_name = 'input'

if(train): #for the file's title
    data_name = 'train'
else:
    data_name = 'val'

ntrain = 17000
nval =1000

#from which position of "y" we plot (depending on train or validation plotting)
if(train):
    from_num = 0
else:
    from_num = ntrain

ndata=17000
x = x[from_num:from_num+ndata,:]
y = y[from_num:from_num+ndata,:]
y_new = y_new[from_num:from_num+ndata,:]

#DIVIDE DATA IN GROUPS (i,j)
if(Ne==2):
    #1. Create empty matrix
    data_x_groups = np.ndarray((nbins,nbins,ndata,Ne))
    data_y_groups = np.ndarray((nbins,nbins,ndata,Ne))
    number_of_data = np.zeros((nbins,nbins))
    cost_matrix = np.zeros((nbins,nbins))

    #2. Fill the matrix data_groups
    for i in range(ndata):
        if(evaluate_output):
            k=int(y[i,1]/delta)
            l=int(y[i,0]/delta)
        else:
            k=int(x[i,1]/delta)
            l=int(x[i,0]/delta)
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
    if(make_gap):
        directory = '/Users/user/Desktop/cost_distribution/gap/'
    else:
        directory = '/Users/user/Desktop/cost_distribution/alpha/'
    if(hyp_test):
        if(make_gap):
            file_name = 'cost_distribution(hyp_test)_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'_gap'+str(gap)+'.png'
        else:
            file_name = 'cost_distribution(hyp_test)_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'.png'
    else:
        if(make_gap):
            file_name = 'cost_distribution_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'_gap'+str(gap)+'.png'
        else:
            file_name = 'alpha_cost_distribution_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'.png'
    Title = 'Cost distribution, '+res_name+' restriction, '+ev_name

    fig = plt.figure(figsize=(7,7))
    plt.imshow(cost_matrix,extent=[0,1,0,1])
    plt.title("Cost distribution",fontsize=30,fontname= 'Gill Sans')
    if(evaluate_output):
        plt.xlabel(r'$\alpha_1$',fontsize=25,fontname='Gill Sans')
        plt.ylabel(r'$\alpha_2$',fontsize=25,fontname='Gill Sans')
    else:
        plt.xlabel(r'$F_1$',fontsize=25,fontname='Gill Sans')
        plt.ylabel(r'$F_2$',fontsize=25,fontname='Gill Sans')
    plt.colorbar()
    plt.savefig(directory+file_name, bbox_inches='tight')
    plt.show()

if(Ne==3):
    nbins_z = 10
    delta_z = 1/nbins_z
    #1. Create empty matrix
    data_x_groups = np.ndarray((nbins,nbins,nbins_z,ndata,Ne))
    data_y_groups = np.ndarray((nbins,nbins,nbins_z,ndata,Ne))
    number_of_data = np.zeros((nbins,nbins,nbins_z))
    cost_matrix = np.zeros((nbins,nbins,nbins_z))

    #2. Fill the matrix data_groups
    for i in range(ndata):
        if(evaluate_output):
            k=int(y[i,1]/delta)
            l=int(y[i,0]/delta)
            m=int(y[i,2]/delta_z)
        else:
            k=int(x[i,1]/delta)
            l=int(x[i,0]/delta)
            m=int(x[i,2]/delta_z)
        num=int(number_of_data[k,l,m])
        data_x_groups[k,l,m,num] = x[i,:]
        data_y_groups[k,l,m,num] = y[i,:]
        number_of_data[k,l,m]=number_of_data[k,l,m]+1

    for k in range(nbins_z):
        for i in range(nbins):
            for j in range(nbins):
                if(number_of_data[i,j,k]==0):
                    cost_matrix[i,j,k]=0
                    print("No data in element (",i,",",j,",",k,"): ")
                else:
                    num = int(number_of_data[i,j,k])
                    score = model_12.evaluate(data_x_groups[i,j,k,0:num,:],data_y_groups[i,j,k,0:num,:])
                    print("Score element (",i,",",j,"): ",score)
                    cost_matrix[i,j,k]=score

    #PLOT

    for k in range(nbins_z):
        cost_m = cost_matrix[:,:,k]
        cost_m=np.flip(cost_m,axis=0)
        if(make_gap):
            directory = '/Users/user/Desktop/cost_distribution_3values/gap/'
        else:
            directory = '/Users/user/Desktop/cost_distribution_3values/'
        if(hyp_test):
            if(make_gap):
                file_name = 'k'+str(k)+'cost_distribution(hyp_test)_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'_gap'+str(gap)+'.png'
            else:
                file_name = 'k'+str(k)+'cost_distribution(hyp_test)_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'.png'
        else:
            if(make_gap):
                file_name = 'k'+str(k)+'cost_distribution_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'_gap'+str(gap)+'.png'
            else:
                file_name = 'k'+str(k)+'cost_distribution_'+ev_name+'_'+func_name+'_'+res_name+'_'+data_name+'.png'
        Title = 'Cost distribution, '+res_name+' restriction, '+ev_name

        fig = plt.figure(figsize=(7,7))
        plt.imshow(cost_m,extent=[0,1,0,1])
        plt.title('Cost distribution, k = '+str(k),fontsize=30,fontname= 'Gill Sans')
        if(evaluate_output):
            plt.xlabel(r'$x_1$',fontsize=25,fontname='Gill Sans')
            plt.ylabel(r'$x_2$',fontsize=25,fontname='Gill Sans')
        else:
            plt.xlabel(r'$F_1$',fontsize=25,fontname='Gill Sans')
            plt.ylabel(r'$F_2$',fontsize=25,fontname='Gill Sans')
        plt.colorbar()
        plt.savefig(directory+file_name, bbox_inches='tight')
        plt.show()



########## PREDICT AND PLOT ##########
######################################################