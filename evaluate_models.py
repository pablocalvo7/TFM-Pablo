######################################################
########## IMPORT PACKAGES ##########
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import xdivy_eager_fallback
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## FUNCTIONS ##########
def read_data(fileX, filey):
    X = pd.read_csv(fileX, header = None)
    X = np.array(X)
    y = pd.read_csv(filey, header = None)
    y = np.array(y)
    return X, y
########## FUNCTIONS ##########
######################################################


######################################################
########## IMPORT MODELS AND DATA ##########

Ne = 3 #Number of atoms
rigid = True #Type of hamiltonian. True: rigid; False: periodic

model_nores=models.load_model('/Users/user/Desktop/HR_models_more_data/Model_3atoms_HR_nores')
model_invres=models.load_model('/Users/user/Desktop/HR_models_more_data/Model_3atoms_HR_invres')


filex = '/Users/user/Desktop/more_data_HR/ENERGIES_3atoms_HR_invres.csv'
filey = '/Users/user/Desktop/more_data_HR/EIGENVALUES_3atoms_HR_invres.csv'
energies,eigenvalues = read_data(filex,filey)

#NORMALIZATION OF EIGENVALUES DATA [0,1)
if(Ne==2):  #2 atoms
        min_eig = -1.0
        max_eig = 2.0
if(Ne==3): #3 atoms
    if(rigid): #HR
        min_eig = -1.4142135623730951
        max_eig = 2.4142135623730945
    else: #HP
        min_eig = -1.0000000000000002
        max_eig = 3.0
if(Ne==4): #4 atoms
    if(rigid): #HR
        min_eig = -1.6180339887498938
        max_eig = 2.6180339887498936
    else: #HP
        min_eig = -1.9999999999999984
        max_eig = 3.0000000000000004
eigen_norm=(eigenvalues-min_eig)/(max_eig-min_eig)
x=eigen_norm; y=energies #QM

########## IMPORT MODELS AND DATA ##########
######################################################


######################################################
########## EVALUATE DATA ##########

ntrain = 10000
nval = 5000

xtrain = x[0:ntrain,:]
ytrain = y[0:ntrain,:]
xval = x[ntrain:ntrain+nval,:]
yval = y[ntrain:ntrain+nval,:]

score_train = model_invres.evaluate(xtrain, ytrain, verbose=0)
score_val = model_invres.evaluate(xval, yval, verbose=0)

print("SCORE OF TRAIN DATA: ",score_train)
print("SCORE OF VALIDATION DATA: ",score_val)

########## EVALUATE DATA ##########
######################################################