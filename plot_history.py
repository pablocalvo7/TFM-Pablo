######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################



F1 = True
Ne = 5 #Number of values
if(F1): #For the plot's title
    func_name = 'F1'

hist_6 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_sweep_5/history.txt')
hist_5 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_sweep_4/history.txt')
hist_4 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_sweep_3/history.txt')
hist_3 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_sweep_2/history.txt')
hist_2 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_sweep_1/history.txt')
hist_1 = np.loadtxt('/Users/user/Desktop/barrer_datos_2/models/(grid)Model_5atoms_HP_nores/history.txt')


epochs=hist_1[:,0]
#val_nores=hist_1[:,1]
train_1=hist_1[:,2]
#val_invres=hist_invres[:,1]
train_2=hist_2[:,2]
#val_res1=hist_res1[:,1]
train_3=hist_3[:,2]
train_4=hist_4[:,2]
train_5=hist_5[:,2]
train_6=hist_6[:,2]


directory= '/Users/user/Desktop/barrer_datos_2/plots/'
file_name = 'compare.png'

Title = 'Compare symmetry breaking'
sweep_1 = r'$\sigma_1$'
sweep_2 = r'$\sigma_1 , \sigma_3 $'
sweep_3 = r'$\sigma_1 , \sigma_3 , \sigma_4 , C_5 , C_5^4 $'
sweep_4 = r'$\sigma_1 , \sigma_3 , \sigma_4 , \sigma_5 , C_5 , C_5^4 $'
sweep_5 = r'$\sigma_1 , \sigma_2 , \sigma_3 , \sigma_4 , \sigma_5 , C_5 , C_5^2 , C_5^3 , C_5^4 $'
no_res ='No symmetry breaking'

dat = [[epochs, train_1],[epochs, train_2],[epochs,train_3],[epochs,train_4],[epochs,train_5],[epochs,train_6]]

scm.GraphData_history(dat,['r', 'b','k','lime','g','c'], ['','','','','',''],Title,directory+file_name, Axx='Epochs',Axy='Cost')