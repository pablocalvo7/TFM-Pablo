######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################



F1 = True
Ne = 2 #Number of values
if(F1): #For the plot's title
    func_name = 'HP'

hist_sym = np.loadtxt('/Users/user/Desktop/HP_symmetry_loss/models/(grid)Model_2atoms_HP_nores_sym/history.txt')
hist_res = np.loadtxt('/Users/user/Desktop/HP_symmetry_loss/models/(grid)Model_2atoms_HP_invres/history.txt')
hist_nores = np.loadtxt('/Users/user/Desktop/HP_symmetry_loss/models/(grid)Model_2atoms_HP_nores/history.txt')


epochs=hist_nores[0:80,0]
#val_nores=hist_1[:,1]
train_1=hist_nores[0:80,2]
#val_invres=hist_invres[:,1]
train_2=hist_sym[0:80,2]
#val_res1=hist_res1[:,1]
train_3=hist_res[0:80,2]



directory= '/Users/user/Desktop/'
file_name = 'compare_2 atoms.png'

Title = '2 atoms'
res = 'Data restriction'
sym = 'Symmetry cost function'
no_res ='No symmetry breaking'
label_list = [no_res,sym,res]

dat = [[epochs, train_1],[epochs, train_2],[epochs,train_3]]

scm.GraphData_history(dat,['r', 'b','k'], label_list,Title,directory+file_name, Axx='Epochs',Axy='Cost')
