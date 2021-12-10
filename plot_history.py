######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################



F1 = True
Ne = 3 #Number of values
if(F1): #For the plot's title
    func_name = 'F1'

hist_1 = np.loadtxt('/Users/user/Desktop/TFM/6. Simple functions/models/gap/Model_F1_3values_3res_gap0.01/history.txt')
hist_2 = np.loadtxt('/Users/user/Desktop/TFM/6. Simple functions/models/gap/Model_F1_3values_3res_gap0.05/history.txt')
hist_3 = np.loadtxt('/Users/user/Desktop/TFM/6. Simple functions/models/gap/Model_F1_3values_3res_gap0.1/history.txt')
#hist_4 = np.loadtxt('/Users/user/Desktop/HP_models_more_data/inv_plus_cyc/Model_4atoms_HP_allres/history.txt')


epochs=hist_1[20:,0]
#val_nores=hist_1[:,1]
train_1=hist_1[20:,2]
#val_invres=hist_invres[:,1]
train_2=hist_2[20:,2]
#val_res1=hist_res1[:,1]
train_3=hist_3[20:,2]


file_graph = '/Users/user/Desktop/compare_gaps_3values.png'

scm.GraphData_history([
               [epochs, train_1],
               [epochs, train_2],[epochs,train_3]],['r', 'b','k'], 
              ['gap = 0.01', 'gap = 0.05','gap = 0.1'],'3 values',file_graph, Axx='Epochs', Axy='Cost')