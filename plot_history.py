######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subroutines_chain_model as scm
########## IMPORT PACKAGES ##########
######################################################




rigid = False #Type of hamiltonian. True: rigid; False: periodic
Ne = 3 #Number of atoms
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

hist_1 = np.loadtxt('/Users/user/Desktop/HP_models_more_data/e1<e2<e3/Model_3atoms_HP_nores/history.txt')
hist_2 = np.loadtxt('/Users/user/Desktop/HP_models_more_data/e1<e2<e3/Model_3atoms_HP_allres/history.txt')
#hist_3 = np.loadtxt('/Users/user/Desktop/HR_models_more_data/Model_6atoms_HR_invres_alt_2/history.txt')
#hist_4 = np.loadtxt('/Users/user/Desktop/HP_models_more_data/inv_plus_cyc/Model_4atoms_HP_allres/history.txt')


epochs=hist_1[:,0]
#val_nores=hist_1[:,1]
train_1=hist_1[:,2]
#val_invres=hist_invres[:,1]
train_2=hist_2[:,2]
#val_res1=hist_res1[:,1]
#train_3=hist_3[:,2]


file_graph = '/Users/user/Desktop/HP_models_more_data/'+H_name+'_'+str(Ne)+'atoms.png'

scm.GraphData_history([
               [epochs, train_1],
               [epochs, train_2]],['r', 'b','k'], 
              ['No restricted', 'All restricted'],H_name+', '+str(Ne)+' atoms (train data)',file_graph, Axx='Epochs', Axy='Loss')