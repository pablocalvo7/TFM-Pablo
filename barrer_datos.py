######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## SET TYPE OF DATA ##########
generate_random = True
generate_grid = False
J=1 #Hopping
if(generate_grid):
    Nsamples = 100000 #Number of samples ; GENERATE RANDOM
if(generate_random):
    Nsamples = 50000
Ne = 5 #Number of atoms
rigid = False #Type of hamiltonian. True: rigid; False: periodic
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

#Data restrictions
res_1 = False
res_2 = False
res_3 = False
res_4 = False
res_5 = False
res_5_diff = True
#pentagon

#For the plot's title
if(res_1):
    res_name = 'sweep_1'
if(res_2):
    res_name = 'sweep_2'
if(res_3):
    res_name = 'sweep_3'
if(res_4):
    res_name = 'sweep_4'
if(res_5):
    res_name = 'sweep_5'
if(res_5_diff):
    res_name = 'sweep_5_diff'
########## SET TYPE OF DATA ##########
######################################################


######################################################
########## IMPORT DATA AND SWEEP ##########
x = []
y = []

directory = '/Users/user/Desktop/barrer_datos_2/data/'
filex = directory+'ENERGIES_5atoms_HP_sweep_4.csv'
filey = directory+'EIGENVALUES_5atoms_HP_sweep_4.csv'

energies,eigenvalues=scm.read_data(filex,filey)

for i in range(Nsamples):
    print('Nsample:',i)
    list_energies = energies[i,:].tolist()

    #Data estrictions 
    if(res_1):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_1(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0

    if(res_2):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_2(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0

    if(res_3):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_3(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0

    if(res_4):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_4(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0

    if(res_5):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_5(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0

    if(res_5_diff):
        num_cyc = 0 #number of cyclic permutations
        while(not scm.save_sweep_5_diff(list_energies,Ne)):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                num_cyc = 0
    #End data restrictions

    autovals = scm.autoenergies_chain_Natoms(list_energies,Ne,J,rigid)
    
    x.append(list_energies)
    y.append(autovals)

    
x = np.array(x)
y = np.array(y) #EIGENVALUES NOT NORMALIZED
########## GENERATE DATA ##########
######################################################


######################################################
########## SAVE DATA ##########
dfx = pd.DataFrame(x)
dfy = pd.DataFrame(y)

directory = '/Users/user/Desktop/barrer_datos_2/data/'

if(generate_random):
    dfx.to_csv(directory+'ENERGIES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv(directory+'EIGENVALUES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)

if(generate_grid):
    dfx.to_csv(directory+'(grid)ENERGIES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv(directory+'(grid)EIGENVALUES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
########## SAVE DATA ########## 
######################################################
