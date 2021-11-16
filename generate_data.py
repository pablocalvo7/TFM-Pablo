######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines_chain_model as scm
########## IMPORT PACKAGES ##########
######################################################



######################################################
########## SET TYPE OF DATA ##########
J=1 #Hopping
Nsamples=20000 #Number of samples for training/validation data
Ne=6 #Number of atoms
rigid = True #Type of hamiltonian. True: rigid; False: periodic
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

#Data restrictions
invres = False #Inversion restriction (e_1<=e_N)
invres_alternative_4atoms = False #Alternative nversion restriction (e_1<e_4 and e_2<e_3)
cyclicres = False #Cyclic restriction
inv_plus_cyc = False #save e_1 <= e_2 <= ... <= e_N
order_res = False
invres_alternative_6atoms = True

#For the plot's title
if(invres):
    res_name = 'invres'
if(cyclicres):
    res_name = 'cycres'
if((invres) and (cyclicres)):
    res_name = 'allres'
if((not invres) and (not cyclicres)):
    res_name = 'nores'
if(order_res):
    res_name = 'sorted'
########## SET TYPE OF DATA ##########
######################################################


######################################################
########## GENERATE DATA ##########
x=[] #Input data
y=[] #Output data

for i in range(Nsamples):
    print('Nsample:',i)
    list_energies = np.random.rand(Ne)

    #Data estrictions
    if(cyclicres): #I save the permutation with the higher energy in 2nd position
        maxx=max(list_energies)
        while(list_energies[1]<maxx):
            list_energies=scm.cyclic_permutation(list_energies,Ne)

    if(invres): #I save the permutation with the lower energy in 1st position
        if(list_energies[0]>list_energies[Ne-1]):
            list_energies = np.flip(list_energies)

    if(invres_alternative_4atoms):
        inside = (list_energies[0]<=list_energies[3]) & (list_energies[1]<=list_energies[2])
        inside_flip = (list_energies[0]>list_energies[3]) & (list_energies[1]>list_energies[2])
        if(inside_flip):
            list_energies = np.flip(list_energies)
        
        while(not inside):
            list_energies = np.random.rand(Ne)
            inside = (list_energies[0]<=list_energies[3]) & (list_energies[1]<=list_energies[2])
            inside_flip = (list_energies[0]>list_energies[3]) & (list_energies[1]>list_energies[2])
            if(inside_flip):
                list_energies = np.flip(list_energies)

    if(inv_plus_cyc):
        #save: e_1 <= e_2 <= ... <= e_N is fulfilled!
        save = list_energies[0] <= list_energies[1]
        for j in range(1,Ne):
            save = save and (list_energies[j] <= list_energies[j+1])
    
        num_cyc = 0 #number of cyclic permutations
        while(not save):
            num_cyc = num_cyc+1
            list_energies = scm.cyclic_permutation(list_energies,Ne)
            save = list_energies[0] <= list_energies[1]
            for j in range(1,Ne):
                save = save and (list_energies[j] <= list_energies[j+1])
            if(num_cyc==Ne):
                list_energies = np.flip(list_energies)
                save = list_energies[0] <= list_energies[1]
                for j in range(1,Ne):
                    save = save and (list_energies[j] <= list_energies[j+1])
                num_cyc = 0

    if(order_res):
        list_energies = sorted(list_energies)

    if(invres_alternative_6atoms):
        if(list_energies[0]>list_energies[Ne-1]):
            list_energies = np.flip(list_energies)
        if(list_energies[1]>list_energies[4]):
            list_energies = scm.two_elements_permutation(list_energies,Ne,1,4)
        #if(list_energies[2]>list_energies[3]):
            #list_energies = scm.two_elements_permutation(list_energies,Ne,2,3)

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

res_name ='invres_alt_1'

dfx.to_csv('/Users/user/Desktop/more_data_HR/alt_6atoms/ENERGIES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
dfy.to_csv('/Users/user/Desktop/more_data_HR/alt_6atoms/EIGENVALUES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
########## SAVE DATA ########## 
######################################################
