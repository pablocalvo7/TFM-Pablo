######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## SET TYPE OF DATA ##########
J=1 #Hopping
Nsamples = 50000 #Number of samples for training/validation data ; GENERATE RANDOM
nbins = 12 #GENERATE GRID
delta = 1/nbins
generate_random = True
generate_grid = False
Ne = 4 #Number of atoms
rigid = True #Type of hamiltonian. True: rigid; False: periodic
if(rigid): #For the plot's title
    H_name = 'HR'
else:
    H_name = 'HP'

change_variables = True

#Data restrictions
invres = False #Inversion restriction (e_1<=e_N)
cyclicres = False #Cyclic restriction
inv_plus_cyc = False #save e_N > e_1, e_2, e_3, ... and e_1 < e_{N-1}
invres_max_diff = True
no_res = False
#pentagon
res_4 = False
res_5 = False
res_6 = False
res_7 = False
#pentagon

#For the plot's title
if(invres):
    res_name = 'invres'
if(cyclicres):
    res_name = 'cycres'
if(invres_max_diff):
    res_name = 'invres_max'
if(inv_plus_cyc):
    res_name = 'inv_plus_cyc'
if(no_res):
    res_name = 'nores'
if(res_4):
    res_name = 'res_4'
if(res_5):
    res_name = 'res_5'
if(res_6):
    res_name = 'res_6'
if(res_7):
    res_name = 'res_7'
########## SET TYPE OF DATA ##########
######################################################


######################################################
########## GENERATE DATA ##########
x=[] #Input data
y=[] #Output data


if(generate_random):

    for i in range(Nsamples):
        print('Nsample:',i)
        list_energies = np.random.rand(Ne)

        #Data estrictions
        if(cyclicres): #I save the permutation with the higher energy in Nth position
            maxx=max(list_energies)
            while(list_energies[Ne-1]<maxx):
                list_energies=scm.cyclic_permutation(list_energies,Ne)

        if(invres): #I save the permutation with the lower energy in 1st position
            if(list_energies[0]>list_energies[Ne-1]):
                list_energies = np.flip(list_energies)

        if(inv_plus_cyc):
            #save: e_N > e_1, e_2, e_3, ... and e_1 < e_{N-1} is fulfilled!
            print(list_energies)
            num_cyc = 0 #number of cyclic permutations
            while(not scm.save_list(list_energies,Ne)):
                num_cyc = num_cyc+1
                list_energies = scm.cyclic_permutation(list_energies,Ne)
                if(num_cyc==Ne):
                    list_energies = np.flip(list_energies)
                    num_cyc = 0
    
        if(invres_max_diff):
            diff_14 = abs(list_energies[0] - list_energies[3])
            diff_23 = abs(list_energies[1] - list_energies[2])
        
            if(diff_14 > diff_23):
                if(list_energies[0] > list_energies[3]):
                    list_energies = np.flip(list_energies)
            else:
                if(list_energies[1] > list_energies[2]):
                    list_energies = np.flip(list_energies)

        if(res_4):
            #save: e_2 > e_4, e_5, e_1 is fulfilled!
            num_cyc = 0 #number of cyclic permutations
            while(not scm.save_list_4(list_energies,Ne)):
                num_cyc = num_cyc+1
                list_energies = scm.cyclic_permutation(list_energies,Ne)
                if(num_cyc==Ne):
                    list_energies = np.flip(list_energies)
                    num_cyc = 0
        
        if(res_5):
            #save: e_2 < e_4, e_5 is fulfilled!
            num_cyc = 0 #number of cyclic permutations
            while(not scm.save_list_5(list_energies,Ne)):
                num_cyc = num_cyc+1
                list_energies = scm.cyclic_permutation(list_energies,Ne)
                if(num_cyc==Ne):
                    list_energies = np.flip(list_energies)
                    num_cyc = 0

        if(res_6):
            #save: e_3 > e_4, e_5 and e_4 < e_2 is fulfilled!
            num_cyc = 0 #number of cyclic permutations
            while(not scm.save_list_6(list_energies,Ne)):
                num_cyc = num_cyc+1
                list_energies = scm.cyclic_permutation(list_energies,Ne)
                if(num_cyc==Ne):
                    list_energies = np.flip(list_energies)
                    num_cyc = 0

        if(res_7):
            #save: e_5 > e_1, e_2 is fulfilled!
            num_cyc = 0 #number of cyclic permutations
            while(not scm.save_list_7(list_energies,Ne)):
                num_cyc = num_cyc+1
                list_energies = scm.cyclic_permutation(list_energies,Ne)
                if(num_cyc==Ne):
                    list_energies = np.flip(list_energies)
                    num_cyc = 0

        #End data restrictions

        autovals = scm.autoenergies_chain_Natoms(list_energies,Ne,J,rigid)

        if(change_variables):
            a = list_energies[0] - list_energies[3]
            b = list_energies[1] - list_energies[2]
            c = 0.5*(list_energies[0] + list_energies[3])
            d = 0.5*(list_energies[1] + list_energies[2])

            list_energies = [a,b,c,d]

        x.append(list_energies)
        y.append(autovals)

if(generate_grid): #grid of data: Ne = 5 --> i,j,k,l,m (pentagon)
    number_of_data = 0
    for i in range(nbins):
        for j in range(nbins):
            for k in range(nbins):
                for l in range(nbins):
                    for m in range(nbins):
                        number_of_data = number_of_data + 1
                        print("grid position: ",i,j,k,l,m, "ndata: ",number_of_data)
                        list_energies = [i*delta, j*delta, k*delta, l*delta, m*delta]

                        #Data restrictions
                        if(invres): #I save the permutation with the lower energy in 1st position
                            if(list_energies[0]>list_energies[Ne-1]):
                                list_energies = np.flip(list_energies)
                        
                        if(cyclicres): #I save the permutation with the higher energy in Nth position
                            maxx=max(list_energies)
                            while(list_energies[Ne-1]<maxx):
                                list_energies=scm.cyclic_permutation(list_energies,Ne)

                        if(inv_plus_cyc):
                            #save: e_N > e_1, e_2, e_3, ... and e_1 < e_{N-1} is fulfilled!
                            num_cyc = 0 #number of cyclic permutations
                            while(not scm.save_list(list_energies,Ne)):
                                num_cyc = num_cyc+1
                                list_energies = scm.cyclic_permutation(list_energies,Ne)
                                if(num_cyc==Ne):
                                    list_energies = np.flip(list_energies)
                                    num_cyc = 0

                        if(res_4):
                            #save: e_2 > e_4, e_5, e_1 is fulfilled!
                            num_cyc = 0 #number of cyclic permutations
                            while(not scm.save_list_4(list_energies,Ne)):
                                num_cyc = num_cyc+1
                                list_energies = scm.cyclic_permutation(list_energies,Ne)
                                if(num_cyc==Ne):
                                    list_energies = np.flip(list_energies)
                                    num_cyc = 0

                        if(res_5):
                            #save: e_2 < e_4, e_5 is fulfilled!
                            num_cyc = 0 #number of cyclic permutations
                            while(not scm.save_list_5(list_energies,Ne)):
                                num_cyc = num_cyc+1
                                list_energies = scm.cyclic_permutation(list_energies,Ne)
                                if(num_cyc==Ne):
                                    list_energies = np.flip(list_energies)
                                    num_cyc = 0

                        if(res_6):
                            #save: e_3 > e_4, e_5 and e_4 < e_2 is fulfilled!
                            num_cyc = 0 #number of cyclic permutations
                            while(not scm.save_list_6(list_energies,Ne)):
                                num_cyc = num_cyc+1
                                list_energies = scm.cyclic_permutation(list_energies,Ne)
                                if(num_cyc==Ne):
                                    list_energies = np.flip(list_energies)
                                    num_cyc = 0

                        if(res_7):
                            #save: e_5 > e_1, e_2 is fulfilled!
                            num_cyc = 0 #number of cyclic permutations
                            while(not scm.save_list_7(list_energies,Ne)):
                                num_cyc = num_cyc+1
                                list_energies = scm.cyclic_permutation(list_energies,Ne)
                                if(num_cyc==Ne):
                                    list_energies = np.flip(list_energies)
                                    num_cyc = 0
                        #End data restrictions

                        autovals = scm.autoenergies_chain_Natoms(list_energies,Ne,J,rigid)

                        if(change_variables):
                            a = list_energies[0] - list_energies[3]
                            b = list_energies[1] - list_energies[2]
                            c = 0.5*(list_energies[0] + list_energies[3])
                            d = 0.5*(list_energies[1] + list_energies[2])

                            list_energies = [a,b,c,d]

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

directory = '/Users/user/Desktop/HR_abcd/'

if(generate_random):
    dfx.to_csv(directory+'ENERGIES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv(directory+'EIGENVALUES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)

if(generate_grid):
    dfx.to_csv(directory+'(grid)ENERGIES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv(directory+'(grid)EIGENVALUES_'+str(Ne)+'atoms_'+H_name+'_'+res_name+'.csv', sep = ',', header = False,index=False)
########## SAVE DATA ########## 
######################################################
