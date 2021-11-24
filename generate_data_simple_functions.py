######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines_chain_model as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## FUNCTIONS ##########

def function_1(list_x,Ne):
    list_function=[]
    for i in range(Ne):
        x_pow = [j ** (i+1) for j in list_x]
        list_function.append(sum(x_pow))
    
    return list_function


########## FUNCTIONS ##########
######################################################


######################################################
########## SET TYPE OF DATA ##########
Nsamples=20000 #Number of samples for training/validation data
Ne = 2 #Number of x values --> F_j(x1,...xN)
F1 = True

#Data restrictions
perm_2values = False

#For the plot's title
if(perm_2values):
    res_name = '12'
else:
    res_name = 'nores'

########## SET TYPE OF DATA ##########
######################################################


######################################################
########## GENERATE DATA ##########
x=[] #Input data
y=[] #Output data

for i in range(Nsamples):
    print('Nsample:',i)
    list_x = np.random.rand(Ne)

    #Data estrictions
    if(perm_2values):
        if(list_x[0]>list_x[1]):
            list_x = np.flip(list_x)
    #End data restrictions

    if(F1):
        func = function_1(list_x,Ne)
    
    x.append(list_x)
    y.append(func)  
    
x = np.array(x)
y = np.array(y) #EIGENVALUES NOT NORMALIZED
########## GENERATE DATA ##########
######################################################


######################################################
########## SAVE DATA ##########
dfx = pd.DataFrame(x)
dfy = pd.DataFrame(y)


dfx.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/x_'+str(Ne)+'values_'+res_name+'.csv', sep = ',', header = False,index=False)
dfy.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/F_'+str(Ne)+'values_'+res_name+'.csv', sep = ',', header = False,index=False)
########## SAVE DATA ########## 
######################################################
