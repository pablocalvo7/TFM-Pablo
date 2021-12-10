######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################



######################################################
########## SET TYPE OF DATA ##########
Nsamples=20000 #Number of samples for training/validation data
Ne = 1 #Number of x values --> F_j(x1,...xN)
F1 = False
F_square = True
gap = 0.1

#Data restrictions
number_res = 1 #2,3,4,..., number of sorted values starting from the beginning
square_restriction = True
make_gap = False

#For the file's title
if(number_res==0):
    res_name = 'nores'
else:
    res_name = str(number_res)+'res'

res_name = 'positive_res'

if(F1):
    func_name = 'F1'

if(F_square):
    func_name = 'F_square'
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
    if(make_gap):
        while((abs(list_x[0]-list_x[1])<gap) or (abs(list_x[1]-list_x[2])<gap)):
            list_x = np.random.rand(Ne)

    #scm.sort_Nvalues(list_x,number_res)
    #End data restrictions
    if(square_restriction):
        if(list_x[0] < 0):
            list_x[0] = -list_x[0]

    if(F_square):
        func = list_x[0]*list_x[0]

    if(F1):
        func = scm.function_1(list_x,Ne)
    
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

if(make_gap):
    dfx.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/gap/x_'+func_name+'_'+str(Ne)+'values_'+res_name+'_gap'+str(gap)+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/gap/F_'+func_name+'_'+str(Ne)+'values_'+res_name+'_gap'+str(gap)+'.csv', sep = ',', header = False,index=False)
else:
    dfx.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/x_'+func_name+'_'+str(Ne)+'values_'+res_name+'.csv', sep = ',', header = False,index=False)
    dfy.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/F_'+func_name+'_'+str(Ne)+'values_'+res_name+'.csv', sep = ',', header = False,index=False)

########## SAVE DATA ########## 
######################################################
