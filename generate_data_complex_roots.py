######################################################
########## IMPORT PACKAGES ##########
import numpy as np
import pandas as pd
import subroutines as scm
########## IMPORT PACKAGES ##########
######################################################


######################################################
########## SET TYPE OF DATA ##########
Nsamples = 100000 #Number of samples for training/validation data 
n = 4 #exponent z^n

#Data restrictions
no_res = True

#For the plot's title
if(no_res):
    res_name = 'nores'

########## SET TYPE OF DATA ##########
######################################################


######################################################
########## GENERATE DATA ##########
x=[] #Input data
y=[] #Output data



for i in range(Nsamples):
    print('Nsample:',i)
    s = np.random.uniform(0,1)
    beta = np.random.uniform(0,2*np.pi)
    z = [s,beta]

    #Data estrictions

    #End data restrictions

    w = scm.power_n(s,beta,n) #w = z^n

    x.append(z)
    y.append(w)


    
x = np.array(x)
y = np.array(y) #EIGENVALUES NOT NORMALIZED

########## GENERATE DATA ##########
######################################################


######################################################
########## SAVE DATA ##########
dfx = pd.DataFrame(x)
dfy = pd.DataFrame(y)

directory = '/Users/user/Desktop/TFM/9. Complex roots/data/'


dfx.to_csv(directory+'z_'+str(n)+'root_'+res_name+'.csv', sep = ',', header = False,index=False)
dfy.to_csv(directory+'w_'+str(n)+'root_'+res_name+'.csv', sep = ',', header = False,index=False)

########## SAVE DATA ########## 
######################################################
