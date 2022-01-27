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
N_broken = 1

#Data restrictions
res_name = str(N_broken)+'broken'


########## SET TYPE OF DATA ##########
######################################################


######################################################
########## IMPORT DATA AND SWEEP ##########
x = []
y = []

directory = '/Users/user/Desktop/TFM/9. Complex roots/data/'
filex = directory+'z_4root_nores.csv'
filey = directory+'w_4root_nores.csv'

z,w=scm.read_data(filex,filey)

for i in range(Nsamples):
    print('Nsample:',i)
    z_new = z[i,:].tolist()
    z_bug_1 = z_new
    #Data estrictions 
    while( z_new[1] > (2*np.pi*(n-N_broken))/n ):
        z_new = scm.transformation_cnk(z_new,n,1)
    z_bug_2 = z_new

    num_transf = np.random.randint(0,n-N_broken)
    z_new = scm.transformation_cnk(z_new,n,num_transf)
    z_bug_3 = z_new
    print('1: ',z_bug_1,' 2: ',z_bug_2,' 3: ',z_bug_3 )
    #End data restrictions

    #w = scm.power_n(z_new[0],z_new[1],n) #w = z^n
    
    x.append(z_new)
    y.append(w)

    
x = np.array(x)
y = np.array(y) #EIGENVALUES NOT NORMALIZED
########## GENERATE DATA ##########
######################################################


######################################################
########## SAVE DATA ##########
dfx = pd.DataFrame(x)
#dfy = pd.DataFrame(y)

directory = '/Users/user/Desktop/TFM/9. Complex roots/data/'


dfx.to_csv(directory+'z_'+str(n)+'root_'+res_name+'.csv', sep = ',', header = False,index=False)
#dfy.to_csv(directory+'w_'+str(n)+'root_'+res_name+'.csv', sep = ',', header = False,index=False)

########## SAVE DATA ########## 
######################################################
