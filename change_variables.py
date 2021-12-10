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
Ne = 2 #Number of x values --> F_j(x1,...xN)
F1 = True
gap = 0.1

filex = '/Users/user/Desktop/TFM/6. Simple functions/data/x_F1_2values_12.csv'
filey = '/Users/user/Desktop/TFM/6. Simple functions/data/F_F1_2values_12.csv'

x,F = scm.read_data(filex,filey)


#Data restrictions
number_res = 2 #2,3,4,..., number of sorted values starting from the beginning
make_gap = False

#For the file's title
if(number_res==0):
    res_name = 'nores'
else:
    res_name = str(number_res)+'res'

if(F1):
    func_name = 'F1'
########## SET TYPE OF DATA ##########
######################################################


######################################################
########## GENERATE DATA ##########

a=[] #new input data: a_1=0.5(x1+x2); a_2=x2-x1

for i in range(Nsamples):
    a_1 = 0.5*(x[i,0]+x[i,1])
    a_2 = x[i,1]-x[i,0]
    list_a =[a_1,a_2]
    a.append(list_a)

a = np.array(a)

########## GENERATE DATA ##########
######################################################


######################################################
########## SAVE DATA ##########

dfa = pd.DataFrame(a)
dfa.to_csv('/Users/user/Desktop/TFM/6. Simple functions/data/alpha_'+func_name+'_'+str(Ne)+'values_'+res_name+'.csv', sep = ',', header = False,index=False)

########## SAVE DATA ########## 
######################################################
