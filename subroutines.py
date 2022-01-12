######################################################
########## IMPORT PACKAGES ##########
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow.keras.backend as K
########## IMPORT PACKAGES ##########
######################################################




######################################################
########## FUNCTIONS ##########

#CHAIN MODEL

def autoenergies_chain_Natoms(e,N,J,rigid): #periodic hamiltonian
    A=[]
    for i in range(N):
        row=[]
        for j in range(N):
            if(i==j):
                row.append(e[j])
            else:
                if((j==i-1)|(j==i+1)):
                    row.append(J)
                else:
                    row.append(0)
        if(not rigid): #if periodic hamiltonian
            if(i==0):
                row[N-1]=J
            if(i==N-1):
                row[0]=J
        A.append(row)
    A=np.array(A)
    autovals=np.linalg.eigvals(A)
    autovals=sorted(autovals)

    return autovals


def cyclic_permutation(e,N): #(e_1, ..., e_N) --> (e_N, e_1, ..., e_{N-1})
    e_new=[]
    e_new.append(e[N-1])
    for i in range(N-1):
        e_new.append(e[i])

    return e_new

#inv_plus_cyc restriction
def save_list(list_energies, Ne): #save: e_N > e_1, e_2, e_3, ... and e_1 < e_{N-1} is fulfilled!

    save_1 = list_energies[0] <= list_energies[Ne-1]
    for j in range(1,Ne-1):
        save_1 = save_1 and (list_energies[j] <= list_energies[Ne-1])
    save_2 = list_energies[0] <= list_energies[Ne-2]
    save_tot = save_1 and save_2

    return save_tot


#break all symmetries apart from c_5 , c_5^4 , s_4 and s_5 in pentagon
def save_list_4(list_energies, Ne): #save: e_2 > e_4, e_5, e_1 is fulfilled!

    save_tot = list_energies[0] <= list_energies[1]
    save_tot = save_tot and (list_energies[3] <= list_energies[1])
    save_tot = save_tot and (list_energies[Ne-1] <= list_energies[1]) 

    return save_tot

#break symmetries c_5^2 , c_5^3 , s_1 and s_2 in pentagon
def save_list_5(list_energies, Ne): #save: e_2 < e_4, e_5 is fulfilled!

    save_tot = list_energies[1] <= list_energies[3]
    save_tot = save_tot and (list_energies[1] <= list_energies[Ne-1])

    return save_tot

#break symmetries s_1 , s_2 and s_3 in pentagon
def save_list_6(list_energies, Ne): #save: e_3 > e_4, e_5 and e_4 < e_2 is fulfilled!

    save_tot = list_energies[3] <= list_energies[2]
    save_tot = save_tot and (list_energies[Ne-1] <= list_energies[2])
    save_tot = save_tot and (list_energies[3] <= list_energies[1])

    return save_tot

#break symmetries s_1 and s_2 in pentagon
def save_list_7(list_energies, Ne): #save: e_5 > e_1, e_2 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[1] <= list_energies[Ne-1])

    return save_tot

#break symmetry s_1 in pentagon
def save_sweep_1(list_energies, Ne): #save: e_5 > e_1 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]

    return save_tot

#break symmetries s_1 and s_3 in pentagon
def save_sweep_2(list_energies, Ne): #save: e_5 > e_1, e_3 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[2] <= list_energies[Ne-1])

    return save_tot

#break symmetries s_1 , s_3 , s_4 , c_5 and c_5^4 in pentagon
def save_sweep_3(list_energies, Ne): #save: e_5 > e_1 , e_3 , e_4 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[2] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[3] <= list_energies[Ne-1])

    return save_tot

#break symmetry s_1 , s_3 , s_4 , s_5 , c_5 and c_5^4 in pentagon
def save_sweep_4(list_energies, Ne): #save: e_5 > e_1 , e_3 , e_4 and e_1 < e_4 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[2] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[3] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[0] <= list_energies[3])

    return save_tot

#break all symmetries in pentagon
def save_sweep_5(list_energies, Ne): #save: e_5 > e_1 , e_2 , e_3 , e_4 and e_1 < e_4 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[1] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[2] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[3] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[0] <= list_energies[3])

    return save_tot

#break all symmetries in pentagon
def save_sweep_5_diff(list_energies, Ne): #save: e_5 > e_1 , e_2 , e_3 , e_4 and MAX_DIFF e_1 < e_4 or e_2 < e_3 is fulfilled!

    save_tot = list_energies[0] <= list_energies[Ne-1]
    save_tot = save_tot and (list_energies[1] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[2] <= list_energies[Ne-1])
    save_tot = save_tot and (list_energies[3] <= list_energies[Ne-1])
    diff_03 = abs(list_energies[0] - list_energies[3])
    diff_12 = abs(list_energies[1] - list_energies[2])
    if(diff_03 > diff_12):
        save_tot = save_tot and (list_energies[0] <= list_energies[3])
    else:
        save_tot = save_tot and (list_energies[1] <= list_energies[2])

    return save_tot

def normalization(energies, eigenvalues, Ne, J, rigid): #NORMALIZATION OF EIGENVALUES DATA [0,1)

    lowest = np.zeros(Ne)
    highest = np.ones(Ne)
    min_eig = autoenergies_chain_Natoms(lowest,Ne,J,rigid)[0]
    max_eig = autoenergies_chain_Natoms(highest,Ne,J,rigid)[Ne-1]
    eigen_norm=(eigenvalues-min_eig)/(max_eig-min_eig)
    x=eigen_norm; y=energies #NN input: eigenvalues; NN output: energies (inverse problem)

    return x,y


def read_data(fileX, filey):
    X = pd.read_csv(fileX, header = None)
    X = np.array(X)
    y = pd.read_csv(filey, header = None)
    y = np.array(y)

    return X, y


def GraphData_history(datalist, typeplotlist, labellist, Title, filename_data,\
              Axx = "x", Axy = "y",\
              left=None, right=None, bottom=None, top = None):
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16.
    ngraph = len(datalist)
    for il in range(ngraph):
        print(labellist[il])
        plt.plot(datalist[il][0], datalist[il][1],typeplotlist[il], 
                 label=labellist[il], markersize=10, linewidth=4)
    plt.axis([left, right, bottom, top])
    if ngraph != 1:
        plt.legend(loc='best')
    plt.xlabel(Axx,fontsize=30,fontname='Gill Sans')
    plt.ylabel(Axy,fontsize=30,fontname='Gill Sans')
    plt.title(Title, fontsize=30,fontname= 'Gill Sans')
    plt.grid()
    plt.savefig(filename_data, bbox_inches='tight')
    plt.show()

def GraphData_prediction(datalist, labellist, Title, filename_data,\
              Axx = "x", Axy = "y",\
              left=None, right=None, bottom=None, top = None):
    plt.rcParams["figure.figsize"] = (6,6)
    fig,ax=plt.subplots()
    #ax.rcParams['font.size'] = 16.
    #line x=y
    colorlist = ['r','b','k','lime']
    markerlist = ['s','^','P','D']
    xx=np.linspace(-0.1,1.1,20)
    yy=xx
    ax.plot(xx,yy,color='k',linewidth=3,label='NN=QM')
    ngraph = len(datalist)
    for il in range(ngraph):
        print(labellist[il])
        ax.scatter(datalist[il][0], datalist[il][1], color = colorlist[il],
                 marker = markerlist[il], linewidths=2,label=labellist[il])

    #plt.axis([left, right, bottom, top])
    plt.xlabel(Axx,fontsize=25,fontname='Gill Sans')
    plt.ylabel(Axy,fontsize=25,fontname='Gill Sans')
    if ngraph != 1:
        plt.legend(loc='best')
    plt.title(Title, fontsize=30,fontname= 'Gill Sans')

    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.0])

    x_ticks=np.linspace(0.0,1.0,5)
    ax.set_xticks(x_ticks)
    y_ticks=np.linspace(0.0,1.0,5)
    ax.set_yticks(y_ticks)

    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(direction='in',top=True,right=True,width=2,length=5.5)


    #plt.ylim(-25,60)
    plt.savefig(filename_data, bbox_inches='tight')
    plt.show()




def save_history(history_keras,path_hist):
    history_dict = history_keras.history        
    training_cost = history_dict['loss'] #P: guarda los valores del coste para cada epoch
    evaluation_cost = history_dict['val_loss']

    epochs=len(evaluation_cost) #P: numero de epochs
    xx = np.linspace(0,epochs-1,epochs) #P: lista de 0 a epochs-1
    
    f1=open(path_hist,"w")
    for i in range(0,epochs): #P: str() convierte valor numérico en caracter
        summary= str(xx[i])+' '+str(evaluation_cost[i])+' '+' '+str(training_cost[i])+' '+''+'\n'
        f1.write(summary)
        
    f1.close()   

def two_elements_permutation(list,length,pos1,pos2):
    new_list=[]
    for i in range(length):
        if(i==pos1):
            new_list.append(list[pos2])
        else:
            if(i==pos2):
                new_list.append(list[pos1])
            else:
                new_list.append(list[i])
    
    return new_list

def cyclic_permutation_tensor(tensor,N): #(e_1, ..., e_N) --> (e_N, e_1, ..., e_{N-1})
    tensor_new = tensor[N-1]
    tensor_new = tf.reshape(tensor_new,[1,])
    for i in range(N-1):
        tensor_add = tensor[i]
        tensor_add = tf.reshape(tensor_add,[1,])
        tensor_new = tf.concat([tensor_new, tensor_add], axis=0)
    return tensor_new

def MSE_one_value(y_true,y_pred):
    sq_diff = tf.square(y_true-y_pred)
    sq_diff = tf.math.reduce_sum(sq_diff)
    mse = 0.5*sq_diff
    return mse
    

def symmetry_loss(y_true,y_pred):
    b_size = y_pred.shape[0]
    Ne = y_pred.shape[1]

    loss_vector =[]
    for i in range(b_size):
        min_loss = MSE_one_value(y_true[i],y_pred[i])
        sym_y = y_pred[i]
        for j in range(Ne):
            sym_y = cyclic_permutation_tensor(sym_y,Ne)
            new_loss = MSE_one_value(y_true[i],sym_y)
            if( new_loss < min_loss ):
                min_loss = new_loss

        sym_y = K.reverse(sym_y,axes=0)

        for j in range(Ne):
            sym_y = cyclic_permutation_tensor(sym_y,Ne)
            new_loss = MSE_one_value(y_true[i],sym_y)
            if( new_loss < min_loss ):
                min_loss = new_loss

        loss_vector.append(min_loss)

    loss_vector = tf.convert_to_tensor(loss_vector)
    loss = K.mean(loss_vector)
    return loss

def symmetry_loss_square(y_true,y_pred):
    batch_size = np.shape(y_true)[0]

    loss_vector =[]
    for i in range(batch_size):
        min_loss = MSE_one_value(y_true[i],y_pred[i])
        sym_y = -y_pred[i]
        new_loss = MSE_one_value(y_true[i],sym_y)
        if( new_loss < min_loss ):
            min_loss = new_loss
        loss_vector.append(min_loss)

    loss = sum(loss_vector)/batch_size
    return loss

#SIMPLE FUNCTIONS

def function_1(list_x,Ne):
    list_function=[]
    for i in range(Ne):
        x_pow = [j ** (i+1) for j in list_x]
        list_function.append(sum(x_pow))
    
    return list_function


def normalization_function_1(F,Ne): #NORMALIZATION OF F DATA [0,1)

    #if we generate values 0<x_i<1
    min = 0
    max= Ne #maximum value of any component F_j
    F_norm = (F-min)/(max-min)

    return F_norm

def normalization_xx_range_test(xx): #NORMALIZATION OF F DATA [0,1)

    #if we generate values -1<x_i<1
    min = -1
    max= 1 #maximum value of any component F_j
    xx_norm = (xx-min)/(max-min)

    return xx_norm

def sort_Nvalues(arr,number_res):

    aux = np.sort(arr[0:number_res])
    arr[0:number_res]=aux
    arr_new = np.append(aux,arr[number_res:])

    return arr_new


########## FUNCTIONS ##########
######################################################
