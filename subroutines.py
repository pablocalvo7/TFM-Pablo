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
########## IMPORT PACKAGES ##########
######################################################




######################################################
########## FUNCTIONS ##########
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


def cyclic_permutation(e,N): #(e_1, ..., e_N) --> (e_2, ..., e_N, e_1)
    e_new=[]
    for i in range(N-1):
        e_new.append(e[i+1])
    e_new.append(e[0])

    return e_new


def normalization(energies, eigenvalues, Ne, J, rigid): #NORMALIZATION OF EIGENVALUES DATA [0,1)

    lowest = np.zeros(Ne)
    highest = np.ones(Ne)
    min_eig = autoenergies_chain_Natoms(lowest,Ne,J,rigid)[0]
    max_eig = autoenergies_chain_Natoms(highest,Ne,J,rigid)[Ne-1]
    eigen_norm=(eigenvalues-min_eig)/(max_eig-min_eig)
    x=eigen_norm; y=energies #NN input: eigenvalues; NN output: energies

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
    #plt.legend(['NN=QM','NN no restricted','NN all restricted'],fontsize=12.8,loc="upper left")
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


########## FUNCTIONS ##########
######################################################
