#tools.py
#-*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt

class LabeledSet:  
    
    def __init__(self,x,y,input_dim,output_dim):
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    #Renvoie la dimension de l'espace d'entrée
    def getInputDimension(self):
        return self.input_dim
       
    
    #Renvoie la dimension de l'espace de sortie
    def getOutputDimension(self):
        return self.output_dim
    
    #Renvoie le nombre d'exemple dans le set
    def size(self):
        return len(self.x)
    #Renvoie la valeur de x_i
    def getX(self,i):
        return self.x[i]
        
    
    #Renvoie la valeur de y_i
    def getY(self,i):
        return self.y[1]



def createGaussianDataset(positive_center_1,positive_center_2,positive_sigma,negative_center_1,negative_center_2,negative_sigma,nb_points):
    pos = True
    first = True
    
    while nb_points>0:
        
        if pos:
            a = np.random.multivariate_normal([positive_center_1,positive_center_2],[[positive_sigma,0],[0,positive_sigma]])
            if first:
                x=a
                first = False
                y = np.array([1])
            else:
                x = np.vstack((x,a))
                y = np.vstack((y,np.array([1])))
                pos = False
        else:
            b = np.random.multivariate_normal([negative_center_1,negative_center_2],[[negative_sigma,0],[0,negative_sigma]])
            x = np.vstack((x,b))
            y = np.vstack((y,np.array([-1])))
            pos = True
        
        
        nb_points -= 1
       
         
    return LabeledSet(x,y,2,1)




def plot2DSet(set):
    plt.scatter(set.x[:,0],set.x[:,1])
    plt.show()

def plot_frontiere(x,f,step=20):
    mmax=x.max(0)
    mmin=x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))

# calcul de la prediction pour chaque point de la grille
    res=np.array([f(grid[i,:])[0] for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["orange","gray"],levels=[-1000,0,1000],linewidth=2)


def f(x):
    score=[x[0]+x[1]]
    return(score)