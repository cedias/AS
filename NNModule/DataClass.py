#DataClass.py
#-*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata


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


def getMnistDualDataset():
    mnist=fetch_mldata('MNIST original')
    #Creation des vecteurs d'entrée
    mnist_6=mnist.data[mnist.target==6]
    nb_6=len(mnist_6)
    mnist_8=mnist.data[mnist.target==8]
    nb_8=len(mnist_8)
    mnist_6_8=np.vstack((mnist_6,mnist_8))
    print "%d 6s and %d 8s" % (nb_6,nb_8)

    #Creation des vecteurs de sortie
    target_6_8=np.array([[1]])
    for i in range(nb_6-1):
        target_6_8=np.vstack((target_6_8,[1]))
    for i in range(nb_8):
        target_6_8=np.vstack((target_6_8,[-1]))
        
    print "%d/%d vecteurs d'apprentissage" % (len(target_6_8),len(mnist_6_8))
    randomvec=np.random.rand(len(target_6_8))
    randomvec=randomvec>0.8
    target_6_8=target_6_8[randomvec]
    mnist_6_8=mnist_6_8[randomvec]
    print "%d/%d vecteurs d'apprentissage apres echantillonage" % (len(target_6_8),len(mnist_6_8))
    randomvec=np.random.rand(len(target_6_8))
    randomvec=randomvec>0.5

    train_data=mnist_6_8[randomvec]
    train_label=target_6_8[randomvec]
    test_data=mnist_6_8[np.logical_not(randomvec)]
    test_label=target_6_8[np.logical_not(randomvec)]
    print "%d training examples and %d testing examples " % (len(train_data),len(test_data))
    return (train_data,train_label,test_data,test_label)