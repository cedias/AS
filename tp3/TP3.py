# -*- coding: utf-8 -*-


import numpy as np



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

       
# Implémentation des Losses


class Loss:
    
    #Calcule la valeur du loss étant données les valeurs prédites et désirées
    def getLossValue(self,predicted_output,desired_output):
        pass
    
    #Calcule le gradient (pour chaque cellule d'entrée) du coût
    def backward(self, predicted_output,desired_output):
        pass 


#Square Loss

class SquareLoss(Loss):
    def getLossValue(self,predicted_output,desired_output):
        return np.power(desired_output-predicted_output,2)
    
    def backward(self, predicted_output,desired_output):
        return 2*(predicted_output-desired_output)
    
#HingeLoss

class HingeLoss(Loss):
    def getLossValue(self,predicted_output,desired_output):
        return np.max(np.zeros(predicted_output.size), desired_output*predicted_output))
    
    def backward(self, predicted_output,desired_output):
        if(desired_output * predicted_output < 1)
            return -desired_output*predicted_output
        else
            return np.zeros(predicted_output.size)
    pass

# <markdowncell>

# Implémentation des Modules
# ======
# 
# Nous allons maintenant implémenter quelques modules de base

# <codecell>

class Module:
    
    #Permet le calcul de la sortie du module
    def forward(self,input):
        pass
    
    #Permet le calcul du gradient des cellules d'entrée
    def backward_delta(self,input,delta_module_suivant):
        pass
    
    #Permet d'initialiser le gradient du module
    def init_gradient(self):
        pass
    
    #Permet la mise à jour des parmaètres du module avcec la valeur courante di gradient
    def update_parameters(self,gradient_step):
        pass
    
    #Permet de mettre à jour la valeur courante du gradient par addition
    def backward_update_gradient(self,input,delta_module_suivant):
        pass
    
    #Permet de faire les deux backwar simultanément
    def backward(self,input,delta_module_suivant):
        self.backward_update_gradient(input,delta_module_suivant)
        return self.backward_delte(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        pass
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self, variance):
        pass
    
    

# <markdowncell>

# Nous allons commencer par implémenter le module lineaire classique (sans biais)

# <codecell>

class LinearModule(Module):
    
    #Permet le calcul de la sortie du module
    def __init__(self,entry_size,layer_size):
        self.entry_size = entry_size
        self.layer_size = layer_size
        self.init_gradient()
        self.randomize_parameters()
    
    def forward(self,input):
        return np.dot(self.parameters,input)
    
    #Permet le calcul du gradient des cellules d'entrée
    def backward_delta(self,input,delta_module_suivant):
        return np.dot(input,delta_module_suivant)
        
        
    #Permet d'initialiser le gradient du module
    def init_gradient(self):
        self.gradient = np.zeros(self.layer_size)
        return
    
    #Permet la mise à jour des parmaètres du module avcec la valeur courante di gradient
    def update_parameters(self,gradient_step):
        self.parameters -= self.gradient*gradient_step
        return
    #Permet de mettre à jour la valeur courante du gradient par addition
    def backward_update_gradient(self,input,delta_module_suivant):
        self.gradient += np.dot(input,delta_module_suivant)
        return
    
    #Permet de faire les deux backwar simultanément
    def backward(self,input,delta_module_suivant):
        self.backward_update_gradient(input,delta_module_suivant)
        return self.backward_delta(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        return self.parameters
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self):
        self.parameters = np.random.randn(self.layer_size,self.entry_size)
        return


class TanhModule(Module):
    
    #Permet le calcul de la sortie du module
    def __init__(self,entry_size,layer_size):
        self.entry_size = entry_size
        self.layer_size = layer_size
        self.init_gradient()
        self.randomize_parameters()
    
    def forward(self,input):
        return np.tanh(input)
    
    #Permet le calcul du gradient des cellules d'entrée
    def backward_delta(self,input,delta_module_suivant):
        pass ##TODO
        return np.dot(input,delta_module_suivant)
        
        
    #Permet d'initialiser le gradient du module
    def init_gradient(self):
        self.gradient = np.zeros(self.layer_size)
        return
    
    #Permet la mise à jour des parmaètres du module avcec la valeur courante di gradient
    def update_parameters(self,gradient_step):
        self.parameters -= self.gradient*gradient_step
        return
    #Permet de mettre à jour la valeur courante du gradient par addition
    def backward_update_gradient(self,input,delta_module_suivant):
        self.gradient += np.dot(input,delta_module_suivant)
        return
    
    #Permet de faire les deux backwar simultanément
    def backward(self,input,delta_module_suivant):
        self.backward_update_gradient(input,delta_module_suivant)
        return self.backward_delta(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        return self.parameters
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self):
        self.parameters = np.ones(self.layer_size,self.entry_size)
        return


#multimodule
class MultiModule(Module):
    
    #Permet le calcul de la sortie du module
    def __init__(self,modules,loss):
        self.modules = modules
        self.loss =loss
    
    def forward(self,input):
        
        for module in modules:
            input = module.forward(input)
        return input
    
    #Permet le calcul du gradient des cellules d'entrée
    def backward_delta(self,input,delta_module_suivant):
        for module in modules:
            input = module.forward(input)
        return input
        return np.dot(input,delta_module_suivant)
        
        
    #Permet d'initialiser le gradient du module
    def init_gradient(self):
        self.gradient = np.zeros(self.layer_size)
        return
    
    #Permet la mise à jour des parmaètres du module avcec la valeur courante di gradient
    def update_parameters(self,gradient_step):
        self.parameters -= self.gradient*gradient_step
        return
    #Permet de mettre à jour la valeur courante du gradient par addition
    def backward_update_gradient(self,input,delta_module_suivant):
        self.gradient += np.dot(input,delta_module_suivant)
        return
    
    #Permet de faire les deux backwar simultanément
    def backward(self,input,delta_module_suivant):
        self.backward_update_gradient(input,delta_module_suivant)
        return self.backward_delta(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        return self.parameters
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self):
        self.parameters = np.random.randn(self.layer_size,self.entry_size)
        return

# <markdowncell>

# Test du Système
# ======
# 
# Nous allons maintenant tester notre système (Module Linéaire-> Square Loss) sur un jeu de données classiques (jeu en 2D du TP précédent). Est-ce que ca marche ? 
# Essayez maintenant avec un hinge loss. Est-ce que ca marche?

# <codecell>

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

setGauss = createGaussianDataset(1,2,4,-5,-2,3,100)

# <codecell>

linmod = LinearModule(100,5)
sqloss = SquareLoss()

for i in xrange(0,100):
    x = np.random.rand(100)
    x1 = linmod.forward(x)
    loss = sqloss.getLossValue(x1,np.zeros(5))
    print loss
    bck = sqloss.backward(x1,np.zeros(5))
    linmod.backward(x1,bck)






# <markdowncell>

# Modules Additionnels
# ======
# 
# Nous allons implémenter les modules suivants:
# * Module Tangente Hyperbolic
# * Module Séquentiel
# 
# Nous pouvons maintenant faire des réseaux de neurones ! 

# <codecell>


# <codecell>


