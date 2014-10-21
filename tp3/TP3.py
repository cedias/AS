#!/usr/bin/python
#-*- coding: utf-8 -*-           


import numpy as np
import tools as tt
       
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
        return np.max(np.zeros(predicted_output.size), -desired_output*predicted_output)
    
    def backward(self, predicted_output,desired_output):
        res = np.zeros(desired_output.size)
        prod = -desired_output*predicted_output
        index = np.where(prod >=0 )
        res[index] = -desired_output[index]

        return res

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
        return np.dot(self.parameters, delta_module_suivant)  
        
    #Permet d'initialiser le gradient du module
    def init_gradient(self):
        self.gradient = np.zeros((self.entry_size,self.layer_size))
        return
    
    #Permet la mise à jour des parmaètres du module avcec la valeur courante di gradient
    def update_parameters(self,gradient_step):
        self.parameters -= self.gradient*gradient_step
        return

    #Permet de mettre à jour la valeur courante du gradient par addition
    def backward_update_gradient(self,input,delta_module_suivant):
        newGrad = np.zeros((self.entry_size,self.layer_size))
        print delta_module_suivant.shape
        for i in xrange(0,self.layer_size):
            di = delta_module_suivant[i]
            newGrad[i,:] = (di*input)

        self.gradient += newGrad
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
        return self.backward_delta(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        pass
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self):
        pass


class logisticModule(Module):
    
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
        return self.backward_delta(input,delta_module_suivant)

    #Retourne les paramètres du module
    def get_parameters(self):
        pass
    
    #Initialize aléatoirement les paramètres du module
    def randomize_parameters(self):
        pass



#multimodule
class MultiModule():
    
    #Permet le calcul de la sortie du module
    def __init__(self,modules,loss):
        self.modules = modules
        self.loss = loss

    
    def forward(self,input):
        self.inputs = []
        for module in self.modules:
            self.inputs.append(input)
            input = module.forward(input)
        return input
    
    #Permet le calcul du gradient des cellules d'entrée
    def backward(self,predicted,wanted,batch=False,gradient_step=0.001):
        loss_delta = self.loss.backward(predicted,wanted)
        for module,input in zip(reversed(self.modules),reversed(self.inputs)):
            loss_delta = module.backward(input,loss_delta)

            if not batch:
                module.backward_update_parameters(gradient_step)


        return

    def update_parameters(gradient_step):
        for module in self.modules:
            module.update_parameters(gradient_step)
        return
        
    

# Test du Système
# ======


linmod = LinearModule(100,5)
sqloss = SquareLoss()

for i in xrange(0,100):
    x = np.random.rand(100)
    x1 = linmod.forward(x)
    loss = sqloss.getLossValue(x1,np.zeros(5))
    bck = sqloss.backward(x1,np.zeros(5))
    print bck
    bb = linmod.backward(x,bck)


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


