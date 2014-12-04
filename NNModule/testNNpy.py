#testNNpy.py
#-*- coding: utf-8 -*-

from NNPy import *
from DataClass import *

def perceptron(inDim,outDim):
    #Construction d'un Perceptron
    lm = LinearModule(inDim,outDim)
    hl = HingeLoss()
    hm = HorizontalModule([lm])

    #Perceptron
    return NetworkModule([hm],hl)

def multiLayerPerceptron(inDim,hidden,out):
   return NetworkModule([HorizontalModule([LinearModule(inDim,size),TanhModule(size,size)]),HorizontalModule([LinearModule(size,outDim)])],SquareLoss())

def multiLayerPerceptronDropout(inDim,outDim,size):
    lm = LinearModule(inDim,size)
    return (NetworkModule([HorizontalModule([DropoutModule(1,1),lm,TanhModule(size,size)])],SquareLoss()),lm)

print("----======  Linear Separable Points  =======----")

#dataset
dataset = createGaussianDataset(1,2,4,-5,-2,3,200)


trainV = []
trainL = []
testV = []
testL = []

for i,vect in enumerate(dataset.x):
	if i < len(dataset.x)/2:
		trainV.append(vect)
		trainL.append(dataset.y[i])
	else:
		testV.append(vect)
		testL.append(dataset.y[i])


print("----Perceptron----")
NBITER = 10
network = perceptron(2,1)

print("=======TRAIN ERROR=======")
for i in xrange(0,NBITER):
    network.stochasticIter(trainV, trainL,gradient_step=0.0001, verbose=False)
    predicted = network.forwardAll(trainV)
    ok=0
    ko=0
    for pred,exp in zip(predicted,trainL):
        if pred[0]*exp[0] > 0:
            ok+=1
        else:
            ko+=1

    print("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
print("Learning done")
    
print("=======TEST ERROR=======")  
predicted = network.forwardAll(testV)

ok=0
ko=0

for pred,exp in zip(predicted,testL):
    if pred[0]*exp[0] > 0:
        ok+=1
    else:
        ko+=1
print("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
	
print('----multiLayerPerceptron-----')
NBITER = 10
network = multiLayerPerceptron(2,1,1)

print("=======TRAIN ERROR=======")
for i in xrange(0,NBITER):
    network.stochasticIter(trainV, trainL,gradient_step=0.0001, verbose=False)
    predicted = network.forwardAll(trainV)
    ok=0
    ko=0
    for pred,exp in zip(predicted,trainL):
        if pred[0]*exp[0] > 0:
            ok+=1
        else:
            ko+=1

    print("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
print("Learning done")
    
print("=======TEST ERROR=======" )
predicted = network.forwardAll(testV)

ok=0
ko=0

for pred,exp in zip(predicted,testL):
    if pred[0]*exp[0] > 0:
        ok+=1
    else:
        ko+=1
print("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))


print("----======  MNIST 8/6  =======----")

trainV,trainL,testV,testL = getMnistDualDataset()

print("----Perceptron----")
NBITER = 10
network = perceptron(28*28,1)

print("=======TRAIN ERROR=======")
for i in xrange(0,NBITER):
    network.stochasticIter(trainV, trainL,gradient_step=0.0001, verbose=False)
    predicted = network.forwardAll(trainV)
    ok=0
    ko=0
    for pred,exp in zip(predicted,trainL):
        if pred[0]*exp[0] > 0:
            ok+=1
        else:
            ko+=1

    print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
print ("Learning done")
    
print ("=======TEST ERROR=======")
predicted = network.forwardAll(testV)

ok=0
ko=0

for pred,exp in zip(predicted,testL):
    if pred[0]*exp[0] > 0:
        ok+=1
    else:
        ko+=1
print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
    
print ('----multiLayerPerceptron-----')
NBITER = 10
network = multiLayerPerceptron(28*28,1,100)

print ("=======TRAIN ERROR=======")
for i in xrange(0,NBITER):
    network.stochasticIter(trainV, trainL,gradient_step=0.0001, verbose=False)
    predicted = network.forwardAll(trainV)
    ok=0
    ko=0
    for pred,exp in zip(predicted,trainL):
        if pred[0]*exp[0] > 0:
            ok+=1
        else:
            ko+=1

    print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
print ("Learning done")
    
print ("=======TEST ERROR=======")  
predicted = network.forwardAll(testV)

ok=0
ko=0

for pred,exp in zip(predicted,testL):
    if pred[0]*exp[0] > 0:
        ok+=1
    else:
        ko+=1
print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))


print ('----multiLayerPerceptron-----')
NBITER = 10

network = multiLayerPerceptron(28*28,1,100)

print ("=======TRAIN ERROR=======")
for i in xrange(0,NBITER):
    network.stochasticIter(trainV, trainL,gradient_step=0.0001, verbose=False)
    predicted = network.forwardAll(trainV)
    ok=0
    ko=0
    for pred,exp in zip(predicted,trainL):
        if pred[0]*exp[0] > 0:
            ok+=1
        else:
            ko+=1

    print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))
print ("Learning done")
    
print ("=======TEST ERROR=======")  
predicted = network.forwardAll(testV)

ok=0
ko=0

for pred,exp in zip(predicted,testL):
    if pred[0]*exp[0] > 0:
        ok+=1
    else:
        ko+=1
print ("%d correct (%f%%), %d incorrect (%f%%) " % (ok,ok/(ok+ko+0.0)*100,ko,ko/(ok+ko+0.0)*100))