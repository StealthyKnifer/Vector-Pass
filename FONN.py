 #TODO add sparse_categorical_entropy loss, add convolutions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
class FONN():
    
    #INITIALIZER
    
    def __init__(self, mini_batch_size = None, classes = None):
        #CONSTANTS
        self.__mini_batch_size = mini_batch_size
        self.__classes = classes
        self.__lossFunc = None
        self.__optimizer = None
        #LISTS
        self.__costs = [ ]
        self.__accuracy = [ ]
        self.__layer_dims = [ ]
        self.__dropouts = [ 1 ]
        self.__activations = [ "None" ]
        #DICTIONARIES
        self.__params = { }
        self.__nOuts = { }
        self.__pOuts = { }
        self.__pDerivs = { }
    #USER UNSUABLE FUNCTIONS
    
    #EXCEPTIONS REQUIRED FOR THE CLASS
    class NotNumpyArraysError(Exception):
        def __init__(self):
            self.message = "Either of X and Y is not a numpy array."
            super().__init__(self.message)
    class ActivationError(Exception):
        def __init__(self):
            self.message = "Activation is not defined."
            super().__init__(self.message)
    class ModelNotFitError(Exception):
        def __init__(self):
            self.message = "Please fit the model before adding a layer."
            super().__init__(self.message)
    class ModelNotCompiledError(Exception):
        def __init__(self):
            self.message = "Please compile the model before training."
            super().__init__(self.message)
    #MINI BATCH CREATOR
    def __create_mini_batches(self, X, Y): 
        minibatch = [ ]
        mbatch_size = self.__mini_batch_size
        m = X.shape[ 1 ]
        num_mini_batches = math.floor( m / mbatch_size)
        perms = np.random.permutation(m)
        shuff_X = X[ :, perms ]
        shudd_Y = Y[ :, perms ]
        for num in range(num_mini_batches):
            m_X = X[ :, num * mbatch_size: (num + 1) * mbatch_size ]
            m_Y = Y[ :, num * mbatch_size: (num + 1) * mbatch_size ]
            mbatch = ( m_X, m_Y )
            minibatch.append(mbatch)
        if(m % num_mini_batches != 0):
            m_X = X[ :, num_mini_batches * mbatch_size: ]
            m_Y = Y[ :, num_mini_batches * mbatch_size: ]
            mbatch = ( m_X, m_Y )
            minibatch.append(mbatch)
        return minibatch
    
    #ONE HOT ENCODER FOR Y
    def __runOneHot(self, Y): #TODO remove this altogether and make user use sparse categorical crossentropy
        classes = self.__classes
        modY = np.zeros(( classes, Y.shape[1] ))
        Y = Y.astype('object')
        for m in range(Y.shape[1]):
            modY[Y[0,m], m] = 1
        return modY
    
    #ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
    def __G(self, z, activation):
        if(activation == "sigmoid"):
            return 1 / (1 + np.exp(-z))
        if(activation == "relu"):
            return np.maximum(0, z)
        if(activation == "softmax"):
            return (np.exp(z)) / (np.sum(np.exp(z), axis=0))
    def __GD(self, z, activation):
        if(activation == "sigmoid"):
            sig = self.__G(z, "sigmoid")
            return sig * (1 - sig)
        if(activation == "relu"):
            return np.where(z <= 0, 0, 1)
        if(activation == "softmax"):
            soft = self.__G(z, "softmax")
            return soft * (1 - soft)
        
    #FEED FORWARD FUNCTION
    def __fForward(self, X, params, nOuts, activations, dropout):
        l = len(self.__layer_dims)
        nOuts['A0'] = X
        m = X.shape[1]
        for i in range(1, l):
            W = params[f'W{i}']
            b = params[f'b{i}']
            A = nOuts[f'A{i-1}']
            activation = activations[i]
            Z = np.dot(W,A) + b
            A1 = self.__G(Z, activation)
            keep_prob = dropout[i]
            if(keep_prob!=1):
                D1 = np.random.rand(A1.shape[0], A1.shape[1])
                D1 = D1 < keep_prob
                A1 = A1 * D1
                A1 = A1 / keep_prob
                nOuts[f'D{i}'] = D1
            nOuts[f'Z{i}'] = Z
            nOuts[f'A{i}'] = A1
    
    #LOSS AND ACCURACY CALCULATOR
    def __calLoss(self, Y, nOuts, lossFunc): #TODO make this better
        l = len(self.__layer_dims)
        A = nOuts[f'A{l-1}']
        m = Y.shape[1]
        if(lossFunc == "binary_crossentropy"):
            loss = ((-1/m)*np.sum(np.multiply(Y,np.log(A)) + np.multiply(1-Y,np.log(1-A))))
            acc = ((np.sum(A.round()==Y)))*100
        if(lossFunc == "categorical_crossentropy"):
            acc=0
            for i in range(m):
                acc = acc + (A[:,i].round()==Y[:,i]).all()
            acc *=100
            loss = ((-1/m)*np.sum(np.multiply(Y,np.log(A))))
        return loss,acc
        
    #BACKPROPAGATION FUNCTION
    def __bPropagation(self, Y, nOuts, params, pDeriv, lossFunc, dropout): 
        l = len(self.__layer_dims)
        A = nOuts[f'A{l-1}']        
        m = Y.shape[1]
        dal = A - Y
        for i in reversed(range(1,l)):
            Z = nOuts[f'Z{i}']
            W = params[f'W{i}']
            b = params[f'b{i}']
            keep_prob = dropout[i-1]
            A_prev = nOuts[f'A{i-1}']
            activation = self.__activations[i]
            dz = dal * self.__GD(Z,activation)
            dW = (1/m)*np.dot(dz,A_prev.T)
            db = (1/m)*np.sum(dz,axis=1,keepdims=True)
            dal = np.dot(W.T,dz)
            if(keep_prob!=1):
                D = nOuts[f'D{i-1}']
                dal = dal * D
                dal = dal / dropout[i]
            pDeriv[f'dW{i}'] = dW
            pDeriv[f'db{i}'] = db
    
    #PARAMETER UPDATER FUNCTION
    def __updateParams(self,params,pDeriv,optimizer,lr,t,epsilon,beta1,beta2): #TODO make this cleaner
        l = len(self.__layer_dims)
        if(optimizer == "gradient_descent"):
            for i in range(1,l):
                params[f'W{i}'] = params[f'W{i}'] - lr * pDeriv[f'dW{i}']
                params[f'b{i}'] = params[f'b{i}'] - lr * pDeriv[f'db{i}'] 
        if(optimizer == "gradient_descent_momentum"):
            for i in range(1,l):
                params[f'VW{i}'] = beta1 * params[f'VW{i}'] + (1-beta1) * pDeriv[f'dW{i}']
                params[f'Vb{i}'] = beta1 * params[f'Vb{i}'] + (1-beta1) * pDeriv[f'db{i}']     
                params[f'W{i}'] = params[f'W{i}'] - lr * params[f'VW{i}']
                params[f'b{i}'] = params[f'b{i}'] - lr * params[f'Vb{i}']
        if(optimizer == "RMSProp"):
            for i in range(1,l):
                dW = pDeriv[f'dW{i}']
                db = pDeriv[f'db{i}']
                SW = params[f'SW{i}']
                Sb = params[f'Sb{i}']
                
                SW = beta1 * SW + (1 - beta1) * dW * dW
                Sb = beta1 * Sb + (1 - beta1) * db * db
                
                params[f'SW{i}'] = SW/(1-beta1**t)
                params[f'Sb{i}'] = Sb/(1-beta1**t)
                
                params[f"W{i}"] -= lr * dW/np.sqrt(SW + epsilon)
                params[f"b{i}"] -= lr * db/np.sqrt(Sb + epsilon)
        if(optimizer == "Adam"):
            for i in range(1,l):
                dW = pDeriv[f'dW{i}']
                db = pDeriv[f'db{i}']
                SW = params[f'SW{i}']
                Sb = params[f'Sb{i}']
                VW = params[f'SW{i}']
                Vb = params[f'Sb{i}']
                
                SW = beta1 * SW + (1 - beta1) * dW * dW
                Sb = beta1 * Sb + (1 - beta1) * db * db
                
                params[f'SW{i}'] = SW/(1-beta1**t)
                params[f'Sb{i}'] = Sb/(1-beta1**t)
                
                params[f'VW{i}'] = beta1 * params[f'VW{i}'] + (1-beta1) * pDeriv[f'dW{i}']
                params[f'Vb{i}'] = beta1 * params[f'Vb{i}'] + (1-beta1) * pDeriv[f'db{i}']     
                params[f'W{i}'] = params[f'W{i}'] - lr * params[f'VW{i}']/np.sqrt(SW + epsilon)
                params[f'b{i}'] = params[f'b{i}'] - lr * params[f'Vb{i}']/np.sqrt(Sb + epsilon)
                
            
        
    #USER USABLE FUNCTIONS
    
    #FITTING THE MODEL TO INPUTS AND OUTPUTS
    def fit(self,X,Y,onehot): 
        
        if(type(X) != np.ndarray or type(Y) != np.ndarray):
            raise self.__NotNumpyArraysError
        if(onehot==True):
            Y = self.__runOneHot(Y)
        if(self.__mini_batch_size==None):
            batch = [(X,Y)]
        else:
            batch = self.__create_mini_batches(X,Y) 
        self.__batch = batch
        self.__layer_dims.append(X.shape[0])
        self.__X = X
        self.__Y = Y
    
    #COMPILING THE MODEL
    def compile(self,lossFunc,optimizer,lr=0.03,beta1=0.9,beta2=0.999,epsilon=1e-8): #TODO add exceptions for unknown loss and
                                                                                     #and optimizers
        self.__lossFunc = lossFunc
        self.__optimizer = optimizer
        self.__lr = lr
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
    #ADDING LAYERS TO MODEL    
    def addLayer(self,layer_dim,activator,dropout=0,layer_type=None): #TODO separate dropout as a layer
        if(activator not in "sigmoid relu softmax".split(" ")):
            raise self.__ActivationError
        else:
            if(len(self.__layer_dims)==0):
                raise self.__ModelNotFitError
            else:
                l = len(self.__layer_dims)
                self.__layer_dims.append(layer_dim)
                self.__activations.append(activator)
                self.__dropouts.append(1-dropout)
                nH = layer_dim
                nHm1 = self.__layer_dims[l-1]
                self.__params[f'W{l}'] = np.random.randn(nH,nHm1)*np.sqrt(2/nHm1)
                self.__params[f'b{l}'] = np.zeros((nH,1))
                self.__params[f'SW{l}'] = np.zeros(self.__params[f'W{l}'].shape)
                self.__params[f'Sb{l}'] = np.zeros(self.__params[f'b{l}'].shape)
                self.__params[f'VW{l}'] = np.zeros(self.__params[f'W{l}'].shape)
                self.__params[f'Vb{l}'] = np.zeros(self.__params[f'b{l}'].shape)
                
    #TRAINING THE MODEL
    def train(self,epochs,getCost=False): #TODO remove get cost and always return a cache containing all data
        if(self.__lossFunc == None): 
            raise self.__ModelNotCompiledError
        pDeriv = self.__pDerivs
        params = self.__params
        nOuts = self.__nOuts
        epsilon = self.__epsilon
        beta1 = self.__beta1
        beta2 = self.__beta2 
        getCost = getCost
        optimizer = self.__optimizer
        activations = self.__activations
        dropouts = self.__dropouts
        lr = self.__lr
        lossFunc = self.__lossFunc
        t = 0
        for epoch in range(epochs+1): #TODO make this code a bit better
            costtotal = 0
            acctotal = 0
            for batch in self.__batch:
                X = batch[0]
                Y = batch[1]
                self.__fForward(X,params,nOuts,activations,dropouts)
                loss,acc = self.__calLoss(Y,nOuts,lossFunc)
                costtotal = costtotal + loss
                acctotal = acctotal + acc
                self.__bPropagation(Y,nOuts,params,pDeriv,lossFunc,dropouts)
                t += 1
                self.__updateParams(params,pDeriv,optimizer,lr,t,epsilon,beta1,beta2)
            self.__costs.append(costtotal/self.__X.shape[1])
            self.__accuracy.append(acctotal/self.__X.shape[1])
            if(epoch%10 == 0):
                print(f"Loss after epoch {epoch} is {self.__costs[-1]} and accuracy is {self.__accuracy[-1]}.")
        if(getCost):
            return self.__costs,self.__accuracy
    
    #PREDICTING FOR TEST SET/DEV SET
    def predict(self,X,Y,onehot=False): #TODO separate predict and evaluate functions
        l = len(self.__layer_dims)
        if(onehot):
            Y = self.__runOneHot(Y)
        pOuts = self.__pOuts
        params = self.__params
        dropouts = self.__dropouts
        activations = self.__activations
        self.__fForward(X,params,pOuts,activations,dropouts)
        A = pOuts[f'A{l-1}']
        if(self.__lossFunc == "categorical_crossentropy"):
            acc = 0
            for i in range(Y.shape[1]):
                acc += (A[:,i].round() == Y[:,i]).all() 
            acc /= Y.shape[1]
        else:
            acc = ((np.sum(A.round()==Y)))/Y.shape[1]
        print(f"Accuracy on the test set is {acc*100}.")

    #PLOTS TWO SEPARATE GRAPHS FOR ACCURACY AND COSTW
    def plotCostAndAccuracy(self): #TODO add a bool if a user wants to plot after training directly
        plt.plot(self.__costs)
        plt.xlabel("EPOCHS")
        plt.ylabel("COST")
        plt.title("COST vs EPOCH GRAPH")
        plt.show()
        plt.plot(self.__accuracy)
        plt.xlabel("EPOCHS")
        plt.ylabel("ACCURACY")
        plt.title("ACCURACY vs EPOCH GRAPH")
        plt.show()
#© Aditya Rangarajan 2020