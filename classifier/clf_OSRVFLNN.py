""" OSRVFL classifier."""

import numpy as np
from sklearn import preprocessing
from numpy import random
import time
class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 1e-6)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 1e-6)

class node_generator:
    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        return np.maximum(data, 0)
    
    def tansig(self,data):
        return (2/(1+np.exp(-2*data)))-1

    def orth(self, W):

        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu,
                          'tansig': self.tansig
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb

class OSRVFLNN:
    def __init__(self,
                 Ne=10,
                 N2=10,
                 E1=1,
                 E2=10,
                 enhence_function='sigmoid',
                 reg=0.001,
                 n_class=0):


        self._Ne = Ne
        self._enhence_function = enhence_function
        self._reg = reg
        self._N2 = N2
        self._E1 = E1
        self._E2 = E2
        self._n_class = n_class

        self.W = 0
        self.pesuedoinverse = 0
        self.K = 0
        self.P = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.enhence_generator = node_generator(whiten=True)
        self.enhence_generatorforaddtionalnodesOne = node_generator(whiten=True)
        self.tempTestMappingNodes = []
        self.tempgeneratordic = {}
        self.tempTestAddtionalNodes = []
        self.tempinputdata = []

    def fit(self, oridata, orilabel):
        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(np.matrix(orilabel).T)
        else:
            label = np.eye(self._n_class)[orilabel]

        enhencedata = self.enhence_generator.generator_nodes(data, self._Ne, self._N2, self._enhence_function)
        inputdata = np.column_stack((data, enhencedata))

        self.pesuedoinverse = self.pinv(inputdata)
        self.W = self.pesuedoinverse.dot(label)
        # self.K = inputdata.T.dot(inputdata) + self._reg * np.eye(r)
        self.P = self.pinv(inputdata)
        self.tempdata = data
        self.tempinputdata = inputdata
        self.K = inputdata.T.dot(inputdata)

    def softmax_norm(self, array):
        exp_array = np.matrix(np.exp(array))
        sum_exp_array = np.sum(exp_array, axis=1)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def pinv(self, A):
        return np.mat(self._reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        return self.decode(self.softmax_norm(logit))


    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return self.softmax_norm(org_prediction)

    def transform(self, data):
        enhencedata = self.enhence_generator.transform(data)
        inputdata = np.column_stack((data, enhencedata))
        if len(self.tempTestAddtionalNodes) == 0:
            self.tempTestAddtionalNodes = np.empty((inputdata.shape[0],0))
        if len(self.tempTestMappingNodes) == 0:
            self.tempTestMappingNodes = np.empty((inputdata.shape[0],0))
            self.tempTestMappingNodes = np.hstack([self.tempTestMappingNodes,data])
        if len(self.tempgeneratordic) > 0 :
            tempadditionalnodes = np.empty((inputdata.shape[0],0))
            for key,value in self.tempgeneratordic.items():
                if key == "feature":
                    addtionalmappingdata = value.transform(data)
                    self.tempTestMappingNodes = np.hstack([self.tempTestMappingNodes,addtionalmappingdata])
                    tempadditionalnodes = np.hstack([tempadditionalnodes,addtionalmappingdata])
                if key == "enhence1":
                    addtionalenhancenodes1 = value.transform(self.tempTestMappingNodes)
                    tempadditionalnodes = np.hstack([tempadditionalnodes,addtionalenhancenodes1])
                if key == "enhence2":
                    addtionalenhancenodes2 = value.transform(addtionalmappingdata)
                    tempadditionalnodes = np.hstack([tempadditionalnodes,addtionalenhancenodes2])
            self.tempTestAddtionalNodes = np.hstack([self.tempTestAddtionalNodes,tempadditionalnodes])
            inputdata = np.hstack([inputdata,self.tempTestAddtionalNodes])
    
        return inputdata
    
    def predictionresult(self,V):
        prediction = V.dot(self.W)
        return prediction
    
    def partial_fit(self, data, label):
        xdata = self.normalscaler.transform(data)
        xlabel = self.onehotencoder.transform(np.mat(label).T) if self._n_class == 0 else np.eye(self._n_class)[label]
           
        addtionalenhencedata = self.enhence_generatorforaddtionalnodesOne.generator_nodes(xdata, self._E1, self._E2, self._enhence_function)
        TheAddedNodes = addtionalenhencedata
        self.tempgeneratordic["enhence1"] = self.enhence_generatorforaddtionalnodesOne
        
        K = TheAddedNodes.T.dot(TheAddedNodes)
        self.K = self.K + K
        self.W = self.W + (np.linalg.inv(self.K).dot(TheAddedNodes.T)).dot(xlabel - TheAddedNodes.dot(self.W))
            
                
















