""" SOSELM classifier."""

from sklearn import preprocessing
from numpy import random
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
import numpy as np


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
        self.seed = 0

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

    def updateseed(self):
        self.seed += 10000

    def generator(self, shape, times):
        for i in range(times):
            # random.seed(i+self.seed)
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

class OSELM:
    def __init__(self, Ne, N2, enhence_function, reg, n_class=0):
        self._Ne = Ne
        self._N2 = N2
        self._enhence_function = enhence_function
        self._reg = reg

        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.enhence_generator = node_generator(whiten=True)
        self._n_class = n_class

        self.W = []
        self.K = []
        self.P = []
    def pinv(self, A):
        return np.mat(self._reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)
    
    def updateseed(self):
        self.enhence_generator.updateseed()

    def fit(self, oridata, orilabel):
        
        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(orilabel.reshape(-1, 1))
        else:
            label = np.eye(self._n_class)[orilabel]
        inputdata = self.enhence_generator.generator_nodes(oridata, self._Ne, self._N2, self._enhence_function)

        r, w = inputdata.T.dot(inputdata).shape
        self.pesuedoinverse = np.linalg.inv(inputdata.T.dot(inputdata) + self._reg * np.eye(r))
        self.W = (self.pesuedoinverse.dot(inputdata.T)).dot(label)
        self.K = inputdata.T.dot(inputdata) + self._reg * np.eye(r)
        self.P = np.linalg.inv(self.K)



    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        return self.decode(test_inputdata.dot(self.W))

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return org_prediction

    def transform(self, data):
        inputdata = self.enhence_generator.transform(data)
        return inputdata

    def partial_fit(self, extratraindata, extratrainlabel):

        xdata = self.normalscaler.transform(extratraindata)
        xdata = self.transform(xdata)
        xlabel = self.onehotencoder.transform(extratrainlabel.reshape(-1,1))
        temp = (xdata.dot(self.P)).dot(xdata.T)
        r, w = temp.shape
        self.P = self.P - (((self.P.dot(xdata.T)).dot(np.linalg.inv(np.eye(r) + temp))).dot(xdata)).dot(self.P)
        self.W = self.W + (self.P.dot(xdata.T)).dot(xlabel - xdata.dot(self.W))