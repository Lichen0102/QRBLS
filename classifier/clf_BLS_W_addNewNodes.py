""" PF-BLS-SMW classifier."""

import numpy as np
from sklearn import preprocessing
from numpy import random
import csv
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
        self.seed += 1

    def generator(self, shape, times):
        for i in range(times):
            random.seed(i+self.seed)
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

class BLS_W_addNewNodes:
    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 M1=10,
                 M2=10,
                 E1=1,
                 E2=10,
                 E3=10,
                 map_function='sigmoid',
                 enhence_function='sigmoid',
                 reg=0.001,
                 n_class=0):

        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function
        self._enhence_function = enhence_function
        self._reg = reg
        self._N1 = N1
        self._N2 = N2
        self._M1 = M1
        self._M2 = M2
        self._E1 = E1
        self._E2 = E2
        self._E3 = E3
        self._n_class = n_class

        self.W = 0
        self.pesuedoinverse = 0
        self.mapping_generatorforaddtionalnodes = node_generator()
        self.enhence_generatorforaddtionalnodesOne = node_generator(whiten=True)
        self.enhence_generatorforaddtionalnodesTwo = node_generator(whiten=True)
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []
        self.tempgeneratordic = {}
        self.tempinputdata = []
        self.tempLastFeatureLayer = []
        self.tempTestAddtionalNodes = []
        self.tempTestMappingNodes = []
        self.count = 0

    def updateseed(self):
        self.mapping_generator.updateseed()
        self.enhence_generator.updateseed()

    def fit(self, oridata, orilabel):
        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(orilabel.reshape(-1, 1))
        else:
            label = np.eye(self._n_class)[orilabel]

        mappingdata = self.mapping_generator.generator_nodes(data, self._Nf, self._N1, self._map_function)

        enhencedata = self.enhence_generator.generator_nodes(mappingdata, self._Ne, self._N2, self._enhence_function)
        inputdata = np.column_stack((mappingdata, enhencedata))

        self.pesuedoinverse = self.pinv(inputdata)
        self.W = (self.pesuedoinverse).dot(label)
        
        self.tempinputdata = inputdata.T
        self.tempLastFeatureLayer = mappingdata

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
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata, enhencedata))
        if len(self.tempTestAddtionalNodes) == 0:
            self.tempTestAddtionalNodes = np.empty((inputdata.shape[0],0))
        if len(self.tempTestMappingNodes) == 0:
            self.tempTestMappingNodes = np.empty((inputdata.shape[0],0))
            self.tempTestMappingNodes = np.hstack([self.tempTestMappingNodes,mappingdata])
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

    def partial_fit(self, data, label,mode = "addEnhancementNodes"):

        xdata = self.normalscaler.transform(data)
        if self._n_class == 0:
            xlabel = self.onehotencoder.transform((label).reshape(-1, 1))
        else:
            xlabel = np.zeros([label.shape[0],self._n_class])
            for i in range(label.shape[0]):
                xlabel[i,int(label[i])] = 1  
         
        # 增量增加节点
        if mode == "addNewFeatureNodes":
            addtionalmappingdata = self.mapping_generatorforaddtionalnodes.generator_nodes(xdata, self._M1, self._M2, self._map_function)
            self.tempLastFeatureLayer = np.hstack([self.tempLastFeatureLayer,addtionalmappingdata])
            addtionalenhencedataOne = self.enhence_generatorforaddtionalnodesOne.generator_nodes(self.tempLastFeatureLayer, self._E1, self._E2, self._enhence_function)
            addtionalenhencedataTwo = self.enhence_generatorforaddtionalnodesTwo.generator_nodes(addtionalmappingdata, self._E1, self._E3, self._enhence_function)
            TheAddedNodes = (np.hstack([addtionalmappingdata,addtionalenhencedataOne,addtionalenhencedataTwo])).T
            self.tempgeneratordic["feature"] = self.mapping_generatorforaddtionalnodes
            self.tempgeneratordic["enhence1"] = self.enhence_generatorforaddtionalnodesOne
            self.tempgeneratordic["enhence2"] = self.enhence_generatorforaddtionalnodesTwo
        else:
            addtionalenhencedata = self.enhence_generatorforaddtionalnodesOne.generator_nodes(self.tempLastFeatureLayer, self._E1, self._E2, self._enhence_function)
            TheAddedNodes = addtionalenhencedata.T
            self.tempgeneratordic["enhence1"] = self.enhence_generatorforaddtionalnodesOne
        # 增量更新参数

        D = self.tempinputdata.dot(self.tempinputdata.T)-(self.tempinputdata.dot(TheAddedNodes.T)).dot(np.linalg.inv(TheAddedNodes.dot(TheAddedNodes.T))).dot(TheAddedNodes.dot(self.tempinputdata.T))
        C = -((self.pinv(D).dot(self.tempinputdata)).dot(TheAddedNodes.T).dot(np.linalg.inv(TheAddedNodes.dot(TheAddedNodes.T))))
        B = np.linalg.inv(TheAddedNodes.dot(TheAddedNodes.T)) - (np.linalg.inv(TheAddedNodes.dot(TheAddedNodes.T)).dot(TheAddedNodes)).dot(self.tempinputdata.T.dot(C))
  
        W = np.hstack([self.tempinputdata.T.dot(self.pinv(D))+TheAddedNodes.T.dot(C.T),self.tempinputdata.T.dot(C)+TheAddedNodes.T.dot(B)])
        self.tempinputdata = np.vstack([self.tempinputdata,TheAddedNodes])
        self.W = np.array(W.T.dot(xlabel))



        # self.count += self._E2

        # count_str = str(self.count)

        # PF_BLS_direct_matrix = self.pinv(self.tempinputdata)
        # PF_BLS_direct_array = np.zeros(PF_BLS_direct_matrix.shape)

        # for i in range(PF_BLS_direct_matrix.shape[0]):
        #     for j in range(PF_BLS_direct_matrix.shape[1]):
        #         PF_BLS_direct_array[i,j] = PF_BLS_direct_matrix[i,j]
        # extension_matrix_direct = './Results/Results_matrix/PF-BLS_direct_matrix_%s.csv' % count_str
        # with open(extension_matrix_direct, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(PF_BLS_direct_array)
        
        # PF_BLS_iter_matrix = W.T
        # PF_BLS_iter_array = np.zeros(PF_BLS_iter_matrix.shape)
        # for i in range(PF_BLS_iter_matrix.shape[0]):
        #     for j in range(PF_BLS_iter_matrix.shape[1]):
        #         PF_BLS_iter_array[i,j] = PF_BLS_iter_matrix[i,j]
        # extension_matrix_iter = './Results/Results_matrix/PF-BLS_iter_matrix_%s.csv' % count_str
        # with open(extension_matrix_iter, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(PF_BLS_iter_array)



        direct_matrix = self.pinv(self.tempinputdata).T
        
        iter_matrix = W.T

        return direct_matrix, iter_matrix















