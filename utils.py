from skmultiflow.data import DataStream, WaveformGenerator, SEAGenerator, HyperplaneGenerator
import pandas as pd
import numpy as np
import csv
from sklearn.datasets import fetch_openml

from classifier.clf_BLS_QR_timeDependent import BLS_QR_timeDependent
from classifier.clf_BLS_QR import BLS_QR


def get_stream(name):
    if name == "Jiaolong":
        with open('./datasets/Jiaolong_DSMS_V2.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            X = []
            Y = []
            for row in csvreader:
                data_row = row[:-1]
                label = row[-1]
                X.append(data_row)
                Y.append(label)
            X = np.array(X, dtype=float)
            Y = np.array(Y, dtype=int)
        stream = DataStream(X, Y)
    elif name == "covtype":
        # 下载 Covtype 数据集
        covtype = fetch_openml(data_id=180)
        # 获取数据
        x = covtype.data.to_numpy()
        y = covtype.target
        X = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                X[i,j] = x[i,j]
        num_target = 0
        dic = {}
        for i in y:
            if i not in dic:
                dic[i] = num_target
                num_target += 1
        Y = np.array([dic[i] for i in y])
        stream = DataStream(X, Y)
    elif name == "waveform":
        stream = WaveformGenerator(random_state=1)
    elif name == "waveform_nosiy":
        stream = WaveformGenerator(random_state=1, has_noise=True)
    elif name == "SEA":
        stream = SEAGenerator(random_state=1)
    elif name == "hyperplane":
        stream = HyperplaneGenerator()
    else:
        data = pd.read_csv('./datasets/' + name + '.csv')
        data = data.values
        vol, col = data.shape
        X = data[:, 0:col - 1]
        Y = data[:, col - 1]
        Y = np.array([int(i) for i in Y])
        stream = DataStream(X, Y)
    return stream

def get_pt(stream, n_pt):
    data, labels = stream.next_sample(n_pt)
    return data, labels

class para_init:

    def __init__(self, X_pt_source=np.array([[]]), y_pt_source=np.array([[]]), n_class=2, n_ratio_max=0.2, n_anchor=10, theta=0.2):
        self.n_class = n_class
        self.X_pt_source = X_pt_source
        self.y_pt_source = y_pt_source
        self.n_ratio_max = n_ratio_max
        self.n_anchor = n_anchor
        self.theta = theta

    def get_clf(self, name):
        if name == "clf_QRBLS":
            return BLS_QR(Nf=20,
                     Ne=20,
                     N1=10,
                     N2=10,
                     M1=0,
                     M2=0,
                     E1=1,
                     E2=100,
                     E3=0,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=1e-2)
        elif name == "clf_QRBLS-TDS":
            return BLS_QR_timeDependent(Nf=20,
                     Ne=20,
                     N1=10,
                     N2=10,
                     M1=0,
                     M2=0,
                     E1=1,
                     E2=100,
                     E3=0,
                     ntimeDependent=1,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=1e-2)

