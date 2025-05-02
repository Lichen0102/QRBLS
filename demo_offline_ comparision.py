import csv
import copy
import numpy as np
import warnings
import os
import time
from visualization import plot_acc, plot_macro_f1
from sklearn.metrics import accuracy_score, f1_score
from utils import para_init
from classifier.clf_BLS_QR_timeDependent import BLS_QR_timeDependent
from classifier.clf_BLS_QR import BLS_QR
from utils import get_stream, get_pt
from visualization.plot_comparison import plot_comparison
import sklearn
from skmultiflow.data import WaveformGenerator,SEAGenerator,SineGenerator
warnings.filterwarnings("ignore")
"""  """


#settings
max_samples = 1000  # The range of tested stream
# max_times = 1     # The time for add node
n_round = 3   #Number of run round
n_pt = 1000    #Number of train samples99
theta = 0.15  #Parameter for US
# dataset_name = ["covtype","hyperplane","Faults_SPFD","phishing","SEA_a","SEA","sensor_readings_24","waveform","weather"]
dataset_name = ["waveform_noisy"]
clf_name_list = [ "RVFLNN", "OSELM", "ARF", "IWDA_Multi", "QRBLS", "QRBLS-TDS"]
# num_str = len(str_name_list)
num_clf = len(clf_name_list)

acc_list = [[[] for _ in range(n_round)] for _ in range(num_clf) ]
acc_all = [[] for _ in range(len(dataset_name)) ]
result_path = "./Results/"

if not os.path.exists(result_path):
    os.makedirs(result_path)

directory_path = "./Results/Results_comparision/" 

if not os.path.exists(directory_path):
    os.makedirs(directory_path)


for name_i in range(len(dataset_name)):
    dataset = dataset_name[name_i]
    stream = get_stream(dataset)
    print("-----------Dataset----------- \n", dataset)
    acc_all_list = [[] for _ in range(num_clf) ]
    X_pt, y_pt = stream.next_sample(n_pt)
    X, y = stream.next_sample(max_samples)
    for clf_i in range(num_clf):
        clf_name = clf_name_list[clf_i]
        para_clf = para_init(theta=0.10)
        clf = para_clf.get_clf("clf_" + clf_name)
        acc_all_list[clf_i].append(clf_name)
        print("%s_%d_%d " % (clf_name,n_pt, max_samples))

        #Pretrain
        for i in range(n_round):
            # w = clf.fit(X_pt, y_pt)
            clf.fit(X_pt, y_pt)
            y_pred = np.array(clf.predict(X)).T
            acc_list[clf_i][i] = accuracy_score(y_pred, y)

        print("\nAccuracy %s : %.3f ± %.3f" % (clf_name, np.mean(acc_list[clf_i]), np.std(acc_list[clf_i])))
        acc_all_list[clf_i].append(np.mean(acc_list[clf_i]))
        acc_all_list[clf_i].append(np.std(acc_list[clf_i]))
    acc_results = np.array(acc_all_list)
    result_acc_name = './Results/Results_comparision/acc_comparision_%s.csv' % dataset
    with open(result_acc_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(acc_results)
