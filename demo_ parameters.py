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
from skmultiflow.data import WaveformGenerator
warnings.filterwarnings("ignore")
"""  """


#settings
max_samples = 1000  # The range of tested stream
max_times = 1     # The time for add node 
n_round = 1   #Number of run round
n_pt = 1000    #Number of train samples99
n_ratio_max = 1  #Annotation ratios
theta = 0.15  #Parameter for US
dataset_names = ["Faults_SPFD"]
clf_name_list = ["QRBLS","QRBLS-TDS"]

for dataset_name in dataset_names:

    # num_str = len(str_name_list)
    num_clf = len(clf_name_list)


    acc_list = [[[] for _ in range(n_round)] for _ in range(num_clf) ]

    result_path = "./Results/"

    stream_pt = get_stream(dataset_name)


    X_pt_source, y_pt_source = stream_pt.next_sample(n_pt)

    Nfs = [10,20,30,40,50,60,70,80,90,100]
    Nes = [10,20,30,40,50,60,70,80,90,100]

    n_end = X_pt_source.shape[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #Result Record
    directory_path = "./Results/Results_parameters/Results_%s_%d_%d/" % (dataset_name, n_pt, max_samples*max_times)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    print("-----------Dataset----------- \n", dataset_name)

    acc_all_list = [[] for _ in range(num_clf) ]
    X_pt, y_pt = copy.deepcopy(X_pt_source), copy.deepcopy(y_pt_source)

    X, y = X_pt, y_pt

    for nf in Nfs:
        for ne in Nes:
            clf_list =  [BLS_QR(Nf=nf, Ne=ne,N1=10,N2=10,M1=1,M2=20,E1=1,E2=100,E3=20,map_function='sigmoid',enhence_function='sigmoid',reg=0.001),
                BLS_QR_timeDependent(Nf=nf, Ne=ne,N1=10,N2=10,M1=1,M2=20,E1=1,E2=100,E3=20,ntimeDependent=1,map_function='sigmoid',enhence_function='sigmoid',reg=0.001)]
            for clf_i in range(len(clf_list)):
                clf = clf_list[clf_i]
                clf_name = clf_name_list[clf_i]
                print("%s_%d_%d " % (clf_name,nf,ne))
                #Pretrain
                for i in range(n_round):
                    clf.fit(X_pt, y_pt)
                    y_pred = clf.predict(X)
                    acc_list[clf_i][i] = accuracy_score(y_pred, y)
                print("\nAccuracy %s : %.3f ± %.3f" % (clf_name_list[clf_i], np.mean(acc_list[clf_i]), np.std(acc_list[clf_i])))
                acc_all_list[clf_i].append(np.mean(acc_list[clf_i]))
    acc_results = np.array(acc_all_list)
    result_acc_name = './Results/Results_parameters/Results_%s_%d_%d/acc_parameter_%s.csv' % (dataset_name, n_pt, max_samples*max_times,dataset_name)
    with open(result_acc_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(acc_results)



