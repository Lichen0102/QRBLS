import csv
import copy
import numpy as np
import warnings
import os
import time
from visualization import plot_acc, plot_macro_f1
from sklearn.metrics import accuracy_score, f1_score
from utils import para_init
from utils import get_stream, get_pt
from visualization.plot_comparison import plot_comparison
import sklearn
from skmultiflow.data import WaveformGenerator, HyperplaneGenerator, SEAGenerator,SineGenerator
warnings.filterwarnings("ignore")
"""  """


#settings
max_samples = 1  # The range of tested stream
max_times = 1000     # The time for add node
n_round = 3   #Number of run round
n_pt = 200    #Number of train samples99
n_ratio_max = 1  #Annotation ratios
theta = 0.15  #Parameter for US
dataset_name_list = ["waveform_noisy"]
# dataset_name_list = ["waveform"]

clf_name_list = [["QRBLS-TDS","QRBLS","BLS-SMW"],
                 ["ARF"],
                 ["OSELM","ARF","IWDA_Multi","BLS-SMW","QRBLS","QRBLS-TDS"]][2]
action = ["parameter","addNewData","addNewEnhancementNodes","addNewFeatureNodes"][1]
extralMethods = ["QRBLS","QRBLS-TDS","BLS","BLS-TDS","RVFLNN","PF-BLS"]


# num_str = len(str_name_list)
num_clf = len(clf_name_list)

acc_list = [[[] for _ in range(n_round)] for _ in range(num_clf) ]
f1_list = [[[] for _ in range(n_round)] for _ in range(num_clf)]
result_path = "./Results/"


if not os.path.exists(result_path):
    os.makedirs(result_path)
#Result Record

for dataset_name in dataset_name_list:

    directory_path = "./Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/" % (dataset_name, n_pt, max_samples*max_times)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    print("-----------Dataset----------- \n", dataset_name)
    list_time = []
    for n_clf in range(len(clf_name_list)):

        clf_name = clf_name_list[n_clf]

        para_clf = para_init(theta=0.10)

        # for n_str in range(1):

        n_annotation_list = []
        y_pred_all = []
        y_true_all = []
        acc_all = []
        t1 = time.time()
        print("{} ".format(clf_name_list[n_clf]))
        for round in range(n_round):
            print('round:', round)
            clf = para_clf.get_clf("clf_" + clf_name)
            y_pred_list = []
            y_true_list = []
            acc_all_list = []

            'stream initialization'
            stream = get_stream(dataset_name)

            X_pt, y_pt = stream.next_sample(n_pt)


            # Setup Hyper-parameters
            count = 0

            #Pretrain
            if clf_name == "DGEBLS":
                clf.fit(X_pt, y_pt, np.eye(X_pt.shape[0]))
            elif clf_name == "QRBLS" or clf_name == "QRBLS-TDS" or clf_name == "BLS-TDS":
                clf.fit(X_pt, y_pt, action = action)
            else:
                clf.fit(X_pt, y_pt)
            # Train the classifier with the samples provided by the data stream
            if action == "addNewData":
                while count < max_times and stream.has_more_samples():
                    count += 1
                    X, y = stream.next_sample(max_samples)
                    y_pred = clf.predict(X)
                    # print(y_pred,y)
                    for i in range(len(y_pred)):
                        y_pred_list.append(y_pred[i])
                        y_true_list.append(y[i])
                    clf.partial_fit(X, y)
                        

            acc_list[n_clf][round] = accuracy_score(y_true_list, y_pred_list)

            y_pred_all = y_pred_all + y_pred_list
            y_true_all = y_true_all + y_true_list
            acc_all = acc_all + acc_all_list

        t2 = time.time()
        if action == "addNewData":
            result_pred = np.array(y_pred_all).reshape(n_round, max_samples*max_times)
            result_true = np.array(y_true_all).reshape(n_round, max_samples*max_times)
        else:
            result_pred = np.array(y_pred_all).reshape(n_round, max_samples*max_times)
            result_true = np.array(y_true_all).reshape(n_round, max_samples*max_times)
            result_acc = np.array(acc_all).reshape(n_round, max_times)
            result_acc_name = './Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/acc_%s.csv' % (dataset_name, n_pt, max_samples*max_times, clf_name_list[n_clf])
            with open(result_acc_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_acc)
        result_pred_name = './Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/Prediction_%s.csv' % (dataset_name, n_pt, max_samples*max_times, clf_name_list[n_clf])
        result_true_name = './Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/True_%s.csv' % (dataset_name, n_pt, max_samples*max_times, clf_name_list[n_clf])
        with open(result_pred_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_pred)
        with open(result_true_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_true)

        list_time_single_method = [clf_name_list[n_clf], np.round((t2 - t1) / n_round,4)]
        list_time.append(list_time_single_method)

        print("\nAccuracy %s : %.3f ± %.3f" % (clf_name_list[n_clf], np.mean(acc_list[n_clf]), np.std(acc_list[n_clf])))
        print("Average Time %s : %.4f s\n" % (clf_name_list[n_clf], (t2 - t1) / n_round))
    result_time_path = './Results/Results_time/'
    if not os.path.exists(result_time_path):
        os.makedirs(result_time_path)

    result_time = './Results/Results_time/time_%s_%s_%s.csv' % (dataset_name, n_pt, max_samples*max_times)
    with open(result_time, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.array(list_time))


