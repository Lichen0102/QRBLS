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
from skmultiflow.data import WaveformGenerator, HyperplaneGenerator, SEAGenerator
warnings.filterwarnings("ignore")
"""  """
def download_matrxi_data(iter_matrix, path):
    iter_array = np.zeros(iter_matrix.shape)
    for i in range(iter_matrix.shape[0]):
        for j in range(iter_matrix.shape[1]):
            iter_array[i,j] = iter_matrix[i,j]
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(iter_array)

#settings
max_samples = 1000  # The range of tested stream
max_times = 25     # The time for add node 
n_round = 1   #Number of run round
n_pt = 1000    #Number of train samples99
n_ratio_max = 1  #Annotation ratios
theta = 0.15  #Parameter for US
dataset_names = ["Faults_SPFD","sensor_readings_24","covtype","phishing","weather","waveform","hyperplane"] #addNewEnhancementNodes
# dataset_names = ["sensor_readings_24","hyperplane","waveform","covtype"] #addNewFeatureNodes
clf_name_list = [["BLS","PF-BLS","RVFLNN","QRBLS-TDS"],
                 ["BLS","BLS-TDS","BLS-TDS"],
                 ["BLS","PF-BLS","RVFLNN","QRBLS","QRBLS-TDS"]][2]
action = ["parameter","addNewData","addNewEnhancementNodes","addNewFeatureNodes"][2]
extralMethods = ["QRBLS","QRBLS-TDS","BLS","BLS-TDS","RVFLNN","PF-BLS"]

for dataset_name in dataset_names:
    # num_str = len(str_name_list)
    num_clf = len(clf_name_list)

    acc_list = [[[] for _ in range(n_round)] for _ in range(num_clf) ]
    f1_list = [[[] for _ in range(n_round)] for _ in range(num_clf)]
    result_path = "./Results/"

    if dataset_name == "waveform_noisy":
        stream_pt = WaveformGenerator(has_noise=True)
    else:
        stream_pt = get_stream(dataset_name)
    # stream_pt = SEAGenerator()

    X_pt_source, y_pt_source = get_pt(stream=stream_pt, n_pt=n_pt)



    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #Result Record
    directory_path = "./Results/Results_acc/Results_addnewnodes/Results_%s_%d_%d/" % (dataset_name, n_pt, max_samples)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    directory_path = "./Results/Results_matrix/"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # n_end = X_pt_source.shape[0]
    print("-----------Dataset----------- \n", dataset_name)
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
            # stream = SEAGenerator()
            if dataset_name == "waveform_noisy":
                stream = WaveformGenerator(has_noise=True)
            else:
                stream = get_stream(dataset_name)

            X_pt, y_pt = copy.deepcopy(X_pt_source), copy.deepcopy(y_pt_source)


            # Setup Hyper-parameters
            count = 0

            #Pretrain
            if clf_name == "DGEBLS":
                clf.fit(X_pt, y_pt, np.eye(X_pt.shape[0]))
            elif clf_name == "QRBLS" or clf_name == "QRBLS-TDS":
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
                    if clf_name in extralMethods:
                        clf.partial_fit(X, y, mode = action)
                    else:
                        clf.partial_fit(X, y)
                        
            else:
                X, y  = stream.next_sample(max_samples)
                while count < max_times:
                    count += 1
                    y_pred = clf.predict(X)
                    for i in range(X.shape[0]):
                        y_pred_list.append(y_pred[i])
                        y_true_list.append(y[i])
                    if clf_name in extralMethods:
                        matrix_direct, matrix_iter = clf.partial_fit(X_pt, y_pt, mode = action)
                    else:
                        clf.partial_fit(X_pt, y_pt)

                    extension_matrix_direct = './Results/Results_matrix/extension_matrix_direct_%s_%s_%s.csv' % (clf_name, count*100, dataset_name)
                    extension_matrix_iter = './Results/Results_matrix/extension_matrix_iter_%s_%s_%s.csv' % (clf_name, count*100, dataset_name)
                    download_matrxi_data(matrix_direct, extension_matrix_direct)
                    download_matrxi_data(matrix_iter, extension_matrix_iter)

                    print(accuracy_score(y_pred, y))
                    acc_all_list.append(accuracy_score(y_pred, y))
                if action != "addNewData":
                    acc_all_list.append(accuracy_score(y_pred, y))

            acc_list[n_clf][round] = accuracy_score(y_true_list, y_pred_list)


            y_pred_all = y_pred_all + y_pred_list
            y_true_all = y_true_all + y_true_list
            acc_all = acc_all + acc_all_list

        t2 = time.time()
        if action == "addNewData":
            result_pred = np.array(y_pred_all).reshape(n_round, max_samples*max_times)
            result_true = np.array(y_true_all).reshape(n_round, max_samples*max_times)
        else:
            result_pred = np.array(y_pred_all).reshape(n_round, max_samples*(max_times))
            result_true = np.array(y_true_all).reshape(n_round, max_samples*(max_times))
            result_acc = np.array(acc_all).reshape(n_round, max_times+1)
            result_acc_name = './Results/Results_acc/Results_addnewnodes/Results_%s_%d_%d/acc_%s.csv' % (dataset_name, n_pt, max_samples, clf_name_list[n_clf])
            with open(result_acc_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_acc)
        result_pred_name = './Results/Results_acc/Results_addnewnodes/Results_%s_%d_%d/Prediction_%s.csv' % (dataset_name, n_pt, max_samples, clf_name_list[n_clf])
        result_true_name = './Results/Results_acc/Results_addnewnodes/Results_%s_%d_%d/True_%s.csv' % (dataset_name, n_pt, max_samples, clf_name_list[n_clf])
        with open(result_pred_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_pred)
        with open(result_true_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_true)

        

        print("\nAccuracy %s : %.3f ± %.3f" % (clf_name_list[n_clf], np.mean(acc_list[n_clf]), np.std(acc_list[n_clf])))
        print("Average Time %s : %.4f s\n" % (clf_name_list[n_clf], (t2 - t1) / n_round))

