import csv
import copy
import numpy as np
import warnings
import os
import time
from sklearn.metrics import accuracy_score
from utils import para_init
from utils import get_stream, get_pt
warnings.filterwarnings("ignore")
"""  """

#settings
max_samples = 1  # The range of tested stream
max_times = 25     # The time for add node or sample
test_samples = 1000   # The number for add node or sample
n_round = 1   #Number of run round
n_pt = 1000    #Number of train samples99
n_ratio_max = 1  #Annotation ratios
theta = 0.15  #Parameter for US
dataset_names = ["hyperplane"] #addNewEnhancementNodes
clf_name_list = ["QRBLS","QRBLS-TDS"]
action = ["addNewData","addNewNodes"][0]

for dataset_name in dataset_names:
    # num_str = len(str_name_list)
    num_clf = len(clf_name_list)

    acc_list = [[[] for _ in range(n_round)] for _ in range(num_clf) ]

    result_path = "./Results/"

    stream_pt = get_stream(dataset_name)
    # stream_pt = SEAGenerator()

    X_pt_source, y_pt_source = get_pt(stream=stream_pt, n_pt=n_pt)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # #Result Record
    directory_path = "./Results/Results_acc/Results_%s/Results_%s_%d/" % (action, dataset_name, n_pt)

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
            stream = get_stream(dataset_name)

            X_pt, y_pt = copy.deepcopy(X_pt_source), copy.deepcopy(y_pt_source)


            # Setup Hyper-parameters
            count = 0

            #Pretrain
            clf.fit(X_pt, y_pt, action = action)

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

                    clf.partial_fit(X, y, mode = action)

                        
            else:
                X, y  = stream.next_sample(test_samples)
                while count < max_times:
                    count += 1
                    y_pred = clf.predict(X)
                    for i in range(X.shape[0]):
                        y_pred_list.append(y_pred[i])
                        y_true_list.append(y[i])
                    clf.partial_fit(X_pt, y_pt, mode = action)

                    acc_all_list.append(accuracy_score(y_pred, y))

            acc_list[n_clf][round] = accuracy_score(y_true_list, y_pred_list)


            y_pred_all = y_pred_all + y_pred_list
            y_true_all = y_true_all + y_true_list
            acc_all = acc_all + acc_all_list

        t2 = time.time()
        if action == "addNewData":
            result_pred = np.array(y_pred_all).reshape(n_round, max_samples*max_times)
            result_true = np.array(y_true_all).reshape(n_round, max_samples*max_times)
            result_pred_name = './Results/Results_acc/Results_%s/Results_%s_%d/Prediction_%s.csv' % (action, dataset_name, n_pt, clf_name_list[n_clf])
            result_true_name = './Results/Results_acc/Results_%s/Results_%s_%d/True_%s.csv' % (action, dataset_name, n_pt, clf_name_list[n_clf])
            with open(result_pred_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_pred)
            with open(result_true_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_true)

        else:
            result_acc = np.array(acc_all).reshape(n_round, max_times)
            result_acc_name = './Results/Results_acc/Results_%s/Results_%s_%d/acc_%s.csv' % (action, dataset_name, n_pt, clf_name_list[n_clf])
            with open(result_acc_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_acc)
        
        print("\nAccuracy %s : %.3f ± %.3f" % (clf_name_list[n_clf], np.mean(acc_list[n_clf]), np.std(acc_list[n_clf])))
        print("Average Time %s : %.4f s\n" % (clf_name_list[n_clf], (t2 - t1) / n_round))

