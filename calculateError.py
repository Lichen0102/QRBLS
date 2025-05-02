import numpy as np
import csv
import os
datasets = ["Faults_SPFD","sensor_readings_24","covtype","phishing","weather","waveform","hyperplane"]
methods = ["BLS","PF-BLS","RVFLNN","QRBLS","QRBLS-TDS"]



directory_path = "./Results/Results_error/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

def fit_transform(traindata):
    _mean = traindata.mean(axis=1)
    _std = traindata.std(axis=1)
    return (traindata - _mean) / (_std)


# 相对误差
def relative_error(x_k, x_true):
    return np.linalg.norm(x_k - x_true) / np.linalg.norm(x_true)

# 残差
def residual(A, x_k, b):
    return np.linalg.norm(b - A @ x_k)

# 误差范数
def error_norm(x_k, x_true):
    return np.linalg.norm(x_k - x_true)

def mean_square_error(x_k, x_true):
    return np.mean((x_k - x_true) ** 2)

def root_mean_square_error(x_k, x_true):
    return np.sqrt(mean_square_error(x_k, x_true))

for dataset in datasets:
    for method in methods:
        print("--------------------(%s,%s)>>>>>>" % (dataset,method))
        err_list = [["re"],["en"],["mse"],["rmse"]]
        for count in range(25):
            print(count)
            direct_path_file = './Results/Results_matrix/extension_matrix_direct_%s_%s_%s.csv' % (method, (count+1)*100, dataset)
            data_direct = np.loadtxt(direct_path_file, delimiter=',')

            iter_path_file = './Results/Results_matrix/extension_matrix_iter_%s_%s_%s.csv' % (method, (count+1)*100, dataset)
            data_iter = np.loadtxt(iter_path_file, delimiter=',')

            if method == "PF-BLS":
                data_direct = data_direct.T

            re = relative_error(data_iter, data_direct)
            en = error_norm(data_iter,data_direct)
            mse = mean_square_error(data_iter, data_direct)
            rmse = root_mean_square_error(data_iter, data_direct)
            # print("相对误差(re): ",re)
            # print("误差范数(en): ",en)
            # print("均方误差(mse): ",re)
            # print("均方根误差(rmse): ",en)
            err_list[0].append(re)
            err_list[1].append(en)
            err_list[2].append(mse)
            err_list[3].append(rmse)
        err_arr = np.array(err_list)
        path_file = './Results/Results_error/error_%s_%s.csv' % (method, dataset)
        with open(path_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(err_arr)
