import numpy as np
import csv
import os
from visualization import plot_acc
import pandas as pd

datasets = ["covtype","hyperplane","phishing","SEA_a","SEA","sensor_readings_24","waveform_noisy","waveform","weather"]
methods = ["RVFLNN","ELM","ARF","IWDA_Multi","QRBLS-TDS","QRBLS"]

for dataset in datasets:
    # 获取父目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 更改工作目录为父目录
    os.chdir(current_dir)
    os.chdir('./Results/Results_comparision/')
    print("----------------------------->>>>>>>>: %s" % dataset)
    data = np.array(pd.read_csv('./acc_comparision_%s.csv' % dataset, header=None))
    str_ = ""
    for i in range(data.shape[0]):
        str_ = str_ + str(str(np.round(data[i,1]*100,1))+ " $\pm$ "+ str(np.round(data[i,2]*100,1))) + " & "
    print(str_)

    