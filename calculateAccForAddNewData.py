import numpy as np
import csv
import os
from visualization import plot_acc


datasets = ["covtype","hyperplane","phishing","SEA_a","SEA","sensor_readings_24", "waveform_noisy","waveform","weather"]
methods = ["OSELM","ARF","IWDA_Multi","BLS-SMW","QRBLS-TDS","QRBLS"]

for dataset in datasets:
    # 获取父目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 更改工作目录为父目录
    os.chdir(current_dir)
    os.chdir('./Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/'% (dataset, 200, 1000))
    print("----------------------------->>>>>>>>: %s" % dataset)
    str_ = ""
    for method in methods:
        plot_analyzer = plot_acc.plot_tool(pred_file_name=method, true_file_name=method, n_class=2, n_round=3, n_size=1000, linewidth=1, method="%s" % (method), plot_interval=1, std_alpha=0.2)
        acc, std = plot_analyzer.download_acc(1)
        str_ = str_ + str(str(acc)+ " $\pm$ "+ str(std)) + " & "

    print(str_)

    