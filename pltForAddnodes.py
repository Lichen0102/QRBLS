import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import cm
import seaborn as sns

datasets = ["Faults_SPFD","sensor_readings_24","covtype","phishing","weather","waveform","hyperplane"] #addNewFeatureNodes
# datasets = ["phishing","weather","SEA_a","Faults_SPFD"] 
methods = ["PF-BLS","BLS","RVFLNN","QRBLS-TDS","QRBLS"]
parameters1 = [i for i in range(100, 2600, 100)]
parameters2 = [i for i in range(0, 1251, 50)]

colors = ['#A3142E', '#d9541a', '#edb021', '#7d2e8f', '#78ab30']

for dataset in datasets:
    data_all_acc = {}
    data_all_err = {}
    for method in methods:
        acc_path_file = './Results/Results_acc/Results_addnewnodes/Results_%s_%d_%d/acc_%s.csv' % (dataset, 1000, 1000, method)
        err_path_file = './Results/Results_error/error_%s_%s.csv' % (method, dataset)
        data_acc = np.loadtxt(acc_path_file, delimiter=',')
        data_all_acc[method] = data_acc[1:]
        data_eror_df = pd.read_csv(err_path_file)
        data_eror = data_eror_df.to_numpy()
        data_all_err[method] = data_eror



    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

    # 绘制上半部分：每个方法的曲线和填充区域
    for method, color in zip(methods, colors):
        # ax1.fill_between(parameters1, 0, data_all_acc[method], color=color, alpha=0.3)
        ax1.plot(parameters1, data_all_err[method][0,1:], color=color)

    ax1.set_ylabel('EN', labelpad=10, fontsize=20)

    # 绘制下半部分：普通条形图
    bar_width = 15
    offset = bar_width * (len(methods) - 1) / 2  # 调整每个条形图的位置，避免重叠
    y_min = 1
    y_max = 0
    for method in methods:
        y_min = np.min(data_all_acc[method]) if np.min(data_all_acc[method]) < y_min else y_min
        y_max = np.max(data_all_acc[method]) if np.min(data_all_acc[method]) > y_max else y_max
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        # 每个方法的条形图位置依次排列
        ax2.bar(np.array(parameters1) - offset + i * bar_width, data_all_acc[method], bar_width, label=method, color=color)
        y_min = y_min - 0.01 if y_min > 0.01 else y_min
        y_max = y_max + 0.01 if y_max < (1- 0.01) else y_max

        ax2.set_ylim(y_min, y_max)


    ax2.set_xlabel('Additional enhancement nodes', labelpad=10, fontsize=20)
    ax2.set_ylabel('Accuracy', labelpad=10, fontsize=20)

    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())  # 确保上下轴对齐
    # ax3.set_xlabel('Additional feature nodes', labelpad=10, fontsize=20)
    # ax3.set_xticklabels([str(x) for x in range(-250, 1251, 250)], fontsize=15)

    ax3.set_xticklabels([])
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)

    fig.subplots_adjust(bottom=0.25)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=6, fontsize=20)

    # 在上下图之间画一条横线
    plt.subplots_adjust(hspace=0.05)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    fig.savefig('en_acc_'+ dataset +'_addnewnodes.pdf')
    plt.show()




