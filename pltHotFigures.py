import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


datasets = ["covtype","waveform","Faults_SPFD"]
methods = ["QRBLS","BLS"]

def fit_transform(traindata):
    min_v = traindata.min(axis=1)
    max_v = traindata.max(axis=1)
    return ((traindata.T) / (max_v-min_v)).T

for dataset in datasets:
    data_each_dataset = []
    for method in methods:
        folder_path = './Results/Results_weights/weights_%s_%s.csv' % (method, dataset) 
        data = np.loadtxt(folder_path, delimiter=',')
        data = fit_transform(data)
        data_each_dataset.append(data)
        print(data.min(),data.max())
        
    # 创建示例数据，范围在 [-1, 1]
    A = data_each_dataset[0]
    B = data_each_dataset[1]

    # 创建一个图形和两个子图
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # 使用seaborn绘制热力图
    sns.heatmap(A, ax=axes[0], cmap='coolwarm', vmin=np.min(A), vmax=np.max(A))
    axes[0].set_title(methods[0])

    sns.heatmap(B, ax=axes[1], cmap='coolwarm', vmin=np.min(A), vmax=np.max(A))
    axes[1].set_title(methods[1])

    # 显示图形
    plt.tight_layout()
    fig.savefig("weights_"+dataset+'.pdf')

plt.show()