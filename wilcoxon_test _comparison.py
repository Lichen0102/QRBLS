from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
datasets = ["covtype","hyperplane","phishing","SEA_a","SEA","sensor_readings_24","waveform","waveform_noisy","weather"]
methods = ["RVFLNN","ELM","ARF","IWDA","QRBLS-TDS","QRBLS"]
data_all_acc = [[] for i in methods]


for dataset in datasets:
    # 获取父目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 更改工作目录为父目录
    os.chdir(current_dir)
    os.chdir('./Results/Results_comparision/')
    data = np.array(pd.read_csv('./acc_comparision_%s.csv' % dataset, header=None))
    str_ = ""
    dic = {}
    for i in range(data.shape[0]):
        dic[data[i][0]] = data[i][1]
    for i in range(len(methods)):
        method = methods[i]
        if method == "ELM":
            method = "OSELM"
        if method == "IWDA":
            method = "IWDA_Multi"
        data_all_acc[i].append(dic[method])

# 将数据转换成 NumPy 数组
accuracy_data = np.array(data_all_acc)
print(accuracy_data)
# 初始化检验结果矩阵
num_methods = accuracy_data.shape[0]
results_matrix = np.zeros((num_methods, num_methods))

# 对每两个方法之间执行 Wilcoxon 符号秩检验
for i in range(num_methods):
    for j in range(num_methods):
        if i != j:
            _, p_value = wilcoxon(accuracy_data[i], accuracy_data[j])
            results_matrix[i, j] = p_value
        else:
            results_matrix[i, j] = 1.0

# 输出检验结果矩阵
print("检验结果矩阵:")
print(results_matrix)
# 计算 p-value 的最大值和最小值
min_p_value = np.min(results_matrix[results_matrix > 0])
max_p_value = np.max(results_matrix)

# 绘制热图，颜色随着 p-value 的增加从深到浅变化
plt.figure(figsize=(9, 7))
h1 = plt.imshow(results_matrix, cmap='cividis', interpolation='bicubic', vmin=min_p_value, vmax=max_p_value)
# 将刻度标签设置为方法名称
plt.xticks(range(num_methods), methods, rotation=25)
plt.yticks(range(num_methods), methods)
# 初始化 legend
legend_added = False

# 标记 p-value 小于 0.05 的元素
for i in range(num_methods):
    for j in range(num_methods):
        if results_matrix[i, j] < 0.05:
            # 在 (i, j) 处添加一个红色圆圈作为标记，并在第一次添加时添加图例
            plt.scatter(j, i, color='white', s=100, marker='*', edgecolors='#A3142E', label='p-value < 0.05' if not legend_added else None)
            legend_added = True

# 添加图例
plt.legend()
plt.tick_params(axis='both', labelsize=15)
cbar = plt.colorbar(h1, label='p-value')
cbar.ax.tick_params(labelsize=15)  # 设置刻度字体大小
cbar.set_label('p-value', fontsize=15)  # 设置 colorbar 标签的字体大小
# plt.title('Wilcoxon results')
# plt.xlabel('Methods')
# plt.ylabel('Methods')
plt.xticks(range(num_methods))
plt.yticks(range(num_methods))
plt.grid(False)
plt.savefig('wilcoxon_comparision.pdf')
plt.show()