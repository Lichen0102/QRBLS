
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设CSV文件名为"data.csv"，并且数据分隔符是逗号

datasets = ["covtype","SEA_a","waveform","Faults_SPFD"]

for dataset in datasets:
    data = np.loadtxt('./Results/Results_parameters/Results_%s_1000_1000/acc_parameter_%s.csv' % (dataset, dataset), delimiter=',')  # 读取CSV文件

    # 确保数据是两行100列
    assert data.shape == (2, 100)

    # 将每行数据重塑为10x10矩阵
    matrix1 = data[0].reshape(10,-1)
    matrix2 = data[1].reshape(10,-1)

    # 生成网格数据
    no = np.array([10,20,30,40,50,60,70,80,90,100])
    ns = np.array([10,20,30,40,50,60,70,80,90,100])

    No, Ns = np.meshgrid(no, ns)

    # 创建3D图形并保存第一行数据的图
    fig1 = plt.figure(figsize=(9, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    surface1 = ax1.plot_surface(No, Ns, matrix1.T, cmap='viridis', edgecolor='none')
    # ax1.set_xlim(100, 20)
    # ax1.set_title('Surface Plot for the First Row')
    # 添加颜色条
    cbar1 = fig1.colorbar(surface1, ax=ax1, shrink=0.5, aspect=15, pad=0.065)
    # cbar1.set_label('Color Bar')

    ax1.set_xlabel('$n_o$')
    ax1.set_ylabel('$n_s$')
    ax1.set_zlabel('Accuracy')
    fig1.savefig("acc_parameters"+'_QRBLS_'+dataset+'.pdf')  # 保存为PDF

    # 创建3D图形并保存第二行数据的图
    fig2 = plt.figure(figsize=(9, 5))
    ax2 = fig2.add_subplot(111, projection='3d')
    surface2 = ax2.plot_surface(No, Ns, matrix2.T, cmap='viridis', edgecolor='none')
    # ax2.set_title('Surface Plot for the Second Row')
    # 添加颜色条
    cbar2 = fig2.colorbar(surface2, ax=ax2, shrink=0.5, aspect=15, pad=0.065)
    # cbar2.set_label('Color Bar')

    ax2.set_xlabel('$n_o$')
    ax2.set_ylabel('$n_s$')
    ax2.set_zlabel('Accuracy')
    fig2.savefig("acc_parameters"+'_QRBLS-TDS_'+dataset+'.pdf')  # 保存为PDF

# 显示图形
# plt.show()