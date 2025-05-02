""" Demo of plots. """

import os
from visualization import plot_acc, plot_macro_f1
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, train_samples, max_samples, interval, filename_list):
        # 获取父目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        # 更改工作目录为父目录
        os.chdir(parent_dir)
        os.chdir('./Results/Results_acc/Results_addnewdata/Results_%s_%d_%d/'% (dataset, train_samples, max_samples))

        saving_path = '../../../../'
        std_alpha = 0.2

        colors = ['#E8D3C0', '#D89C7A', '#D6C38B', '#849B91', '#C2CEDC', '#686789', '#AB545A', '#9A7549', '#B0B1B6', '#7D7465']

        # filename_list = filename_list + []
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(9, 4))
        axins = inset_axes(ax,
                   width="30%",  # 子图的宽度
                   height="30%",  # 子图的高度
                   bbox_to_anchor=(0.2, -0.1, 1, 1),  # (x, y, width, height)
                   bbox_transform=ax.transAxes,  # 使用父图的变换坐标系
                   loc='center')
        for idx in range(len(filename_list)):
            filename = filename_list[idx]
            plot_analyzer = plot_acc.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, linewidth=1, method="%s" % (filename), plot_interval=interval, std_alpha=std_alpha)
            ax, axins = plot_analyzer.plot_learning_curve(std_area=True, ax=ax, axins=axins, color=colors[idx], interval=interval)
        
        # 设置放大的区域范围（例如，放大最后5个epoch的区域）
        axins.set_xlim(950, 1000)  # 选择局部区域的x轴范围
        axins.set_ylim(0.75, 0.82)  # 选择局部区域的y轴范围
        # 添加放大框
        ax.indicate_inset_zoom(axins)

        ax.set_xlabel("Instances")
        ax.set_ylabel("Accuracy")
        
        ax.set_ylim(0.3,0.9)
        ax.figure.subplots_adjust(bottom=0.2)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=9)
        fig.savefig(saving_path+'Results_acc_addnewdata_{}.pdf'.format(dataset))
        plt.show()

        # fig_f1 = plt.figure(figsize=(9, 4))
        # for idx in range(len(filename_list)):
        #     filename = filename_list[idx]
        #     plot_analyzer = plot_macro_f1.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, linewidth=1.5, method="%s" % (filename), plot_interval=interval, std_alpha=std_alpha)
        #     plt = plot_analyzer.plot_learning_curve(std_area=True, color=colors[idx], interval=interval)
        # plt.legend(fancybox=True, framealpha=0.5, loc='lower right', fontsize=9, ncol=2)
        # plt.savefig('Results_macro_F1_{}.pdf'.format(dataset))
        # plt.show()
