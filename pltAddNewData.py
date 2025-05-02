import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visualization.plot_comparison import plot_comparison


datasets = ["covtype","hyperplane","phishing","SEA_a","SEA","sensor_readings_24","waveform_noisy","waveform","weather"]
methods = ["OSELM","ARF","IWDA_Multi","BLS-SMW","QRBLS-TDS","QRBLS"]

# for dataset in datasets:
plot_comparison(dataset=datasets[8], n_class=2, n_round=3, train_samples = 200,max_samples=1000, interval=1, filename_list=methods)
