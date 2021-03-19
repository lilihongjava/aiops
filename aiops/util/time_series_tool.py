# encoding: utf-8
"""
@author: lee
@time: 2021/3/19 9:54
@file: time_series_tool.py
@desc: 
"""
import numpy as np
from pywt import wavedec
import matplotlib.pylab as plt


def is_stable(data, n_threshold=1.1, show_pic=False):
    # 标准差
    raw_data_std = np.std(data, ddof=1)
    # 一维离散信号的小波变换
    coeffs = wavedec(data, 'db4', level=2)
    cA2, cD2, cD1 = coeffs
    # cD2标准差
    cD2_std = np.std(cD2, ddof=1)

    if show_pic is True:
        plt.subplot(311)
        plt.title('original')
        plt.plot(data)
        plt.subplot(312)
        plt.title('ca2')
        plt.plot(cA2)
        plt.subplot(313)
        plt.title('cd2')
        plt.plot(cD2)
        plt.show()

    # 全局波动指标与局部波动指标的比值来描述两者的差异
    n = raw_data_std / cD2_std
    # 比值小于阈值为平稳数据，大于为波动数据
    if n < n_threshold:
        return True
    else:
        return False
