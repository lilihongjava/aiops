# encoding: utf-8
"""
@author: lee
@time: 2021/3/19 9:54
@file: time_series_tool.py
@desc: 
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pywt import wavedec
from scipy import signal


def is_periodicity(data, show_pic=False):
    """
    :param data:  检测数据，ndarray类型
    :param show_pic: 是否展示图片
    :return: is_cycle为是否周期性, cycles为可能的周期列表
    """
    # 使用周期图法估计功率谱密度
    f, Pxx_den = signal.periodogram(data)

    if show_pic is True:
        plt.plot(Pxx_den)
        # set the x limits
        plt.xlim(-0.5, 100)
        plt.show()

    result = pd.DataFrame(columns=['freq', 'spec'])
    result['freq'] = f
    result['spec'] = Pxx_den
    # 按照频率强弱程度降序排列
    result = result.sort_values(by='spec', ascending=False)
    # 频率转换为周期
    cycle_list = 1 / result.head(2)['freq'].values
    is_cycle = False
    cycles = []
    for cycle in cycle_list:
        # 判断是不是整数
        if cycle % 1 == 0:
            is_cycle = True
            cycles.append(cycle)
    return is_cycle, cycles


def is_stable(data, n_threshold=1.1, show_pic=False):
    """
    :param data: 检测数据，ndarray类型
    :param n_threshold: 阈值
    :param show_pic: 是否展示图片
    :return: True为平稳数据，False波动数据
    """
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
