# encoding: utf-8
"""
@author: lee
@time: 2021/4/6 16:03
@desc: 
"""
import math
import time

import numpy as np

import matplotlib.pylab as plt
import pandas as pd
from sklearn import linear_model
from statsmodels.distributions import ECDF

from aiops.util.common_util import find_nearest


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == '__main__':
    # sudden_drop.csv demo.csv
    df = pd.read_csv('../data/stable_anomaly.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64') // 1e9

    X = df['date'].values
    y = df['count'].values
    plt.subplot(311)
    plt.plot(X, y)

    # z = (y - pre_y_list) / (np.sqrt(pre_y_list))
    z = y
    ecdf = ECDF(z)
    # y_index = np.where(ecdf.y == 0.1)
    find_y_index, find_y_value = find_nearest(ecdf.y, 0.1)
    print(find_y_index, find_y_value)
    threshold = ecdf.x[find_y_index] * 6  # 乘以补偿系数
    print(threshold)

    # 累积法：一段时间窗口内数据的均值超过阈值触发才算异常
    windows = 3
    z = np.append(z[:windows - 1], moving_average(z, windows))

    choose_z_index = np.where(z >= threshold)  # 找到小于阈值的index

    print(X[choose_z_index])

    plt.subplot(312)
    plt.plot(X, z)
    plt.axhline(y=threshold, ls="-", c="red")

    plt.subplot(313)
    plt.plot(ecdf.x, ecdf.y)
    plt.axhline(y=0.1, ls="-", c="red")
    plt.show()
