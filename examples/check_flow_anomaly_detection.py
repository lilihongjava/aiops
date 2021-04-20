# encoding: utf-8
"""
@author: lee
@time: 2021/4/6 16:03
@file: check_flow_anomaly_detection.py
@desc: 
"""
import math
import time

import numpy

import matplotlib.pylab as plt
import pandas as pd
from sklearn import linear_model
from statsmodels.distributions import ECDF

from aiops.util.common_util import find_nearest


def data_equalization(data_frame, windows, length):
    start = int(-windows)
    end = 0
    pre_y_list = numpy.array([])
    while end < length:
        start = start + windows
        end = start + windows - 1
        data = data_frame.loc[start:end]
        pre_y = model_build(data)
        pre_y_list = numpy.append(pre_y_list, pre_y)
    return pre_y_list


def model_build(data_frame):
    X = data_frame['date'].values.reshape(-1, 1)
    y = data_frame['count'].values

    # 使用RANSAC清除异常值高鲁棒对的线性回归模型
    model = linear_model.RANSACRegressor()
    # model = linear_model.LinearRegression()

    model.fit(X, y)
    pre_y = model.predict(X)
    plt.plot(X, pre_y, color='red')
    return pre_y


if __name__ == '__main__':
    # sudden_drop.csv demo.csv
    # df = pd.read_csv('../data/sudden_drop.csv')
    df = pd.read_csv('../data/sudden_rise.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64') // 1e9
    pre_y_list = data_equalization(df, 30, len(df))
    X = df['date'].values
    y = df['count'].values
    plt.subplot(311)
    plt.plot(X, y)

    z = (y - pre_y_list) / (numpy.sqrt(pre_y_list))
    z = -z
    ecdf = ECDF(z)
    # y_index = numpy.where(ecdf.y == 0.1)
    y_index, find_y_value = find_nearest(ecdf.y, 0.1)
    threshold = ecdf.x[y_index] * 3  # 乘以补偿系数

    choose_z_index = numpy.where(z <= threshold)  # 找到小于阈值的index

    print(X[choose_z_index])
    print(numpy.asarray(X[choose_z_index], dtype='datetime64[s]'))

    plt.subplot(312)
    plt.plot(X, z)
    plt.axhline(y=threshold, ls="-", c="red")

    plt.subplot(313)
    plt.plot(ecdf.x, ecdf.y)
    plt.axhline(y=0.1, ls="-", c="red")
    plt.show()
