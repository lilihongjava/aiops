# encoding: utf-8
"""
@author: lee
@time: 2021/4/7 10:18
@file: common_util.py
@desc: 
"""
import numpy as np


def find_nearest(array, value):
    """ 查找ndarray最接近value的值
    :param array: ndarray
    :param value: 要找的值
    :return: index索引，具体的值
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def create_step_dataset(data, step=1, out_units=1):
    data_x, data_y = [], []
    for i in range(len(data) - step):
        # 取step行数据
        data_x.append(data[i:(i + step)])
        # 输出几列
        data_y.append(data[(i + step):(i + step + out_units)])
    return np.array(data_x), np.array(data_y)
