# encoding: utf-8
"""
@author: lee
@time: 2021/6/16 9:49
@file: kde_transformation.py
@desc: 
"""
import math

import numpy as np


def get_kde(x, data_array, bandwidth=0.1):
    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res


if __name__ == '__main__':
    train_data = np.array([22.22, 20.11, 40.23, 30.45, 13.53, 22.25, 11.91, 21.74, 17.54, 22.65])
    N = len(train_data)
    h = 1.06 * np.std(train_data) * N ** (-1 / 5)
    check_data = np.array([80, 33.11])
    y_array = [-math.log(get_kde(check_data[i], train_data, h)) for i in range(check_data.shape[0])]
    max_index = np.argmax(y_array)
    print("异常程度:%s,对应的值:%s" % (y_array[max_index], check_data[max_index]))
