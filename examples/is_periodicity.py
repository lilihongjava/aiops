# encoding: utf-8
"""
@author: lee
@time: 2021/3/19 13:55
@file: is_periodicity.py
@desc: 
"""
import pandas as pd

from aiops.util.time_series_tool import is_periodicity

if __name__ == '__main__':
    # 周期数据
    df = pd.read_csv('../data/api_access_fix.csv')
    data = df['count'].values
    is_cycle, cycles = is_periodicity(data)
    print("是否周期：%s，可能周期：%s" % (is_cycle, cycles))

    # 平稳数据
    df = pd.read_csv('../data/stable.csv')
    data = df['count'].values
    is_cycle, cycles = is_periodicity(data)
    print("是否周期：%s，可能周期：%s" % (is_cycle, cycles))

