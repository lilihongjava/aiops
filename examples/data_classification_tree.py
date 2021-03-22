# encoding: utf-8
"""
@author: lee
@time: 2021/3/19 15:51
@file: data_classification_tree.py
@desc: 
"""
import pandas as pd

from aiops.util.time_series_tool import data_classification_tree

if __name__ == '__main__':
    # 周期数据
    df = pd.read_csv('../data/api_access_fix.csv')
    data = df['count'].values
    print(data_classification_tree(data))

    # 平稳数据
    df = pd.read_csv('../data/stable.csv')
    data = df['count'].values
    print(data_classification_tree(data))

    # 非周期数据
    df = pd.read_csv('../data/aperiodic.csv')
    data = df['count'].values
    print(data_classification_tree(data))
