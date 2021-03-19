# encoding: utf-8
"""
@author: lee
@time: 2021/3/18 16:41
@file: is_stable.py
@desc: 
"""
import pandas as pd

from aiops.util.time_series_tool import is_stable

if __name__ == '__main__':
    # 平稳数据
    df = pd.read_csv('../data/stable.csv')
    data = df['count'].values
    print(is_stable(data, 1.1, True))

    # 周期性数据
    df = pd.read_csv('../data/api_access_fix.csv')
    data = df['count'].values
    print(is_stable(data, 1.1, True))
