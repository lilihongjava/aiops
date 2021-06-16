# encoding: utf-8
"""
@author: lee
@time: 2021/6/16 9:59
@file: root_cause_detection.py
@desc: 
"""
from aiops.root_cause.RCA import RCA
import pandas as pd

if __name__ == '__main__':
    trace_data = pd.read_csv(".././data/process_trace.csv", index_col=False)
    rca_temp = RCA(trace_data=trace_data, alpha=0.99, ub=0.1)
    rca_temp.run()
