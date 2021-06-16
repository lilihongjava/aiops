# encoding: utf-8
"""
@file: RCA.py
@desc: 
"""

import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import t
from termcolor import colored


class RCA():
    """
    根因定位类
    """

    def __init__(self, trace_data, alpha=0.99, ub=0.1, take_minute_averages_of_trace_data=True,
                 division_milliseconds=60000):
        self.trace_data = trace_data
        self.alpha = alpha
        self.ub = ub
        self.anomaly_chart = None
        self.take_minute_averages_of_trace_data = take_minute_averages_of_trace_data
        self.division_milliseconds = division_milliseconds

    def run(self):
        """
        :return: 异常/None列表
        """
        overall_start_time = time.time()

        print('计算异常分数图表')
        self.hesd_trace_detection(alpha=self.alpha, ub=self.ub)
        print('异常分数图表完成')

        output = self.find_anomalous_hosts()

        print('输出结果: ' + colored(str(output), 'magenta'))

        print('处理RCA时间：' + colored('%f', 'cyan') %
              (time.time() - overall_start_time) + '秒.')
        return output

    def esd_test_statistics(self, x, hybrid=True):
        """ 计算用于执行ESD test的location and dispersion样本统计量。
        :param x:
        :param hybrid:
        :return: 用于ESD的两个统计数据
        """
        if hybrid:
            location = np.ma.median(x)  # Median
            dispersion = np.ma.median(np.abs(x - np.median(x)))  # Median Absolute Deviation
        else:
            location = np.ma.mean(x)  # Mean
            dispersion = np.ma.std(x)  # Standard Deviation

        return location, dispersion

    def esd_test(self, x, alpha=0.95, ub=0.499, hybrid=True):
        """ ESD检测
        :param x: List, array, or series containing the time series
        :param alpha: 检测异常值的置信度
        :param ub: 取10%，可标记为异常值的数据点分数的上界（<=0.499）
        :param hybrid: 开启hybrid
        :return: 过去20分钟的分数
        """
        x = [p for p in x if p == p]
        nobs = len(x)
        if ub > 0.4999:
            ub = 0.499
        # k为最大异常个数，ub0.1取数据总量10%
        k = max(int(np.floor(ub * nobs)), 1)
        # ma结构可排除计算过的
        res = np.ma.array(x, mask=False)
        anomalies = []
        med = np.median(x)
        # 1<=i<=k
        for i in range(1, k + 1):
            location, dispersion = self.esd_test_statistics(res, hybrid)
            tmp = np.abs(res - location) / dispersion  # location：median(X)，dispersion：MAD
            idx = np.argmax(tmp)  # Index of the test statistic
            # 残差test_statistic
            test_statistic = tmp[idx]
            n = nobs - res.mask.sum()  # sums  nonmasked values
            # 临界值critical_value,  t.ppf t分布 ppf 百分点函数，概率密度函数的积分值
            critical_value = (n - i) * t.ppf(alpha, n - i - 1) / np.sqrt(
                (n - i - 1 + np.power(t.ppf(alpha, n - i - 1), 2)) * (n - i - 1))
            if test_statistic > critical_value:
                anomalies.append((x[idx] - med) / med)  # 找到异常点后，减中位数再除中位数，作为分数
            res.mask[idx] = True
        if len(anomalies) == 0:
            return 0
        return np.nanmean(np.abs(anomalies))

    def hesd_trace_detection(self, alpha=0.95, ub=0.02, show_pic=False):
        """
        在trace数据上使用S-H-ESD来创建异常图。
        :param alpha:用于ESD的alpha值
        :param ub:ESD ub值
        :param show_pic:
        :return: 异常图(pd.DataFrame())
        """
        grouped_df = self.trace_data.groupby(['cmdb_id', 'serviceName'])[['startTime', 'success', 'elapsedTime']]

        self.anomaly_chart = pd.DataFrame()
        for (a, b), value in grouped_df:
            failure = 0
            # 个性化部分，考虑db异常
            # if 'db' in b:
            #     failure = sum(value['success'] == False) * 5
            value['time_group'] = value.startTime // self.division_milliseconds  # 转换为分钟，取近N分钟，N个点
            value = value.groupby(['time_group'])['elapsedTime'].mean().reset_index()  # time_group、elapsedTime分组取均值
            value = value['elapsedTime'].to_numpy()
            result = 0
            try:
                result = self.esd_test(value, alpha=alpha, ub=ub, hybrid=False)
            except Exception as ex:
                print('esd_test error:')
                print(value)
                traceback.print_exc()
                result = 0
            self.anomaly_chart.loc[b, a] = result + failure

        self.anomaly_chart = self.anomaly_chart.sort_index()

        if show_pic is True:
            from pyecharts import options as opts
            from pyecharts.charts import Graph
            nodes, edges = [], []
            nodes_set = set()
            for col in self.anomaly_chart:
                for index, row in self.anomaly_chart.iterrows():
                    nodes_set.add(index)
                    nodes_set.add(col)
                    if str(row[col]) != 'nan':
                        edges.append({'source': str(col), 'target': str(index)})
            for node in nodes_set:
                nodes.append({'name': str(node), 'symbolSize': 25})
            # Graph(init_opts=opts.InitOpts(width="1024px", height="800px"))
            (
                Graph()
                    .add("", nodes, edges, repulsion=7000, layout="force", is_rotate_label=True,
                         edge_symbol=['none', 'arrow'])
                    .set_global_opts(title_opts=opts.TitleOpts(title="Graph"))
            ).render('./pic/graph.html')

        return self.anomaly_chart

    def find_anomalous_hosts(self, min_threshold=10):
        """异常图里找到根因主机
        :param min_threshold: 最小阈值
        :return: 异常主机字典和kpi字典
        """
        table = self.anomaly_chart.copy()
        # 设置阈值，anomaly_chart最大值的20%，达不到则用min_threshold
        threshold = max(0.2 * table.stack().max(), min_threshold)
        print('异常图中最大的值: %f' % table.stack().max())
        print('阈值: %f' % threshold)

        # rows/columns异常数量，行代表入边，列代表出边
        row_dict = {}
        column_dict = {}
        # rows存异常图表对应所有列的值
        row_confidence_dict = {}
        column_confidence_dict = {}

        for column in table:
            for index, row in table.iterrows():
                increment = 0
                if row[column] >= threshold:
                    # 大于阈值，单元格对应的行列，异常数+1
                    increment = 1

                column_dict[column] = column_dict.get(column, 0)
                column_dict[column] += increment
                column_confidence_dict[column] = column_confidence_dict.get(column, [])
                column_confidence_dict[column].append(row[column])

                row_dict[index] = row_dict.get(index, 0)
                row_dict[index] += increment
                row_confidence_dict[index] = row_confidence_dict.get(index, [])
                row_confidence_dict[index].append(row[column])

        for key, value in column_confidence_dict.items():
            # column_confidence_dict key每列对应所有行求和
            column_confidence_dict[key] = np.nansum(value)

        for key, value in row_confidence_dict.items():
            # row_confidence_dict key每行对应所有行求和
            row_confidence_dict[key] = np.nansum(value)

        final_dict = {}
        # 按行取中值,不受拓扑变化的影响
        for key in list(row_dict.keys()):
            # 有出入边的节点
            if key in list(column_dict.keys()):
                row_dict[key] = (row_dict[key] * 2 + column_dict[key]) // 2
                row_confidence_dict[key] = (row_confidence_dict[key] + column_confidence_dict[key]) / 2
            # 入边的节点计算以及处理过的有出入边的节点
            final_dict[key] = row_dict[key] * row_confidence_dict[key]

        # 异常分数排序
        dodgy_hosts = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))
        # 通过取异常分数Max的10%或1来过滤不太可能的异常
        m = 0.1 * max(list(dodgy_hosts.values()) + [10])
        dodgy_hosts = {k: v for k, v in dodgy_hosts.items() if (v > m)}

        output = self.localize(dodgy_hosts)
        return output

    def localize(self, dodgy_host_dict):
        n = len(dodgy_host_dict)
        print('异常节点数 %d ，具体如下:' % n)
        print(dodgy_host_dict)
        return dodgy_host_dict
