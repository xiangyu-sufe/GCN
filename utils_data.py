import os
from typing import List
import numpy as np
import pandas as pd

import torch

from torch_geometric.data import Data


class FeatureStorage:
    """
    用于生成常见的量价因子数据
    close, turnover, vol, 特异度, amihud非流动性, APB, UMR
    
    """
    def __init__(self, close:pd.DataFrame, adj_date:List[pd.Timestamp]) -> None:
        self.feature = {}
        self.close = close
        
    
    def _add_feature(self, name:str, fac:pd.DataFrame):
        
        self.feature[name] = fac
        print('%s已经加入特征数据集中.' %name)
        
    def gen_past_ret(self, n):
        # 过去n个交易日收益率
        past_ret = self.close.pct_change(periods=n)
        self._add_feature("past_ret_%s" %n, past_ret)
    
    def gen_past_ret_std(self, n):
        past_ret = self.close.pct_change()
        past_ret_std = past_ret.rolling(window=20).std()
        self._add_feature("vol_%s" %n, past_ret_std)
        
    def gen_mom(self, n, m):
        assert n > m
        mom = self.close.shift(m).pct_change(n-m)
        
        self._add_feature("mom_%s" %n, mom)
       
    def _create_single_graph(self, idx, edge_type):
        """
        给定因子数据创建图
        """
        np_data = []
        
        for k, v in feature.feature.items():
            np_data.append(v.iloc[[idx]].values)
        np_data = np.vstack(np_data)     
        
        


 
if __name__ == "__main__":
    print(torch.cuda.is_available())
    data_path = "/gemini/data-3"
    data = pd.read_csv(os.path.join(data_path, "A_2010-01-01_2024-05-01_MovingAverage_data.csv"),
                       compression='gzip',
                       parse_dates=[1])
    data = data.pivot(index = 'date', columns = 'order_book_id', values = 'close')
    print(data.head())
    
    feature = FeatureStorage(close=data, adj_date=None)
    feature.gen_past_ret(5)
    feature.gen_past_ret_std(5)
    feature.gen_mom(20,5)
    
    for k, v in feature.feature.items():
        print(k)
        print(v.iloc[0].values)
        print(v.iloc[[0]].values)

