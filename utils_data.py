import os
from typing import List, Dict
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from utils import Calendar

class FeatureStorage:
    """
    用于生成常见的量价因子数据
    close, turnover, vol, 特异度, amihud非流动性, APB, UMR
    
    """
    def __init__(self, close:pd.DataFrame, adj_date:List[pd.Timestamp]) -> None:
        self.feature:Dict[str, pd.DataFrame] = {}
        self.close = close
        self.adj_date = adj_date
        self.calendar = Calendar()
    
    def _gen_label(self, n=20):
        """
        label: 未来一段时间的收益率
        """
        self.label_df = self.close.pct_change(periods=n, fill_method=None).shift(periods=-n)
        print('label(%d天收益率)生成成功...' %n)    
            
    def _add_feature(self, name:str, fac:pd.DataFrame):
        
        self.feature[name] = fac
        print('%s已经加入特征数据集中.' %name)
        
    def gen_past_ret(self, n):
        # 过去n个交易日收益率
        past_ret = self.close.pct_change(periods=n, fill_method=None)
        self._add_feature("past_ret_%s" %n, past_ret)
    
    def gen_past_ret_std(self, n):
        # 过去n个交易日波动率
        past_ret = self.close.pct_change(fill_method=None)
        past_ret_std = past_ret.rolling(window=n).std()
        self._add_feature("vol_%s" %n, past_ret_std)
        
    def gen_mom(self, n, m):
        # 过去
        assert n > m
        mom = self.close.shift(m).pct_change(n-m, fill_method=None)
        
        self._add_feature("mom_%s" %n, mom)
    
    def gen_tovol(self, n:int, turn_over:pd.DataFrame):
        """
        std(turn_over, n) / mean(turn_over, n)
        """
        std_to = turn_over.rolling(window=n).std()
        mean_to = turn_over.rolling(window=n).mean()
        
        tovol = std_to / (mean_to+1e-5)
        self._add_feature("tovol_%s" %n, tovol)        
    
    def gen_lnto(self, n:int, turn_over:pd.DataFrame):
        """
        log(mean(turn_over, n))
        """
        mean_to = turn_over.rolling(window=n).mean()
        lnto = np.log(mean_to+1e-10)
        self._add_feature("lnto_%s" %n, lnto)   
        
    def gen_ivol(self, n, index:pd.DataFrame, type = 'daily'):
        
        assert type in ['daily', 'weeekly']
        if type == 'weekly':
            stock_ret = self.close.resample('W-FRI').last().pct_change(fill_method=None)
            index_ret = index.resample('W-FRI').last().pct_change(fill_method=None)            
        else:
            stock_ret = self.close.pct_change(fill_method=None)
            index_ret = index.pct_change(fill_method=None)
        ivol = pd.DataFrame(np.nan, columns=self.close.columns, index = self.adj_date)
        for t in self.adj_date:
            try:
                t_i1 = stock_ret.index.get_loc(t)
                t_i2 = index_ret.index.get_loc(t)
                assert t_i1 >= n and t_i2 >= n
                df_x = index_ret.iloc[(t_i2-n+1):(t_i2+1),:].copy()
                df_y = stock_ret.iloc[(t_i1-n+1):(t_i1+1),:].copy()
                selected_col = (df_y.isna().sum() == 0) 
                arr_y = df_y.loc[:,selected_col].values
                df_x.loc[:,'const'] = 1
                arr_x = df_x.values
                residuals = arr_y - arr_x @ np.linalg.inv(arr_x.T @ arr_x) @ arr_x.T @ arr_y
                residuals_std = residuals.std(axis=0)
                ivol.loc[t, selected_col] = residuals_std
            except Exception as err:
                print(err)                
        
        return ivol
    
    def gen_amihud(self, amount:pd.DataFrame):
        stock_ret = self.close.pct_change(fill_method=None).abs()
        amihud = stock_ret / (amount+1e-5)
        amihud_modify = np.log(amihud.resample('M').mean()+1)
       
        return amihud_modify
    
    def create_graphs(self, st, et, ind):
        trading_dates = self.calendar.get_trading_dates(st, et)
        adj_date = self.calendar.nth_trading_date_of_month(trading_dates)        
        stock_pool = self.close.columns
        graphs = []
        for t in adj_date:
            selected_stocks = stock_pool[self._find_stock_(t)]
            single_graph = self._create_single_graph(t, selected_stocks, ind)
            graphs.append(single_graph)
            
        return graphs
            
    def _find_stock_(self, date):
        bool_ = None
        
        for k, v in self.feature.items():
            notna = v.loc[date,:].notna().values
            if bool_ is None:
                bool_ = notna
            else:
                bool_ &= notna
        notna = self.label_df.loc[date,:].notna().values    
        bool_ &= notna
             
        return bool_
    def _create_single_graph(self, date, codes, ind):
        """
        给定因子数据创建图
        """
        np_data = []
        for k, v in self.feature.items():
            v = v.loc[date, codes].values
            v = (v-np.mean(v))/(np.std(v)+1e-5)
            np_data.append(v[:,np.newaxis])
        np_data = np.hstack(np_data)     
        ts_data = torch.from_numpy(np_data).to(torch.float)
        y = self.label_df.loc[date, codes].values
        #标准化
        y = (y-np.mean(y))/(np.std(y)+1e-5)
        ts_y = torch.from_numpy(y).to(torch.float)
        edge_index = self.construct_edges_by_ind(ind, codes)
        ts_edge_index = torch.tensor(edge_index, dtype=torch.long)
        # create graph
        data = Data(x=ts_data, edge_index=ts_edge_index, y=ts_y)
        
        return data 
    
    @staticmethod
    def construct_edges_by_ind(ind:pd.DataFrame, codes:list):
        # 
        # order_book_id  first_industry_code first_industry_name
        def construct_edges_by_list(edges, point_list):
            for i in range(len(point_list)):
                for j in range(i+1, len(point_list)):
                    edges[0].append(point_list[i])
                    edges[0].append(point_list[j])
                    edges[1].append(point_list[j])
                    edges[1].append(point_list[i])                
            
        edges = [[],[]]
        ind = ind[ind['order_book_id'].isin(codes)].copy()
        ind.sort_values('order_book_id', inplace=True)
        ind.reset_index(drop=True, inplace=True)
        ind.loc[:,'num'] = range(len(ind))
        for ind_code, df in ind.groupby(by='first_industry_code'):
            # print(ind_code)
            # print(df)
            point_list = df['num'].to_list()
            construct_edges_by_list(edges, point_list)
        
        return edges
    



if __name__ == "__main__":
    # print(torch.cuda.is_available())
    data_path = "/gemini/data-1"
    data = pd.read_csv(os.path.join(data_path, "A_2010-01-01_2024-05-01_MovingAverage_data.csv"),
                       compression='gzip',
                       parse_dates=[1])
    data = data.pivot(index = 'date', columns = 'order_book_id', values = 'close')
    print(data.head())
    feature = FeatureStorage(close=data, adj_date=None)
    feature.gen_past_ret(5)
    feature.gen_past_ret_std(5)
    feature.gen_mom(20,5)
    feature._gen_label()
    
    # 行业
    ind = pd.read_csv(os.path.join(data_path, "A_2010-01-01_2024-05-01_MovingAverage_ind.csv"),
                       compression='gzip',)
    print(ind.head())
    print(ind.loc[[1,3,5,7,9,10,11],'order_book_id'].to_list())

    graph = feature._create_single_graph("2011-01-04", ["000001.XSHE", "000002.XSHE", "000004.XSHE"], ind)
    

