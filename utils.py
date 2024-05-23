import akshare as ak
import pandas as pd


def df_process(df:pd.DataFrame, filter_bj =  True):
    df = df.rename_axis(index='date', columns = 'order_book_id')
    if filter_bj:
        stock_id = df.columns.to_list()
        filter_stock = list(filter(lambda x: x.split(".")[1] != 'BJ' , stock_id))
        filter_stock.remove("T00018.SH")
        df = df.loc[:, filter_stock]
    df.columns = df.columns.map(lambda x: my_id_convert(x, 'wind'))
    df.index = pd.to_datetime(df.index)
    # filter BJ
    
    return df


def my_id_convert(stock, method='normal'):
  """
  XSHG == SH
  XSHE == SZ
  
  method: normal, from rq to wind
  """
  sh_pre = ('60', '68')
  sz_pre = ('00','30')
  res = []
  if method == 'normal':
    sh = 'SH'
    sz = 'SZ'
    sh1 = 'XSHG'
    sz1 = 'XSHE'
  else:
    sh1 = 'SH'
    sz1 = 'SZ'
    sh = 'XSHG'
    sz = 'XSHE'

  try:
    if isinstance(stock, str):
      if len(stock) > 6:
        code, area = stock.split(".")  
        if area == sh1:
          new_code = code + '.' + sh
        elif area == sz1:
          new_code = code + '.' + sz
        else:
          raise ValueError(stock)
      else:
        # 6位code
        code = stock
        if stock.startswith(sh_pre):
          new_code = code + '.' + sh
        elif stock.startswith(sz_pre):
          new_code = code + '.' + sz
        else:
          raise ValueError(code, '不属于上交所、深交所代码')
  except Exception as err:
    new_code = stock
    print("--->", err)
  
  return new_code

class Calendar:
    
    def __init__(self) -> None:
        self.df = ak.tool_trade_date_hist_sina()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])

    def get_trading_dates(self, start_date, end_date, market='cn') -> list:
        '''获取交易日'''
        tool_trade_date_hist_sina_df = self.df
        cond = (tool_trade_date_hist_sina_df['trade_date'] >= start_date) & (tool_trade_date_hist_sina_df['trade_date'] <= end_date)
        res = tool_trade_date_hist_sina_df.loc[cond].copy()
        res['trade_date_'] = res['trade_date'].map(lambda x: x.to_pydatetime().date()) # Timestamp格式 ==》 date
        
        return pd.to_datetime(res['trade_date_'].values.tolist()) 

    def get_next_n_trading_dates(self, date, n=1):
        try:
            date = pd.to_datetime(date)
            if n == 0:
                if self.is_trade_date(date):
                    return date
                else: 
                    raise ValueError(f'{date} is not trading dates!')
            elif n > 0:
                cond = self.df['trade_date'] > date
                res = self.df.loc[cond,'trade_date']
            else:
                cond = self.df['trade_date'] < date
                res = self.df.loc[cond,'trade_date']                
            if len(res) > 0:
                return res.iloc[n]
            else:
                print("无法获得")
        except Exception as e:
            print(e)
        
    def get_next_n_trading_dates(self, date, n=1):
        try:      
            date = pd.to_datetime(date)   
            flag = self.is_trade_date(date)          
            if n == 0:
                if flag:
                    return date
                else: 
                    raise ValueError(f'{date} is not trading dates!')
            elif n > 0:
                cond = self.df['trade_date'] > date
                res = self.df.loc[cond,'trade_date']
                return res.iloc[n-1]
            else:
                cond = self.df['trade_date'] < date
                res = self.df.loc[cond,'trade_date'] 
                return res.iloc[n]               
        except Exception as e:
            print(e)        
    
    def is_trade_date(self, date):
        '''判断是否是交易日'''

        return date in self.df['trade_date']
    
    @staticmethod
    def nth_trading_date_of_month(date_range, n = 'last')-> list:
        '''
        给出一段完整的交易区间里，每月的最后一个交易日
        '''
        df = pd.DataFrame(date_range, columns = ['date'])
        df['date'] = pd.to_datetime(df['date'])
        df['year-month'] = df['date'].map(lambda x: str(x.year)+str(x.month))
        
        if n == 'last':
            return_ser = df.groupby('year-month')['date'].last()
        elif n == 'first':
            return_ser = df.groupby('year-month')['date'].first()
        else:
            return_ser = df.groupby('year-month')['date'].nth(n)
        return_ser.reset_index(drop=True)
        return_ser = return_ser.sort_values()
        return return_ser.to_list()  