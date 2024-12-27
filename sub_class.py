# Library
import pandas as pd
from datetime import datetime, timedelta
import h5py
import exchange_calendars as xcals

from torch.utils.data import get_worker_info

ST, ED = datetime(2015,1,1), datetime(2024,12,15)
PATH = "e:/Financial_Data/"
H5PATH = PATH + 'data.h5'
SCHEDULE = xcals.get_calendar('XNAS').schedule.loc[ST:ED][['open', 'close']].map(lambda x: x.tz_localize(None) - timedelta(hours=5))
THZ = ['1m', '5m', '30m', '1d', '1w']


class StockDatasetHDF5:
    '''Return "time series" of stock datasets in 5 horizons.'''
    def __init__(self, min_peroid:timedelta, min_peroid_label:timedelta,
        ticker_list=None, date_range=None,
    ):
        with h5py.File(H5PATH, 'r') as store:
            self._tickers = list(store['chart/inlisted_1m/'].keys()) + list(store['chart/delisted_1m/'].keys())
            self._inlisted_tickers_set = set(store['chart/inlisted_1m/'].keys())
        
        self.ticker_list = ticker_list if ticker_list else self._tickers
        self.date_range = date_range if date_range else [ST, ED]
        self.min_peroid = min_peroid
        self.min_peroid_label = min_peroid_label
        
    def __len__(self):
        return len(self.ticker_list)

    def __getitem__(self, index):
        dfs = dict()
        is_inlisted = self.ticker_list[index] in self._inlisted_tickers_set
        
        with pd.HDFStore(H5PATH, mode='r') as store:
            for hz in THZ:
                dfs[hz] = store.get(f'chart/{'in' if is_inlisted else 'de'}listed_{hz}/{self.ticker_list[index]}')
        st, ed = dfs['1m'].index[0] + self.min_peroid, dfs['1m'].index[-1] - self.min_peroid_label
        if st < self.date_range[0]: st = self.date_range[0]
        if ed > self.date_range[1]: ed = self.date_range[1]
        for hz in THZ[:3]:
            dfs[hz] = dfs[hz].loc[st:ed]
        st = st.replace(hour=0, minute=0, second=0)
        dfs['1d'] = dfs['1d'].loc[st:ed]
        st = st - timedelta(days=st.isoweekday() - 1)
        dfs['1w'] = dfs['1w'].loc[st:ed]
        
        # 정보 추가
        dfs['ticker'] = self.ticker_list[index]
        
        return dfs