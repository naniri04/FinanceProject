import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
from datetime import datetime

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

timer.start_timer()
kospi = fdr.StockListing('KOSPI')[['Code', 'Name', 'Market']]
kosdaq = fdr.StockListing('KOSDAQ')[['Code', 'Name', 'Market']]
delisted = fdr.StockListing("KRX-DELISTING")
delisted.drop(delisted[delisted['Market'] == "KONEX"].index, axis=0, inplace=True)
delisted.drop(delisted[list(len(str(x)) != 6 for x in delisted['Symbol'])].index, axis=0, inplace=True)
delisted = delisted[['Symbol', 'Name', 'Market']]
delisted.rename(columns={'Symbol':'Code'}, inplace=True)
delisted['Delisted'] = [1] * len(delisted)
merged = pd.concat([kosdaq , delisted , kospi], ignore_index=True)
merged:pd.DataFrame
merged.loc[merged['Market'] == 'KOSDAQ GLOBAL', 'Market'] = 'KOSDAQ'
merged.fillna(0, inplace=True)
merged.sort_values(by='Code', inplace=True)
merged['Kospi'] = [int(x == 'KOSPI') for x in merged['Market']]
merged.drop(['Market'], axis=1, inplace=True)
merged = merged.astype({'Code':'string', 'Name':'string', 'Delisted':'int64'})
merged.drop_duplicates(inplace=True)
merged.reset_index(inplace=True, drop=True)
merged.to_csv(path_or_buf='C:/Users/Naniri/Documents/FinanceProject/DB/Original/stock_list.csv', sep=',', na_rep='NaN')