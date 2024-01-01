import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
from datetime import datetime
import os

wdf = pd.read_csv('C:/Users/Naniri/Documents/FinanceProject/DB/Original/stock_list_old.csv')

delete = []
wdf['SDate'] = np.nan; wdf['EDate'] = np.nan
st_ed = pd.DataFrame({'SDate':[], 'EDate':[]})
# i=0
for idx, rows in wdf.iterrows():
    path = f'C:/Users/Naniri/Documents/FinanceProject/DB/Original/BackTesting/{rows["Code"]} {rows["Name"]}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        st, ed = df.iloc[0, 1], df.iloc[-1, 1]
        wdf.loc[idx,['SDate', 'EDate']] = [st, ed]
        delete.append(False)
    else:
        delete.append(True)
    # i+=1
    # if i >= 20: break

wdf.drop(wdf.index[delete], axis=0, inplace=True)
wdf.to_csv(path_or_buf='C:/Users/Naniri/Documents/FinanceProject/DB/Original/stock_list.csv', sep=',', na_rep='NaN')