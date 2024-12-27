import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
from datetime import datetime

whole_code = pd.read_csv('C:/Users/Naniri/Documents/FinanceProject/DB/Original/stock_list.csv')
for i in range(2096, len(whole_code)):
    d = whole_code.iloc[i]
    print(d['Code'])
    try: df = fdr.DataReader(d['Code'], exchange=('KRX-DELISTING' if d['Delisted'] else 'KRX'))
    except KeyError: continue
    if d['Delisted']:
        df = df[['Date', 'Open', 'High', 'Lower', 'Close', 'Volume', 'ChangeRate']]
        df = pd.DataFrame(df, index=[i for i in range(len(df)+1, -1, -1)])
        df.dropna(axis=0, inplace=True)
        df = df.astype({'Open':'int64', 'High':'int64', 'Lower':'int64', 'Close':'int64', 'Volume':'int64'})
        df.rename(columns={'Lower':'Low', 'ChangeRate':'Change'}, inplace=True)
    else:
        df['Date'] = df.index
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']]
        #df.set_index(keys=pd.Series([i for i in range(len(df))]), inplace=True)
        df['Change'] = df['Change'].apply(lambda x: np.round(x*100, 2))

    df.drop(df.index[df['Date'] < datetime(2000,1,1)], axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.index.set_names('idx')
    if len(df): df.to_csv(path_or_buf=f'C:/Users/Naniri/Documents/FinanceProject/DB/Original/BackTesting/{d["Code"]} {d["Name"]}.csv', sep=',', na_rep='NaN')