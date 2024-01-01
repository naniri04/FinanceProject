import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
from datetime import datetime
import os

df = pd.read_csv(f'C:/Users/Naniri/Documents/FinanceProject/DB/BackTesting/000020 동화약품.csv')

df['Date'].to_csv(path_or_buf='C:/Users/Naniri/Documents/FinanceProject/DB/trade_date.csv', sep=',', na_rep='NaN', index=False)