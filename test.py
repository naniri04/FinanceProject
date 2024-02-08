import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas import DataFrame
import itertools
import pandas as pd
import mplfinance as mpf
import matplotlib.animation as animation

df = pd.DataFrame({0:[1,2,3], 1:[2,3,4], 2:[3,4,5]})
print(type(df.iloc[1,1]))
