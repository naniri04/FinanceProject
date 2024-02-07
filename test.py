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

def d():
    a = [0]
    def f():
        a[0] = 1
    f()
    print(a[0])

d()
