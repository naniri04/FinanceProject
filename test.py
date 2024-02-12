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




### unused ###

# stylekwargs = dict(up='#b80000', down='#0059b8', wick='in', edge='in', volume='in', ohlc='black')
# plotkwargs = dict(type='candle', volume=True, style='charles', returnfig=True, scale_width_adjustment=dict(lines=0.7),
#                   ylabel='', ylabel_lower='')  # , num_panels=3, main_panel=0, volume_panel=2
# mystyle = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(**stylekwargs), facecolor='#d1d1d1')

# addpl = mpf.make_addplot(madf.tail(dnum), type='line')
# df_plot = df.tail(dnum)
# fig, axlist = mpf.plot(df_plot, **plotkwargs, addplot=addpl)
# axlist[0].yaxis.set_label_position("right")
# axlist[0].yaxis.tick_right() 
# axlist[2].set_facecolor('k')
