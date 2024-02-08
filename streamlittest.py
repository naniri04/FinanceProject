import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.ticker as mticker
import seaborn
import datetime
import pandas as pd
import mplfinance as mpf
import streamlit as st
import time

stock_list = pd.read_csv('DB/stock_list.csv', parse_dates=['SDate', 'EDate'], 
                         dtype={'Code':str, 'Name':str})
stock_labels = stock_list['Code'] + ' ' + stock_list['Name']
stylekwargs = dict(up='#b80000', down='#0059b8', wick='in', edge='in', volume='in', ohlc='black')
plotkwargs = dict(type='candle', volume=True, style='charles', returnfig=True, scale_width_adjustment=dict(lines=0.7),
                  ylabel='', ylabel_lower='')  # , num_panels=3, main_panel=0, volume_panel=2
mystyle = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(**stylekwargs), facecolor='#d1d1d1')


def func(dnum:int, df:pd.DataFrame, madf:pd.DataFrame):
    addpl = mpf.make_addplot(madf.tail(dnum), type='line')
    df_plot = df.tail(dnum)
    fig, axlist = mpf.plot(df_plot, **plotkwargs, addplot=addpl)
    # axlist[0].yaxis.set_label_position("right")
    # axlist[0].yaxis.tick_right() 
    # axlist[2].set_facecolor('k')
    return fig


def ma(df:pd.DataFrame, nlist:tuple):
    mal = dict()
    for n in nlist:
        mal[f'{n}ma'] = df['Close'].rolling(window=n).mean()
    return pd.DataFrame(mal)
    

def main():
    starting_time = [0]
    def time_start():
        starting_time[0] = time.time()
    def get_elapsed_time():
        clock_time = time.time() - starting_time[0]
        with st.empty(): st.write(f"✔️ Task completed in {clock_time:.4f} seconds.") 


    st.title("My Finance Project.")
    name = st.sidebar.selectbox(
        "Select stock.",
        tuple(stock_labels + ' ' + stock_list['Delisted'].apply(lambda x: '🛇' if x else '✔')),
        placeholder="Type to search...",
    )
    st.subheader(name)
 
    time_start()
    df = pd.read_csv(f"DB/BackTesting/{name[:-2]}.csv")
    df.index = pd.to_datetime(df['Date'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    madf = ma(df, (5, 10))
    # st.dataframe(madf)

    dnum = st.sidebar.slider('Data Number', 1, 300, value=70)
    st.pyplot(fig=func(dnum, df, madf))
    get_elapsed_time()
    

if __name__ == '__main__':
    main()