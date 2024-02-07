import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn
import datetime
import pandas as pd
import mplfinance as mpf
import streamlit as st
import time

stock_list = pd.read_csv('DB/stock_list.csv', parse_dates=['SDate', 'EDate'])
stylekwargs = dict(up='#b80000', down='#0059b8', wick='in', edge='in', volume='in', ohlc='black')
plotkwargs = dict(type='candle', volume=True, style='charles', returnfig=True, scale_width_adjustment=dict(lines=0.7))
mystyle = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(**stylekwargs), facecolor='#d1d1d1')


def func(dnum:int, df:pd.DataFrame, name:str, madf:pd.DataFrame):
    addpl = mpf.make_addplot(madf.tail(dnum), type='line')
    fig, axlist = mpf.plot(df.tail(dnum), **plotkwargs, addplot=addpl, title=name)
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
    code = st.sidebar.text_input('Type stock code.', value='000020')

    time_start()
    name = f"{code} {stock_list[stock_list['Code'] == code]['Name'].values[0]}"
    df = pd.read_csv(f"DB/BackTesting/{name}.csv")
    df.index = pd.to_datetime(df['Date'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    madf = ma(df, (5, 10))
    st.dataframe(madf)

    dnum = st.sidebar.slider('Data Number', 1, 300, value=70)
    st.pyplot(fig=func(dnum, df, code, madf))
    get_elapsed_time()
    

if __name__ == '__main__':
    main()