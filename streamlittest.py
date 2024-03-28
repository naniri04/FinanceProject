# ------ < custom python files > ------
# from pages.kiwoomdownload import main as kiwoompage

# ------ < libraries > ------
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# When using simple browser in vscode:
# streamlit run streamlittest.py --server.headless true

# stock_list = pd.read_csv('DB/stock_list.csv', parse_dates=['SDate', 'EDate'], 
#                          dtype={'Code':str, 'Name':str})
# stock_labels = stock_list['Code'] + ' ' + stock_list['Name']
stock_name_list = pd.read_csv('DB/20240314_stocklist.csv')['0'].values


def cs_colorstyle(csdata, **kwargs):
    csdata.increasing.fillcolor = kwargs['ii']
    csdata.increasing.line.color = kwargs['il']
    csdata.decreasing.fillcolor = kwargs['di']
    csdata.decreasing.line.color = kwargs['dl']


def figure(dnum:int, enum:int, df:pd.DataFrame, madf:pd.DataFrame, hide_gap:bool=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('', ''), row_width=[0.2, 0.7])
    df = df.iloc[enum:enum+dnum]

    # ------ < Remove gaps between candlestick > ------
    if hide_gap: x = df.index.strftime("%Y/%m/%d")
    else: x = df.index

    fig.add_trace(go.Candlestick(x=x, open=df['시가'], high=df['고가'], low=df['저가'], close=df['종가']), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=df['거래량'], showlegend=False), row=2, col=1)
    for col in madf.columns:
        fig.add_trace(go.Line(x=x, y=madf[col]), row=1, col=1)
    fig.update_traces(name='')
    fig.update_layout(height=800, yaxis_tickformat='f', showlegend=False)
    fig.update_xaxes(nticks=5, showgrid=True, rangeslider_visible=False, autorange='reversed')
    fig.update_yaxes(tickformat='s', row=2, col=1)

    cs_colorstyle(fig.data[0], ii='#f52047', il='#f52047', di='#14c0ff', dl='#14c0ff')

    return fig


def ma(df:pd.DataFrame, nlist:tuple):
    mal = dict()
    for n in nlist:
        mal[f'{n}ma'] = df['종가'].iloc[::-1].rolling(window=n).mean().iloc[::-1]
    return pd.DataFrame(mal)
    

def main(): 
    starting_time = [0]
    def time_start():
        starting_time[0] = time.time()
    def get_elapsed_time():
        clock_time = time.time() - starting_time[0]
        with st.empty(): st.write(f"✔️ Task completed in {clock_time:.4f} seconds.") 


    st.sidebar.title("My Finance Project.")
    name = st.sidebar.selectbox("Select stock.", tuple(stock_name_list), placeholder="Type to search...")
    st.subheader(name)
 
    time_start()

    df = pd.read_csv(f"DB/Chart/Not_Adjusted/{name}.csv")
    df.index = pd.to_datetime(df['일자'])
    ma_options = st.sidebar.multiselect(
        'Select Moving-Averages',
        (5, 20, 60, 120),
        (5, 20, 60, 120))
    madf = ma(df, ma_options)
    # st.dataframe(madf)

    dnum = st.sidebar.slider('Data Number', 1, 300, value=70)
    enum = st.sidebar.slider('Ending Time', 0, len(df), value=0)
    st.plotly_chart(figure(dnum, enum, df, madf), use_container_width=True)

    get_elapsed_time()


if __name__ == '__main__':
    main()