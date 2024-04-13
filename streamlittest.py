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

# ------ < constants > ------
DAEBI_SIGN = {1: '상한', 2: '상승', 3: '보합', 4: '하한', 5: '하락'}
ADJUSTED_SIGN = {'1':'유증', '2':'무증', '4':'배당', '8':'액분', '16':'액병', '32':'기업합병'
                 , '64':'감자', '256':'권리락'}

# ------ < lambda functions > ------
ma_width_adjust = lambda x: 20/(x+10)+1

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
    min_price, max_price = df['저가'].min(), df['고가'].max()
    print(min_price)
    y_delta = max_price - min_price

    # ------ < Remove gaps between candlestick > ------
    if hide_gap: x = df.index.strftime("%Y/%m/%d")
    else: x = df.index

    adjustment_label = df.drop(df.loc[df['수정비율'].isna()].index)
    # print(adjustment_label)
    fig.add_trace(go.Candlestick(x=x, open=df['시가'], high=df['고가'], low=df['저가'], close=df['종가']), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=df['거래량'], showlegend=False), row=2, col=1)
    for r in adjustment_label.values:
        fig.add_annotation(x=r[0].replace('-', '/'), y=r[3]-y_delta*0.01, text=f"{ADJUSTED_SIGN[r[7]]}:{r[8]}", ay=min_price, ax=0, ayref='y'
                           , bordercolor="#000000", showarrow=True, bgcolor="#ffe642", arrowwidth=1.3, opacity=1, yanchor='top', row=1, col=1)
    for col in madf.columns:
        fig.add_trace(go.Scatter(x=x, y=madf[col], mode='lines', line_color='#000000', line_width=ma_width_adjust(dnum)), row=1, col=1)
    fig.update_traces(name='')
    fig.update_layout(height=600, yaxis_tickformat='f', showlegend=False, yaxis_range=[min_price-y_delta*0.02, max_price+y_delta*0.02])
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


    # st.sidebar.title("My Finance Project.")
    name = st.sidebar.selectbox("Select stock.", tuple(stock_name_list), placeholder="Type to search...")
    # st.subheader(name) 
 
    time_start()

    df = pd.read_csv(f"DB/Chart/Not_Adjusted/{name}.csv", dtype={'수정주가구분':object, '수정비율':object})
    df.index = pd.to_datetime(df['일자'])
    ma_options = st.sidebar.multiselect(
        'Select Moving-Averages',
        (1, 5, 20, 60, 120),  # options
        (1))  # default
    madf = ma(df, ma_options)

    dnum = st.sidebar.slider('Data Number', 1, 300, value=70)
    enum = st.sidebar.slider('Ending Time', 0, len(df), value=0)
    st.plotly_chart(figure(dnum, enum, df, madf), use_container_width=True, config={'displayModeBar': False})

    get_elapsed_time()


if __name__ == '__main__':
    main()