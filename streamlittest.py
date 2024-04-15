# from pages.kiwoomdownload import main as kiwoompage

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os
from tqdm import tqdm

# When using simple browser in vscode:
# streamlit run streamlittest.py --server.headless true
# link: http://localhost:8501

# region [CONSTANT]
DAEBI_SIGN = {1: '상한', 2: '상승', 3: '보합', 4: '하한', 5: '하락'}
ADJUSTED_SIGN = {'1':'유증', '2':'무증', '4':'배당', '8':'액분', '16':'액병', '32':'기업합병'
                 , '64':'감자', '256':'권리락'}
MA_LINE_COLOR = ['#809903', '#a63700', '#008062', '#4e02e6', '#b80093', '#800062'] + ['#000000']*10
# endregion
# region [LAMBDA FUNC]
ma_width_adjust = lambda x: 20/(x+10)+1
# endregion
# region [FIELD]
stock_code_list = pd.read_csv('DB/20240314_stocklist.csv')['0'].values
# endregion


def cs_colorstyle(csdata, **kwargs):
    csdata.increasing.fillcolor = kwargs['ii']
    csdata.increasing.line.color = kwargs['il']
    csdata.decreasing.fillcolor = kwargs['di']
    csdata.decreasing.line.color = kwargs['dl']


def figure(dnum:int, enum:int, df:pd.DataFrame, madf:pd.DataFrame, hide_gap:bool=True, ft=pd.DataFrame(), ftname=[]):
    feature_num = len(ft.columns)
    feature_names = ['Candlestick', 'Volume'] + ftname
    feature_names = [f"<b>{name}</b>" for name in feature_names]
    fig = make_subplots(rows=2+feature_num, cols=1, shared_xaxes=True, vertical_spacing=0.05
                        , subplot_titles=feature_names
                        , row_heights=[1, 0.25]+[0.25]*feature_num)
    
    # region [PLOTTING CANDLESTICK AND VOLUME]
    
    df = df.iloc[enum:enum+dnum]
    min_price, max_price = df['저가'].min(), df['고가'].max()
    y_delta = max_price - min_price

    # ------ < Remove gaps between candlestick > ------
    if hide_gap: x = df.index.strftime("%Y/%m/%d")
    else: x = df.index

    adjustment_label = df.drop(df.loc[df['수정비율'].isna()].index)
    fig.add_trace(go.Candlestick(x=x, open=df['시가'], high=df['고가'], low=df['저가'], close=df['종가']), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=df['거래량'], showlegend=False), row=2, col=1)
    for r in adjustment_label.values:
        fig.add_annotation(x=r[0].replace('-', '/'), y=r[3]-y_delta*0.01, text=f"{ADJUSTED_SIGN[r[7]]}:{r[8]}", ay=min_price, ax=0, ayref='y'
                           , bordercolor="#000000", showarrow=True, bgcolor="#ffe642", arrowwidth=1.3, opacity=1, yanchor='top', row=1, col=1)
    for i in range(len(madf.columns)):
        fig.add_trace(go.Scatter(x=x, y=madf.iloc[enum:enum+dnum, i], mode='lines', line_color=MA_LINE_COLOR[i]
                                 , line_width=ma_width_adjust(dnum)), row=1, col=1)
    fig.update_layout(height=625+125*feature_num, yaxis_tickformat='f', showlegend=False
                      , yaxis_range=[min_price-y_delta*0.02, max_price+y_delta*0.02])
    fig.update_xaxes(nticks=5, showgrid=True, rangeslider_visible=False, autorange='reversed')
    fig.update_yaxes(tickformat='s', row=2, col=1)
    fig.update_yaxes(nticks=10, row=1, col=1)
    
    cs_colorstyle(fig.data[0], ii='#f52047', il='#f52047', di='#14c0ff', dl='#14c0ff')
    
    # endregion
    # region [PLOTTING ADDITIONAL FEATURES]
    
    for i in range(feature_num):
        fig.add_trace(go.Scatter(x=x, y=ft.iloc[enum:enum+dnum, i], mode='lines', line_color='#000000'
                                 , line_width=ma_width_adjust(dnum)), row=3+i, col=1)
    
    # endregion
    # region [FIGURE SETTING]
    
    fig.update_layout(xaxis=dict(showspikes=True, spikemode='across+toaxis', spikesnap='hovered data', spikethickness=0.3
                                , spikecolor='#000000', spikedash='solid'), hovermode='x unified'
                      , hoverlabel=dict(bgcolor='rgba(255,255,255,0.4)'), title_text='BAM')
    fig.update_traces(name='', xaxis='x1')
    for i in range(2+feature_num): fig.layout.annotations[i].update(x=0.01, font={'size':12, 'color':'black'})
    
    # endregion

    return fig


def ma(df:pd.DataFrame, nlist:tuple):
    mal = dict()
    for n in nlist:
        mal[f'{n}ma'] = df['종가'].iloc[::-1].rolling(window=n).mean().iloc[::-1]
    return pd.DataFrame(mal)
    
    
def features(df:pd.DataFrame):
    fts = pd.DataFrame(index=df.index)
    fts['grad'] = (df['종가'].shift(1) - df['종가'].shift(-1)) / df['종가'] * 100
    
    return fts


@st.cache_data
def load_data(fpath:str, arg:dict={}):
    ldir = os.listdir(fpath)
    files = [fname for fname in ldir if os.path.isfile(os.path.join(fpath, fname))]
    data = {fname.removesuffix('.csv'):pd.read_csv('/'.join([fpath, fname]), **arg) for fname in tqdm(files)}
    return data
    

def main(): 
    # region [TIMER]
    starting_time = [0]
    def time_start():
        starting_time[0] = time.time()
    def get_elapsed_time():
        clock_time = time.time() - starting_time[0]
        with st.empty(): st.write(f"✔️ Task completed in {clock_time:.4f} seconds.") 
    # endregion
    
    st.set_page_config(layout="wide")
    code = st.sidebar.selectbox("Select stock.", tuple(stock_code_list), placeholder="Type to search...")
    data = load_data("DB/Chart/Not_Adjusted", dict(dtype={'수정주가구분':object, '수정비율':object}))
 
    time_start()

    df = data[code]
    df.index = pd.to_datetime(df['일자'])
    ma_options = st.sidebar.multiselect(
        'Select Moving-Averages',
        (1, 5, 20, 60, 120),  # options
        (5, 20))  # default
    madf = ma(df, ma_options)

    dnum = st.sidebar.slider('Data Number', 1, 300, value=70)
    enum = st.sidebar.slider('Ending Time', 0, len(df), value=0)
    st.plotly_chart(figure(dnum, enum, df, madf, ft=features(df), ftname=['Grad'])
                    , use_container_width=True, config={'displayModeBar': False})

    get_elapsed_time()


if __name__ == '__main__':
    main()