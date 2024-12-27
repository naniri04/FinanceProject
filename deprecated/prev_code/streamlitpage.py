# from pages.kiwoomdownload import main as kiwoompage

# region [IMPORT]
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
# endregion
# region [MODEL]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import sklearn.metrics as metrics
# endregion

# When using simple browser in vscode:
# streamlit run streamlitpage.py --server.headless true --server.runOnSave true
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
stock_code_list = pd.read_csv('../../FinanceData/DB/20240314_stocklist.csv')['0'].values
# endregion


def cs_colorstyle(csdata, **kwargs):
    csdata.increasing.fillcolor = kwargs['ii']
    csdata.increasing.line.color = kwargs['il']
    csdata.decreasing.fillcolor = kwargs['di']
    csdata.decreasing.line.color = kwargs['dl']


def figure_chart(dnum:int, enum:int, df:pd.DataFrame, hide_gap:bool=True, ft=pd.DataFrame(), ftname=[], ma=[]):
    ft = ft.loc[:,[n for n in ftname if n in ft.columns]]
    feature_num = len(ftname)
    feature_names = ['Chart', 'Volume'] + ftname
    fig = make_subplots(rows=2+feature_num, cols=1, shared_xaxes=True, vertical_spacing=0.05
                        , subplot_titles=[f"<b>{name}</b>" for name in feature_names]
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
    for i, r in adjustment_label.iterrows(): 
        fig.add_annotation(x=r.name.strftime('%Y/%m/%d'), y=r['저가']-y_delta*0.01
                           , text=f"{ADJUSTED_SIGN[r['수정주가구분']]}:{r['수정비율']}"
                           , ay=min_price, ax=0, ayref='y', bordercolor="#000000", showarrow=True, bgcolor="#ffe642"
                           , arrowwidth=1.3, opacity=1, yanchor='top', row=1, col=1)
    for i, n in enumerate(ma):
        fig.add_trace(go.Scatter(x=x, y=df[f'ma{n}'].iloc[enum:enum+dnum], mode='lines', line_color=MA_LINE_COLOR[i]
                                 , line_width=ma_width_adjust(dnum)), row=1, col=1)
    fig.update_layout(height=625+125*feature_num, yaxis_tickformat='f', showlegend=False
                      , yaxis_range=[min_price-y_delta*0.02, max_price+y_delta*0.02])
    fig.update_xaxes(nticks=5, showgrid=True, rangeslider_visible=False, autorange='reversed')
    fig.update_yaxes(tickformat='s', row=2, col=1)
    fig.update_yaxes(nticks=10, row=1, col=1)
    
    cs_colorstyle(fig.data[0], ii='#f52047', il='#f52047', di='#14c0ff', dl='#14c0ff')
    
    # endregion
    # region [PLOTTING ADDITIONAL FEATURES]
    
    for i, n in enumerate(feature_names[2:]):
        if n in ft.columns: y = ft[n].iloc[enum:enum+dnum]
        else: y = df[n].iloc[enum:enum+dnum]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='#000000'
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
    
    
def features(df:pd.DataFrame):
    def lag(df, col_name, lag_n):
        lags = pd.DataFrame(index=df.index)
        for i in range(1, lag_n+1): lags[f'{col_name}_lag_{i}'] = df[col_name].shift(-i)
        return lags
    
    fts = pd.DataFrame(index=df.index)
    return fts


# def display_features(df:pd.DataFrame):
    
    

def modeling(data:dict[str, pd.DataFrame], sub=[], **dl):
    '''
    data: whole dataframes without features
    dl: [sd, ed, test_size]
    return: y_pred_test, y_pred_train, y_test, y_train, r_train, r_test, (sel_num, whole_num)
    '''
    def selection_cond(x: pd.Series):
        cond = x['rate'] > 29 and (x['ma60'] > x['ma120'])
        # cond=True
        return cond
    
    def make_xyr(df:pd.DataFrame):
        ft_names = ['거래량', 'rate'] + [f'rate_lag_{n}' for n in range(1, 11)] + ['ma5disp'] + ['ma20disp']
        ft_exclude = ['수정주가구분', '수정비율', '일자', 'y']
        X = df.loc[:, [ft for ft in ft_names if ft not in ft_exclude]]
        y = df['y']
        r = df['y']
        # breakpoint()
        X.dropna(inplace=True); y.dropna(inplace=True)
        common_index = X.index.intersection(y.index)
        return X.loc[common_index], y.loc[common_index], r.loc[common_index]
    
    @st.cache_data
    def data_selection():  # return X_train, X_test, y_train, y_test, r_train, r_test, selection_rate 
        Xl, yl, rl = [], [], []; whole_num, sel_num = 0, 0; names = ['X', 'y', 'r']
        #
        for k in tqdm((sub if sub else data.keys()), desc="Data Selecting"):
            df = data[k].loc[dl['ed']:dl['sd']]
            wd = pd.concat([df, features(df)], axis=1)
            selected = wd.loc[(bl := wd.apply(selection_cond, axis=1))]
            selected.index = [(i, k) for i in selected.index]
            whole_num += len(bl); sel_num += sum(bl)
            X_part, y_part, r_part = make_xyr(selected)
            for n in names: exec(f"{n}l.append({n}_part)")
        X = pd.concat(Xl, axis=0, ignore_index=False); y = pd.concat(yl, axis=0, ignore_index=False).astype(int)
        r = pd.concat(rl, axis=0, ignore_index=False)
        for n in names: exec(f"{n}.sort_index(inplace=True)")
        X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X, y, r, test_size=dl['test_size'], shuffle=False)
        # st.dataframe(X_test)
        #
        return X_train, X_test, y_train, y_test, r_train, r_test, (sel_num, whole_num)
    
    def classification(X_train, X_test, y_train, y_test):
        model = CatBoostRegressor(iterations=4500, verbose=2000)
        #
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test).tolist()
        y_pred_train = model.predict(X_train).tolist()
        #
        return y_pred_test, y_pred_train
    
    
    X_train, X_test, y_train, y_test, r_train, r_test, sel_tup = data_selection()
    y_pred_test, y_pred_train = classification(X_train, X_test, y_train, y_test)
    #
    return y_pred_test, y_pred_train, y_test, y_train, r_train, r_test, sel_tup


def estimate_eval(y_pred, y_test, eval_list:list):
    result = ''
    if "mse" in eval_list: result += f"MSE: {metrics.mean_squared_error(y_pred=y_pred, y_true=y_test)}\n"
    if "r2" in eval_list: result += f"R^2: {metrics.r2_score(y_pred=y_pred, y_true=y_test)}\n"
    return result


def figure_model_result(y_pred_test, y_pred_train, y_test, y_train, r_train, r_test):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=r_test, y=y_pred_test, mode='markers'), row=1, col=1)
    m, b = np.polyfit(r_test, y_pred_test, 1)
    fig.add_trace(go.Scatter(x=r_test, y=m*r_test+b, mode='lines', name='Regression Line', line=dict(width=3)))
    # fig.add_trace(go.Histogram2dContour(x=r_test, y=y_pred_test), row=1, col=1)
    fig.update_layout(yaxis_range=[-30, 30], xaxis_range=[-30, 30])
    fig.update_traces(marker=dict(size=2))
    fig.add_hline(y=0, line_width=0.5, line_color="red", row=1, col=1)
    fig.add_vline(x=0, line_width=0.5, line_color="red", row=1, col=1)
    #
    return fig
    

@st.cache_data
def load_data(fpath:str, arg:dict={}):
    ldir = os.listdir(fpath)
    files = [fname for fname in ldir if os.path.isfile(os.path.join(fpath, fname))]
    data = {fname.removesuffix('.csv'):pd.read_csv('/'.join([fpath, fname]), **arg) for fname in tqdm(files, desc="Loading Data")}
    # for k in data: data[k].index = pd.to_datetime(data[k]['일자'])
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
    data = load_data("../../FinanceData/DB/AllFeatures"
                     , arg=dict(dtype={'수정주가구분':object, '수정비율':object}, parse_dates=['일자'], index_col='일자'))
    df = data[code]
    ft = features(df); ftname = ['rate']
    wd = pd.concat([df, ft], axis=1)
    # st.dataframe(df)
    ma_options = st.sidebar.multiselect(
        'Select Moving-Averages',
        (1, 5, 20, 60, 120),  # options
        (5, 20))  # default

    # region [PAGE: CHART]
    time_start()

    dnum = st.sidebar.slider('Data Number', 1, 300, value=140)
    enum = st.sidebar.slider('Ending Time', 0, len(df), value=0)
    st.plotly_chart(figure_chart(dnum, enum, df, ft=ft, ftname=ftname, ma=ma_options)
                    , use_container_width=True, config={'displayModeBar': False})

    get_elapsed_time()
    # endregion
    st.subheader('Modeling', divider='rainbow')
    # region [PAGE: MODEL]
    
    if st.button("Fit Model"):
        y_pred_test, y_pred_train, y_test, y_train, r_train, r_test, sel_tup = \
            modeling(data, sub=[], sd=datetime(2015,5,1), ed=datetime(2024,3,1), test_size=0.1)
        st.text(f"{sel_tup[0]} / {sel_tup[1]}")
        st.text(estimate_eval(y_pred_test, y_test, eval_list=['mse', 'r2']))
        st.plotly_chart(figure_model_result(y_pred_test, y_pred_train, y_test, y_train, r_train, r_test)
                        , use_container_width=True, config={'displayModeBar': False})
    
    # endregion


if __name__ == '__main__':
    main()