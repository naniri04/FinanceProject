import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from random import randint
from mplfinance.original_flavor import candlestick2_ohlc
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# stock_list = pd.read_csv('C:/Users/Naniri/Documents/FinanceProject/DB/Original/stock_list.csv')


class mydf(pd.DataFrame):

    def rate(self, stand, val) -> float: return np.round((val-stand)/stand*100, 2)

    def r(self, i, n:int=0) -> int:
        return self.iloc[i-n, 7]

    def c(self, i, n:int=0) -> int:
        return self.iloc[i-n, 5]
    
    def ocr(self, i, n:int=0) -> int:
        return self.rate(self.iloc[i-n, 2], self.iloc[i-n, 5])

    def nmin(self, n:int, i:int) -> int: # i일부터 그 전 n일동안 가격 중 최솟값.
        if n >= i+1: return -1
        return np.min(self.iloc[i-n:i+1, 4])
    
    def ncmin(self, n:int, i:int) -> int: # i일부터 그 전 n일동안 가격 중 종가의 최솟값.
        if n >= i+1: return -1
        return np.min(self.iloc[i-n:i+1, 5])
    
    def ncmax(self, n:int, i:int) -> int: # i일부터 그 전 n일동안 가격 중 종가의 최댓값.
        if n >= i+1: return -1
        return np.max(self.iloc[i-n:i+1, 5])
    
    def nmax(self, n:int, i:int) -> int: # i일부터 그 전 n일동안 가격 중 최댓값.
        if n >= i+1: return -1
        return np.max(self.iloc[i-n:i+1, 3])
    
    def mart(self, n:int, i:int): # 종가와 n-ma 비교
        return self.rate(self.c(i), self.ma(n, i))
    
    def maslope(self, n:int, i:int): # n-ma의 직전 봉과의 기울기 비율
        return self.rate(self.ma(n, i-1), self.ma(n, i))

    def nrt(self, n:int, i:int) -> float: # i에서 n일 전의 값과 i일 값의 비율.
        """
        returns rate of `n` index before i and i.\n
        returns `np.nan` if it doesnt exists at `i`.
        """
        if n >= i+1: return np.nan
        return self.rate(self['Close'].iloc[i-n], self['Close'].iloc[i])

    def ma(self, n:int, i:int) -> float: # i일에서 n일 이동평균. i일 포함.
        """
        returns `n` days moving average.\n
        returns `np.nan` if `n`-ma doesnt exists at `i`.
        """
        if n > i+1: return np.nan
        return np.round(np.mean(a=self['Close'].iloc[i-n+1:i+1]), 1)
    
    def amt(self, i:int) -> int: # i일에서 거래대금(백만)
        """
        returns amount at `i`.
        """
        print(np.mean(self.iloc[i, 1:5])*self.iloc[i, 6].value)
        return np.round((np.mean(self.iloc[i, 1:5])*self.iloc[i, 6].value) // 10**6, 2)
    
    def avg_amt(self, n:int, i:int) -> int: # i일부터 그 전 n일간의 거래대금 평균값.(백만)
        """
        returns average of amount from `i` to `i - n`.\n
        returns `-1` if value doesn't exists.
        """
        if n > i+1: return -1
        return np.round(np.mean([self.amt(i-di) for di in range(n)]), 2)


class myprocess:
    def test(self):
        df = fdr.DataReader('000020')
        df['Date'] = df.index
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']]
        df.set_index(keys=pd.Series([i for i in range(len(df))]), inplace=True)
        # df['Close'].plot()
        # plt.show()
        mdi = mydf(df)
        print(mdi.head(8))
        print(mdi.ma(7, datetime(1999,1,11)))

    def prepros(self, n:int, min_size:int=100):
        whole_stock_df = pd.read_csv('C:/Users/Naniri/Documents/FinanceProject/DB/stock_list.csv')\
            .astype({'Code':'string', 'Name':'string', 'SDate':'datetime64[ns]', 'EDate':'datetime64[ns]'})
        l = []; rl = np.random.choice(range(0, 4023), size=4022, replace=False); i=0
        while len(l) < n:
            if len(rl) == 0: print("no other data for given cond. returns list of len < n."); break

            # condition
            ser = whole_stock_df.iloc[rl[i], 4:6]
            flag = (ser['EDate'] - ser['SDate']).days > min_size * 1.5

            if flag: l.append(rl[i])
            else: pass
            i += 1
        return l

    def find_stock_by_cond(self, rand_date:bool, run_by_string:bool, draw_graph:str, mdd:int, pdd:int,
                                    start_date:datetime=None, end_date:datetime=None, targ_days:int=None,
                                    exec_code:str=None, targ_stock_idx:list=[]):
        """
        if `rand_date == True`, `targ_days` is needed, and run with number of `rand_days` random days.\n
        if `rand_date == False`, it will run from `start_date` to `end_date`, both included. both variables are needed. 

        `draw_graph` could be in [`one`, `whole`, `none`].\n
        `one` = draw graph one by one / `whole` = draw whole graph in one window

        graph is drawn from `d-mdd` to `d+pdd` which condition meets.
        """
        timer.start_timer()
        
        print_days = mdd+pdd+1
        x_axis = np.arange(-mdd, pdd+1, 1) # defining x_axis when draw_graph == 'whole'

        whole_stock_df = pd.read_csv('C:/Users/Naniri/Documents/FinanceProject/DB/stock_list.csv')\
            .astype({'Code':'string', 'Name':'string', 'SDate':'datetime64[ns]', 'EDate':'datetime64[ns]'})
        
        # Determine Condition ----------------------------------------------------------------

        def cond(mdi:mydf, i:int): 
            if i < 15 or mdi.maslope(10, i-3) > 0: return False
            return -8 < mdi.ocr(i, 2) < -1.5 and \
            1.5 < mdi.ocr(i, 1) < 5 and \
            1.5 < mdi.ocr(i, 0) < 8
        
        def rate(stand, val): return np.round((val-stand)/stand*100, 2)

        w_l = []; o_dfl = []

        # calculate --------------------------------------------------------------------------

        checked_day = 0; found_day = 0

        tl = targ_stock_idx if targ_stock_idx else range(len(whole_stock_df))

        for i in tqdm(tl):
            ser = whole_stock_df.loc[i]
            df = pd.read_csv(f'C:/Users/Naniri/Documents/FinanceProject/DB/BackTesting/{ser["Code"]} {ser["Name"]}.csv')
            df = df.astype({'Date': 'datetime64[ns]'})   

            if rand_date:
                if targ_days != -1:
                    idx_list = np.random.choice(range(0, len(df)), size=targ_days, replace=False)
                else:
                    idx_list = list(range(0, len(df)))
            else:
                idx_list = []
                for dd in range((end_date - start_date).days + 1):
                    ti = df[df['Date'] == start_date + timedelta(dd)].index
                    if len(ti.values): idx_list.append(ti.values[0])

            o_dl = pd.DataFrame({'Date': [], 'Low': []}); o_tl = []
            checked_day += len(idx_list)

            for j in idx_list:
                standard_price = df.iloc[j, 5]
                l, h = j-mdd, j+pdd+1

                if l >= 0 and h <= len(df) and not (df.iloc[l:h]['Volume'] == 0).any() and cond(mydf(df), j):
                    found_day += 1
                    if draw_graph == 'whole':
                        w_l.append(list(map(rate, [standard_price]*print_days, df['Close'].iloc[l:h].values)))
                    elif draw_graph == 'one':
                        o_tl.append(df.iloc[j, [1,4]])
                else:
                    pass
            
            o_dl = pd.DataFrame(o_tl)
            if draw_graph == 'one' and len(o_dl): o_dfl.append((df, o_dl))

        # plotting ---------------------------------------------------------------------------
        
        if draw_graph == 'whole':
            if len(w_l) == 0: print('no data for cond.')
            else:
                #region plotting 

                fig = plt.figure(layout="constrained")

                gs = GridSpec(2, 3, figure=fig)
                ax1_1 = fig.add_subplot(gs[0, :])
                ax2_1 = fig.add_subplot(gs[1, 0])
                ax2_2 = fig.add_subplot(gs[1, 1])
                ax2_3 = fig.add_subplot(gs[1, 2])
                w_df = pd.DataFrame(w_l)

                # plot main graph
                ax1_1.grid(color='#eaeaea')
                ax1_1.axvline(x=0, color='#ff5ac5', linewidth=1, alpha=0.6); ax1_1.axhline(y=0, color='#ff5ac5', linewidth=1, alpha=0.6)
                box = ax1_1.boxplot(w_df, positions=x_axis, meanline=True, showmeans=True, flierprops={'marker':'.', 'markersize':3}, \
                                meanprops={'color':'blue', 'linewidth':1.5, 'linestyle':'dotted'}, medianprops={'color':'red', 'linewidth':1.5})
                # vln = ax1_1.violinplot(w_df, positions=x_axis, showextrema=False)
                ax1_1.plot(x_axis, np.mean(w_df, axis=0), linewidth=0.5, color='blue', marker='.', markersize=2)
                ax1_1.set_yticks((lambda x: np.concatenate([x, np.zeros((1)), np.negative(x)]))(
                    np.concatenate([np.arange(-200, -30, 10), np.arange(-30, -10, 5), np.arange(-10, 0, 2)])))
                ax1_1.set_ylim([-25, 25])
                
                # plot hl histogram
                max_ser, min_ser = np.max(w_df.iloc[:, mdd:], axis=1), np.min(w_df.iloc[:, mdd:], axis=1)
                try: max_0_cnt = max_ser.value_counts()[0]
                except: max_0_cnt = 0
                try: min_0_cnt = min_ser.value_counts()[0]
                except: min_0_cnt = 0

                ax2_2.text(0.02, 0.95, f"Rate of max == 0: {np.round(max_0_cnt / len(max_ser), 3)}", transform=ax2_2.transAxes, bbox={'facecolor':'white'})
                ax2_2.hist(max_ser, color='#6fa8dc', bins=np.arange(0, 30.5, 0.5)); ax2_2.set_xticks(np.arange(0, 31, 1))
                ax2_2.set_title("Highest"); ax2_2.grid(color='#adadad'); ax2_2.margins(x=0, y=0.08)

                ax2_1.text(0.02, 0.95, f"Rate of min == 0: {np.round(min_0_cnt / len(min_ser), 3)}", transform=ax2_1.transAxes, bbox={'facecolor':'white'})
                ax2_1.hist(min_ser, color='#6fa8dc', bins=np.arange(-30, 0.5, 0.5)); ax2_1.set_xticks(np.arange(-30, 1, 1))
                ax2_1.set_title("Lowest"); ax2_1.grid(color='#adadad'); ax2_1.yaxis.tick_right(); ax2_1.margins(x=0, y=0.08)

                ax2_3.scatter(max_ser, min_ser, s=4, c='blue', alpha=0.5)
                ax2_3.set_xticks(np.arange(0, 32, 2)); ax2_3.set_yticks(np.arange(-30, 2, 2)); ax2_3.grid(color='black', alpha=0.3)
                ax2_3.set_ylim(-30.5, 0.5); ax2_3.set_xlim(-0.5, 30.5)

                #endregion

            print(f"len: {len(w_df)}")
        
        elif draw_graph == 'one':
            if len(o_dfl) >= 20: print("too many dfs. try again with small amount of stocks.")
            else:
                #region plotting 
                for i in range(len(o_dfl)):
                    df, dl = o_dfl[i]
                    fig, ax = plt.subplots()

                    # x-축 날짜
                    xdate = df.Date.astype('str') 

                    # 종가 및 5,20,60,120일 이동평균
                    ax.plot(xdate, df['Close'], label='Close',linewidth=0.7,color='k')
                    ax.plot(xdate, df['Close'].rolling(window=5).mean(), label="5ma",linewidth=1)
                    ax.plot(xdate, df['Close'].rolling(window=10).mean(), label="10ma",linewidth=1)
                    ax.plot(xdate, df['Close'].rolling(window=20).mean(), label="20ma",linewidth=1)
                    ax.plot(xdate, df['Close'].rolling(window=60).mean(), label="60ma",linewidth=1)
                    ax.plot(xdate, df['Close'].rolling(window=120).mean(), label="120ma",linewidth=1)

                    candlestick2_ohlc(ax,df['Open'],df['High'],df['Low'],df['Close'], width=0.5, colorup='r', colordown='b')

                    dl = dl.astype({'Date':'string'})

                    ymin, ymax = ax.get_ylim(); tv = (ymax-ymin)/100
                    ax.scatter(dl['Date'], (dl['Low']-tv), s=60, color='#58e5ff', edgecolor='#3fe1ff', marker='^')
                    ax.scatter(dl['Date'], [0] * len(dl), marker='_', color='k')
                        
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(25))
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(20))
                    ax.legend(loc=1) # legend 위치
                    plt.xticks(rotation = 45) # x-축 글씨 45도 회전

                plt.grid(color='#eaeaea', linestyle='dashed') # 그리드 표시

                #endregion

        print(f"[ {found_day} / {checked_day} , {np.round(found_day/checked_day*100, 4)}% ]")

        timer.stop_timer()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        plt.show()


if __name__ == "__main__":
    print()
    # pltstyle.use('dark_background')
    mpi = myprocess()
    mpi.find_stock_by_cond(rand_date=True, run_by_string=False, draw_graph='whole', mdd=3, pdd=15,
                           start_date=datetime(2010,1,1), end_date=datetime(2023,3,22),
                           targ_days=500, targ_stock_idx=mpi.prepros(200, min_size=500)) # mpi.prepros(100)
