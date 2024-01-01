import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import timer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas import DataFrame
import itertools

x = np.linspace(1, 4, 100)

# 그래프 그리기
plt.plot(x, x, label='x=y')

# 점 찍기
plt.scatter([2, 4], [3, 1], color='red', label='Points')

# 범례 추가
plt.legend()

# 그래프 보여주기
plt.show()
