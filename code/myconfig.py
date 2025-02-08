# Library
from datetime import datetime, timedelta
import exchange_calendars as xcals

ST, ED = datetime(2015,1,1), datetime(2024,12,15)
PATH = "e:/Financial_Data/"
H5PATH = PATH + 'data.h5'
SCHEDULE = xcals.get_calendar('XNAS').schedule.loc[ST:ED][['open', 'close']].map(lambda x: x.tz_localize(None) - timedelta(hours=5))
THZ = ['1m', '5m', '30m', '1d', '1w']
COL_NAME = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
TO_TIMEDELTA = {
    '1m': lambda x: timedelta(minutes=x),
    '5m': lambda x: timedelta(minutes=x*5),
    '30m': lambda x: timedelta(minutes=x*30),
    '1d': lambda x: timedelta(days=x),
    '1w': lambda x: timedelta(weeks=x),
}
UNIT_TS = {
    '1m': 60,
    '5m': 60*5,
    '30m': 60*30,
    '1d': 60*60*24,
    '1w': 60*60*24*7
}