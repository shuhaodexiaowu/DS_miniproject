from stockstats import StockDataFrame as Sdf

# load dataset
import numpy as np
# import pickle
import pandas as pd

def add_technical_indicator():
    data = pd.read_pickle('.\Dataset_B02_10mins_OHLCV.pkl')
    stockstats_df = Sdf.retype(data.copy())
    data['cci'] =stockstats_df['cci']
    data['kdj'] =stockstats_df['kdjk']
    data['macd'] = stockstats_df['macdh']
    data.fillna(method='bfill',inplace=True)
    data =data.round(4)

    return data


