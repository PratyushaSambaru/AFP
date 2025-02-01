import os
import sys
from datetime import timedelta
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

from data.data_utils import *
from alpha.lead_lag import *
from alpha.performance_evaluation import *
from alpha.portfolio_construction import *

def testData(tickers, date_range, method):
    data_folder = "C:\\Research\\haas-mfe-g2-cg\\data\\ohlcv"
    print("Loading and preprocessing data...")
    data = load_data_for_tickers(tickers, date_range, data_folder)
    return precompute_predictors(data,tickers, method)

def precompute_predictors(data_df, tickers, method, lag=5):
    """
    Precompute predictors for all tickers over the entire dataset.

    We add predictors across the last lag datapoints:
     - volatility
     - avg high low spread
     - avg open close spread
     - returns
     - total volume
     - volatility of volume
    """
    data = data_df.copy()
    edge_matrices = generate_lead_lag_matrix(data,method=method)
    output_data = pd.DataFrame()
    for ticker in tickers:
        predictors = []
        edge_data = []
        out_df = {}
        
        timestamps = data[ticker].index[lag:]

        for time_point in timestamps:
            # Segment the data to look at the last lag periods of data
            stock_data = data[ticker].loc[time_point - timedelta(minutes=lag):time_point - timedelta(minutes=1)]
            if len(stock_data) < lag:
                predictors.append([np.nan] * 6)
                continue

            # Compute predictors
            volatility = stock_data['log_return'].std()
            avg_high_low = (stock_data['high'] - stock_data['low']).mean()
            avg_open_close = (stock_data['open'] - stock_data['close']).mean()
            returns = stock_data['log_return'].sum()
            total_volume = stock_data['volume'].sum()
            volume_volatility = stock_data['volume'].std()

            predictors.append([volatility, avg_high_low, avg_open_close, returns, total_volume, volume_volatility])

        # Add predictors as new columns
        predictor_names = ['volatility', 'avg_high_low', 'avg_open_close', 'returns', 'total_volume', 'volume_volatility']
        predictors_df = pd.DataFrame(predictors, columns=predictor_names, index=timestamps)

        ticker_node_data = data[ticker].join(predictors_df)
        corr_df = pd.DataFrame({
            timestamp: matrix.loc[ticker]
            for timestamp, matrix in edge_matrices.items()
        })
        edge_data = corr_df.T
        edge_data = edge_data.drop(columns=[ticker])

        common_timestamps = ticker_node_data.index.intersection(edge_data.index)
        ticker_node_data = ticker_node_data.loc[common_timestamps]
        edge_data = edge_data.loc[common_timestamps]

        out_df['Ticker_Node_Meta_Data'] = ticker_node_data
        out_df['Ticker_Edge_Meta_Data'] = edge_data
        #data[ticker] = data[ticker].join(predictors_df)
        output_data[ticker] = out_df
        
    return output_data

def generate_lead_lag_matrix(data, method, lookback=30, max_lag=5):
    """
    Generate lead-lag matrix for a single day using the specified method.
    """
    tickers = [key for key in data.keys() if key != 'SPY']
    merged_df = pd.concat([data[ticker] for ticker in tickers], axis=1, keys=tickers)
    output_dict = {}
    for time_point in merged_df.index[lookback + 1:]:
        window_data = merged_df.loc[time_point - timedelta(minutes=lookback + 1):time_point - timedelta(minutes=1)]
        matrix = construct_lead_lag_matrix(window_data, method, max_lag)
        output_dict[time_point] = matrix
    return output_dict

def main():
    tickers = [
    # Consumer Discretionary - Large-Cap
    "AMZN", "TSLA", "HD", "MCD",
    # Consumer Discretionary - Mid-Cap
    #"ROKU", "YUM", "DLTR", "BURL",
    # Consumer Discretionary - Small-Cap
    # "CROX", "PLNT", "AEO", "PLAY"
    ]
    date_range = ("2024-09-25", "2024-09-30")
    result = testData(tickers, date_range, method='C2')
    print("Processed Data: ", result)

if __name__ == "__main__":
    main()