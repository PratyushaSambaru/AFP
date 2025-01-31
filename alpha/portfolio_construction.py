import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_utils import *
from alpha.lead_lag import *
from alpha.ols import *

import pandas as pd
import numpy as np
from datetime import timedelta

def generate_alpha_signals(data, method, lookback=60, max_lag=5, num_stocks=3):
    """
    Generate alpha signals for a single day using the specified method.
    """
    if method in ['C1', 'C2', 'Levy', 'LASSO']:
        return generate_lead_lag_alpha_signals(data, method, lookback, max_lag, num_stocks)
    elif method == 'OLS':
        return generate_regression_alpha_signals(data, 120, 3, num_stocks)
    else:
        # Space for alternative signal method
        return None

def generate_lead_lag_alpha_signals(data, method, lookback=60, max_lag=5, num_stocks=3):
    """
    Generate lead-lag alpha signals for a single day using the specified method.
    """
    tickers = [key for key in data.keys() if key != 'SPY']
    merged_df = pd.concat([data[ticker] for ticker in tickers], axis=1, keys=tickers)
    follower_signals = pd.DataFrame(index=merged_df.index[lookback + 1:], columns=tickers)

    for time_point in merged_df.index[lookback + 1:]:
        window_data = merged_df.loc[time_point - timedelta(minutes=lookback + 1):time_point - timedelta(minutes=1)]
        ranked_stocks = rank_stocks(window_data, method, max_lag)

        num_stocks = min(len(ranked_stocks) // 2, num_stocks)

        leaders = ranked_stocks.head(num_stocks)['Stock'].tolist()  # Top leaders
        followers = ranked_stocks.tail(num_stocks)['Stock'].tolist()  # Bottom followers

        leader_returns = window_data.loc[:, pd.IndexSlice[leaders, 'log_return']].iloc[-2].mean()
        signal = 1 if leader_returns > 0 else -1

        for follower in followers:
            follower_signals.at[time_point, follower] = signal

    return follower_signals


def generate_regression_alpha_signals(data, lookback=60, max_lag=5, num_stocks=3, train_ratio=0.75):
    """
    Generate alpha signals for a single day using a generalized linear regression (LASSO) model.
    """
    tickers = [key for key in data.keys() if key != 'SPY']
    data = precompute_predictors(data, tickers, lag=max_lag)
    merged_df = pd.concat([data[ticker] for ticker in tickers], axis=1, keys=tickers)
    follower_signals = pd.DataFrame(index=merged_df.index[lookback + 1:], columns=tickers)

    for time_point in merged_df.index[lookback + 1:]:
        window_data = merged_df.loc[time_point - timedelta(minutes=lookback):time_point - timedelta(minutes=1)]

        # Signals for time
        signals = generate_regression_signal(
            window_data=window_data,
            tickers=tickers,
            lag=max_lag,
            train_ratio=train_ratio,
            num_stocks=num_stocks
        )

        if signals is None:
            continue

        for stock, signal in signals.items():
            follower_signals.at[time_point, stock] = signal

    return follower_signals


def construct_portfolio(data, spy_data, follower_signals, holding_period=1):
    """
    Construct a zero-cost portfolio using alpha signals and SPY as offset.
    """
    tickers = [key for key in data.keys() if key != 'SPY']
    merged_df = pd.concat([data[ticker] for ticker in tickers], axis=1, keys=tickers)
    portfolio_returns = []
    alpha_returns = []

    for time_point in follower_signals.index:
        # Get active followers and their signals
        active_signals = follower_signals.loc[time_point].dropna()
        if active_signals.empty:
            portfolio_returns.append(0)
            alpha_returns.append(0)
            continue
        followers = active_signals.index.tolist()
        signal = active_signals.values

        # Compute follower return
        follower_returns = merged_df.loc[time_point:time_point + timedelta(minutes=holding_period - 1),
                                         pd.IndexSlice[followers, 'log_return']].mean()
        follower_return = (signal * follower_returns).mean()

        # Compute SPY offset return
        spy_return = - signal[0] * spy_data.loc[time_point:time_point + timedelta(minutes=holding_period - 1), 'log_return'].mean()

        portfolio_return = (follower_return + spy_return) / 2

        portfolio_returns.append(portfolio_return)
        alpha_returns.append(follower_return)

    return pd.Series(portfolio_returns, index=follower_signals.index), pd.Series(alpha_returns, index=follower_signals.index)


def run_alpha_and_portfolio(data, spy_data, method, date, holding_period=1):
    """
    Run alpha signal generation and portfolio construction for a single day.

    - data: Dict of dfs with tickers as keys.
    - spy_data: df of SPY data.
    - method: Alpha signal.
    - date: Analysis date.
    - holding_period: Num periods to hold.

    - portfolio_returns: Portfolio returns for the day.
    - alpha_returns: Alpha (unhedged) returns for the day.
    - follower_signals: Alpha signals (+-1) for each stock and time point.
    """
    daily_data = {ticker: df[df.index.date == pd.Timestamp(date).date()] for ticker, df in data.items()}
    daily_spy_data = spy_data[spy_data.index.date == pd.Timestamp(date).date()]

    if all(len(df) == 0 for df in daily_data.values()) or len(daily_spy_data) == 0:
        print(f"No data available for {date}, skipping...")
        return None, None, None

    print(f"Generating alpha signals for {date} using method {method}...")
    follower_signals = generate_alpha_signals(daily_data, method)

    print(f"Constructing portfolio for {date}...")
    portfolio_returns, alpha_returns = construct_portfolio(
        data=daily_data,
        spy_data=daily_spy_data,
        follower_signals=follower_signals,
        holding_period=holding_period
    )

    return portfolio_returns, alpha_returns, follower_signals



