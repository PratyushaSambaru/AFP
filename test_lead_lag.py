import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_utils import *
from alpha.lead_lag import *
from alpha.performance_evaluation import *
from alpha.portfolio_construction import *

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# temporary file to test the portfolio construction for the initial attempt to replicate the lead lag alphas


def test_portfolio_construction_multi_day(plot_signals):
    """
    Test portfolio construction, alpha signal generation, and portfolio evaluation for multi-day data.
    """
    # Define the tickers and data folder
    tickers = [
        # Consumer Discretionary - Large-Cap
        "AMZN", "TSLA", "HD", "MCD",
        # Consumer Discretionary - Mid-Cap
        #"ROKU", "YUM", "DLTR", "BURL",
        # Consumer Discretionary - Small-Cap
        # "CROX", "PLNT", "AEO", "PLAY"
    ]
    date_range = ("2024-09-03", "2024-09-30")
    data_folder = "data/ohlcv"

    # Load data
    print("Loading and preprocessing data...")
    data = load_data_for_tickers(tickers, date_range, data_folder)
    spy_data = data.pop('SPY')

    methods = ['OLS']  # C1, C2, Levy, LASSO
    portfolio_results = {}

    # each methos
    for method in methods:
        print(f"\nTesting method: {method}")
        cumulative_portfolio_returns = []
        cumulative_alpha_returns = []
        cumulative_follower_signals = pd.DataFrame()

        # each date
        unique_dates = pd.date_range(start=date_range[0], end=date_range[1]).date
        for current_date in unique_dates:
            print(f"Processing date: {current_date}")

            # alpha gen and portfolio construction
            daily_portfolio_returns, daily_alpha_returns, daily_follower_signals = run_alpha_and_portfolio(
                data=data, 
                spy_data=spy_data, 
                method=method, 
                date=current_date, 
                holding_period=1
            )

            if daily_portfolio_returns is None:
                continue

            cumulative_portfolio_returns.append(daily_portfolio_returns)
            cumulative_alpha_returns.append(daily_alpha_returns)
            cumulative_follower_signals = pd.concat([cumulative_follower_signals, daily_follower_signals])

            if plot_signals:
                print(f"Plotting cumulative returns with signals for {current_date}...")
                plot_cumulative_returns_with_signals(
                    merged_df=pd.concat([data[ticker][data[ticker].index.date == current_date] for ticker in tickers], 
                                        axis=1, keys=tickers),
                    follower_signals=daily_follower_signals,
                    tickers=tickers,
                    method=method,
                    spy=spy_data[spy_data.index.date == current_date]
                )

        cumulative_portfolio_returns_series = pd.concat(cumulative_portfolio_returns)
        cumulative_alpha_returns_series = pd.concat(cumulative_alpha_returns)

        portfolio_results[method] = {
            'portfolio_returns': cumulative_portfolio_returns_series,
            'alpha_returns': cumulative_alpha_returns_series,
            'follower_signals': cumulative_follower_signals
        }

        # Evaluation
        print(f"\nPerformance Report for {method}:")
        portfolio_performance_report(portfolio_results[method]['portfolio_returns'])

        # Alpha Evaluation
        print(f"\nAlpha Accuracy Report for {method}:")
        aggregated_signals = portfolio_results[method]['follower_signals'].mean(axis=1).apply(np.sign)
        alpha_accuracy_report(aggregated_signals, portfolio_results[method]['alpha_returns'])


test_portfolio_construction_multi_day(False)