import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

def portfolio_performance_report(portfolio_returns, trading_days=252):
    """
    Generate performance metrics and plots for portfolio returns.
    """
    unique_days = portfolio_returns.index.normalize().nunique()
    trading_minutes = len(portfolio_returns) / unique_days

    cumulative_returns = (1 + portfolio_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(trading_days * trading_minutes)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

    print("\nPerformance Report:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Cumulative Returns")
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.show()

def alpha_accuracy_report(signals, actual_returns):
    """
    Calculate and display alpha accuracy metrics.
    """
    correct_predictions = (signals * actual_returns > 0).sum()
    total_predictions = len(signals)
    accuracy = correct_predictions / total_predictions

    print("\nAlpha Accuracy Report:")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2%}")


def plot_cumulative_returns_with_signals(merged_df, follower_signals, tickers, method, spy):
    """
    Plot cumulative returns with signals dynamically coloring follower stocks.
    Annotate the end of each line with the ticker name instead of using a legend.
    """
    plt.figure(figsize=(12, 8))

    # Plot SPY cumulative returns
    spy_cumulative_returns = (1 + spy['log_return']).cumprod()
    plt.plot(spy_cumulative_returns, label="SPY", color="black", linestyle="--", linewidth=1.5)

    # Plot each ticker's cumulative returns
    for ticker in tickers:
        cumulative_returns = (1 + merged_df.loc[:, pd.IndexSlice[ticker, 'log_return']]).cumprod()
        plt.plot(cumulative_returns, color="grey", linewidth=0.8, alpha=0.7)

        last_time = cumulative_returns.index[0]
        for time_point in cumulative_returns.index:
            if time_point in follower_signals.index and not pd.isna(follower_signals.at[time_point, ticker]):
                signal = follower_signals.at[time_point, ticker]
                color = 'green' if signal == 1 else 'red'
                plt.plot([last_time, time_point], cumulative_returns.loc[[last_time, time_point]], color=color, linewidth=1.5)
            last_time = time_point

        # Add arrow annotation at the end of the line
        end_x = cumulative_returns.index[-1]
        end_y = cumulative_returns.iloc[-1]
        plt.annotate(
            ticker,
            xy=(end_x, end_y),
            xytext=(10, 0),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            fontsize=10,
            color="black"
        )

    plt.title(f"Cumulative Returns with Follower Signals ({method})")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    # Remove the legend since we're adding arrows
    plt.grid()
    plt.show()

