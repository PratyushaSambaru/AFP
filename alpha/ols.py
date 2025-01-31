import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import timedelta


def precompute_predictors(data, tickers, lag=5):
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
    for ticker in tickers:
        predictors = []
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

        data[ticker] = data[ticker].join(predictors_df)

    return data


def construct_predictors_and_outcomes(data, tickers, lag=5, train_ratio=0.75):
    """
    Construct predictors and outcomes for training and testing, with outcomes specific to each ticker.
    """
    predictors = []
    timestamps = []
    outcomes_dict = {}

    # Access predictors for each ticker
    for time_point in data[tickers[0]].index[lag:]:
        row_predictors = []
        for ticker in tickers:
            stock_row = data[ticker].loc[time_point, ['volatility', 'avg_high_low', 'avg_open_close', 
                                                      'returns', 'total_volume', 'volume_volatility']]
            row_predictors.extend(stock_row.values)

        # Add predictors and timestamp
        predictors.append(row_predictors)
        timestamps.append(time_point)

    # Adding predictors for the final period (for prediction)
    last_row_predictors = []
    for ticker in tickers:
        stock_row = data[ticker].iloc[-1][['volatility', 'avg_high_low', 'avg_open_close', 
                                           'returns', 'total_volume', 'volume_volatility']]
        last_row_predictors.extend(stock_row.values)

    next_period_predictors = np.array(last_row_predictors).reshape(1, -1)

    # Compute outcomes for each ticker
    for ticker in tickers:
        outcomes = []
        for time_point in data[ticker].index[lag:]:
            log_return = data[ticker].loc[time_point, 'log_return']
            outcomes.append(1 if log_return > 0 else 0)  # Binary
        outcomes_dict[ticker] = np.array(outcomes)

    predictors = np.array(predictors)

    # Train-test split
    split_idx = int(len(predictors) * train_ratio)
    train_predictors, test_predictors = predictors[:split_idx], predictors[split_idx:]
    train_timestamps, test_timestamps = timestamps[:split_idx], timestamps[split_idx:]

    # Scaling
    scaler = StandardScaler()
    train_predictors = scaler.fit_transform(train_predictors)
    test_predictors = scaler.transform(test_predictors)
    next_period_predictors = scaler.transform(next_period_predictors)

    train_data = {
        "predictors": train_predictors,
        "timestamps": train_timestamps,
    }
    test_data = {
        "predictors": test_predictors,
        "timestamps": test_timestamps,
    }

    return train_data, test_data, outcomes_dict, next_period_predictors


def evaluate_ticker_model(train_data, test_data, outcomes_dict, ticker, next_period_predictors, penalty='l1', C=1.0):
    """
    Train and evaluate a LASSO OLS model for a single ticker.
    """
    train_outcomes = outcomes_dict[ticker][:len(train_data['predictors'])]
    test_outcomes = outcomes_dict[ticker][len(train_data['predictors']):]

    # Logistic Regression
    model = LogisticRegression(penalty=penalty, solver='saga', C=C, max_iter=5000)
    model.fit(train_data['predictors'], train_outcomes)

    # Evaluate model on test set
    y_pred = model.predict(test_data['predictors'])
    accuracy = accuracy_score(test_outcomes, y_pred)

    # Prediction
    predicted_proba = model.predict_proba(next_period_predictors)[0][1]

    return accuracy, predicted_proba


def generate_regression_signal(window_data, tickers, lag=5, train_ratio=0.75, num_stocks=3):
    """
    Generate alpha signals for a single period.

    Args:
        window_data: Data for current time window.
        tickers: Stock tickers.
        lag: Number of minutes to use for predictors.
        train_ratio: Ratio of training to testing data.
        num_stocks: Number of stocks for signals.

    Returns:
        dict: Signals for stocks +-1.
    """
    # Construct predictors and outcomes
    train_data, test_data, outcomes_dict, next_period_predictors = construct_predictors_and_outcomes(
        window_data, tickers, lag, train_ratio
    )

    stock_scores = {}
    for ticker in tickers:
        try:
            accuracy, predicted_proba = evaluate_ticker_model(
                train_data, test_data, outcomes_dict, ticker, next_period_predictors
            )
            stock_scores[ticker] = {"accuracy": accuracy, "predicted_proba": predicted_proba}
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Rank stocks
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    top_stocks = sorted_stocks[:num_stocks]

    # Generate signals
    signals = {}
    for stock, scores in top_stocks:
        if scores['predicted_proba'] > 0.5:
            signals[stock] = 1
        else:
            signals[stock] = -1

    return signals