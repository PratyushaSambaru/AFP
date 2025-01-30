import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def compute_c1(stock1_returns, stock2_returns, max_lag):
    """
    Compute the C1 lead-lag score for two stocks using the maximum lagged Pearson correlation.
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")

    max_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Stock 1 leads Stock 2
            shifted_corr = stock1_returns[:lag].corr(stock2_returns[-lag:])
        elif lag > 0:
            # Stock 2 leads Stock 1
            shifted_corr = stock1_returns[lag:].corr(stock2_returns[:-lag])
        else:
            # No lag
            shifted_corr = stock1_returns.corr(stock2_returns)

        if shifted_corr > max_corr:
            max_corr = shifted_corr

    return max_corr


def compute_c2(stock1_returns, stock2_returns, max_lag, weights=None):
    """
    Compute the C2 lead-lag score for two stocks using the weighted sum of lagged correlations.
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")

    if weights is None:
        weights = np.ones(2 * max_lag + 1) / (2 * max_lag + 1)
    elif len(weights) != 2 * max_lag + 1:
        raise ValueError("Weights must have a length of 2 * max_lag + 1.")

    lagged_correlations = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Stock 1 leads Stock 2
            corr = stock1_returns[:lag].corr(stock2_returns[-lag:])
        elif lag > 0:
            # Stock 2 leads Stock 1
            corr = stock1_returns[lag:].corr(stock2_returns[:-lag])
        else:
            # No lag
            corr = stock1_returns.corr(stock2_returns)

        lagged_correlations.append(corr)

    lagged_correlations = np.array(lagged_correlations)
    weighted_sum = np.dot(lagged_correlations, weights)

    return weighted_sum


def compute_levy_area(stock1_returns, stock2_returns):
    """
    Compute the Lévy-area score for two stocks to capture nonlinear dependencies.
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")

    # Compute the iterated integrals (cross-integrals)
    cross_integral_1 = np.cumsum(stock1_returns) @ stock2_returns
    cross_integral_2 = np.cumsum(stock2_returns) @ stock1_returns

    # Lévy-area computation
    levy_area = 0.5 * (cross_integral_1 - cross_integral_2)

    return levy_area


def construct_lasso_lead_lag_matrix(df, max_lag=5, alpha=0.01):
    """
    Construct a lead-lag matrix using LASSO regression.
    """
    stocks = df.columns.get_level_values(0).unique()
    n_stocks = len(stocks)
    lead_lag_matrix = np.zeros((n_stocks, n_stocks))

    for i, target_stock in enumerate(stocks):
        # Target: next-minute return of the current stock
        y = df[target_stock]['log_return'].shift(-1).iloc[max_lag:-1].dropna()

        # Predictor: past returns of all other stocks
        X = []
        for lag in range(1, max_lag + 1):
            lagged_returns = df.loc[:, pd.IndexSlice[:, 'log_return']].shift(lag).iloc[max_lag:-1]
            X.append(lagged_returns)

        X = pd.concat(X, axis=1).dropna()
        y = y.loc[X.index]

        # Exclude the target stock in predictors
        predictors = [col for col in X.columns if col[0] != target_stock]
        X = X[predictors]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LASSO
        lasso = Lasso(alpha=alpha, fit_intercept=True)
        lasso.fit(X_scaled, y)

        # Get betas
        for j, other_stock in enumerate(stocks):
            if other_stock != target_stock:
                # Sum of all betas
                coefficients = [
                    lasso.coef_[k] for k, col in enumerate(predictors) if col[0] == other_stock
                ]
                lead_lag_matrix[i, j] = np.sum(coefficients)

    # Ensure skew-symmetry
    for i in range(n_stocks):
        for j in range(n_stocks):
            if i != j:
                if lead_lag_matrix[j, i] != -lead_lag_matrix[i, j]:
                    print(f"Skew symmetry violated: {(j, i)}: {lead_lag_matrix[j, i]}, {(i, j)}: {-lead_lag_matrix[i, j]}")
                lead_lag_matrix[j, i] = -lead_lag_matrix[i, j]

    lead_lag_df = pd.DataFrame(lead_lag_matrix, index=stocks, columns=stocks)
    return lead_lag_df


def construct_lead_lag_matrix(df, method, max_lag=5, weights=None, alpha=0.01):
    """
    Construct a skew-symmetric lead-lag matrix for stock returns based on the method.
    """
    stocks = df.columns.get_level_values(0).unique()
    n_stocks = len(stocks)
    lead_lag_matrix = np.zeros((n_stocks, n_stocks))

    if method == 'C1':
        compute_score = lambda stock1, stock2: compute_c1(stock1, stock2, max_lag)
    elif method == 'C2':
        compute_score = lambda stock1, stock2: compute_c2(stock1, stock2, max_lag, weights)
    elif method == 'Levy':
        compute_score = lambda stock1, stock2: compute_levy_area(stock1, stock2)
    elif method == 'LASSO':
        return construct_lasso_lead_lag_matrix(df, max_lag, alpha)
    else:
        raise ValueError("Invalid method. Choose from 'C1', 'C2', 'Levy', or 'LASSO'.")

    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            if i != j:
                stock1_returns = df[stock1]['normalized_return']
                stock2_returns = df[stock2]['normalized_return']
                score = compute_score(stock1_returns, stock2_returns)
                lead_lag_matrix[i, j] = score
                lead_lag_matrix[j, i] = -score

    lead_lag_df = pd.DataFrame(lead_lag_matrix, index=stocks, columns=stocks)
    return lead_lag_df


def rank_stocks(window_data, method, max_lag):
    """
    Rank stocks based on the average of their column values in the lead-lag matrix.
    If the method requires a lead-lag matrix, it computes it first.
    """
    if method in ['C1', 'C2', 'Levy', 'LASSO']:
        # Compute lead-lag matrix 
        lead_lag_matrix = construct_lead_lag_matrix(window_data, method, max_lag)
        scores = lead_lag_matrix.mean(axis=0)
    else:
        # Space for alternative ranking method
        scores = window_data.loc[:, pd.IndexSlice[:, 'log_return']].mean(axis=0)

    ranked_stocks = pd.DataFrame({'Stock': scores.index, 'Score': scores.values})
    ranked_stocks = ranked_stocks.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return ranked_stocks

