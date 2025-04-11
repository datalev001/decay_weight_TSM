# learned decay_param
import pandas as pd  
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

def _prepare_dataframe(DF: pd.DataFrame, S: str, time_unit: str, p: int, ma_window: int, add_seasonal: bool):
    """
    Prepares the dataframe:
      - Converts the time_unit column to datetime and sorts the DataFrame.
      - Creates AR lag features: Y_{t-1}, ..., Y_{t-p}.
      - Creates a moving average (MA) feature over the previous ma_window observations.
      - If add_seasonal is True, extracts the month from the time column and creates dummy variables.
    Drops rows with missing predictors.
    
    Returns:
      df: the augmented DataFrame.
      predictors: a list of predictor column names (in order).
    """
    # Drop rows with missing S or time_unit, and convert time column to datetime.
    df = DF.copy().dropna(subset=[S, time_unit]).reset_index(drop=True)
    if not np.issubdtype(df[time_unit].dtype, np.datetime64):
        df[time_unit] = pd.to_datetime(df[time_unit], errors='coerce')
    df.sort_values(by=time_unit, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Create AR lag features
    for lag in range(1, p + 1):
        df[f"{S}_lag{lag}"] = df[S].shift(lag)
        
    # Create Moving Average (MA) feature: average of the previous ma_window observations, shifted by one.
    df["MA"] = df[S].rolling(window=ma_window, min_periods=ma_window).mean().shift(1)
    
    # Create seasonal dummies based on month if requested.
    seasonal_cols = []
    if add_seasonal:
        df["month"] = df[time_unit].dt.month
        month_dummies = pd.get_dummies(df["month"], prefix="mon", drop_first=True)
        df = pd.concat([df, month_dummies], axis=1)
        seasonal_cols = list(month_dummies.columns)
    
    # Drop rows with missing predictors (lags or MA)
    df.dropna(subset=[f"{S}_lag{p}", "MA"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Build list of predictor columns: AR lags + MA + seasonal dummies
    predictors = [f"{S}_lag{i}" for i in range(1, p+1)] + ["MA"] + seasonal_cols
    
    # Convert all predictor columns explicitly to float.
    df[predictors] = df[predictors].apply(pd.to_numeric, errors='coerce')
    df[predictors] = df[predictors].astype(float)
    # Drop any rows where predictors are NaN after conversion.
    df.dropna(subset=predictors, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, predictors

def _loss_for_decay_param(alpha, y_train, X_train_const):
    """
    Computes the weighted mean squared error for a given decay parameter.
    """
    idx = np.arange(len(y_train))
    w = alpha ** ((len(y_train) - 1) - idx)
    # Here, we use the weight directly in the MSE calculation.
    model = sm.WLS(y_train, X_train_const, weights=np.sqrt(w)).fit()
    residuals = y_train - model.predict(X_train_const)
    weighted_mse = np.mean(w * residuals**2)
    return weighted_mse

def _decay_weighted_forecast_single(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True,
    verbose: bool = False
):
    """
    Internal function to compute decay-weighted AR(p) forecast with additional MA and seasonal features.
    The decay factor is learned by minimizing a weighted loss function.
    Returns forecasted average, MAPE, and the learned decay parameter.
    """
    # 1. Prepare the DataFrame with predictors.
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window, add_seasonal)
    
    # 2. Train/Test Split.
    n = len(df)
    train_size = int(0.8 * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    # 3. Weighted OLS Estimation with learned decay parameter.
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    
    # Learn decay parameter by minimizing the weighted loss.
    res = minimize(lambda alpha: _loss_for_decay_param(alpha, y_train, X_train_const), 
                   x0=[0.9], bounds=[(0.01, 0.99)])
    decay_param = res.x[0]
    if verbose:
        print(f"Learned decay_param = {decay_param:.4f}")
    
    # Compute weights using the learned decay parameter.
    idx = np.arange(len(X_train))
    if verbose:
        print(f"Using decay function 'exponential'.")
    w = decay_param ** ((len(X_train) - 1) - idx)
    w_sqrt = np.sqrt(w)
    wls_model = sm.WLS(y_train, X_train_const, weights=w_sqrt).fit()
    
    # 4. Iterative Forecasting.
    last_known = list(train_df[S].values.astype(float))
    preds = []
    for i in range(K):
        # AR lag features: the last p values.
        ar_vals = np.array(last_known[-p:], dtype=float)
        # MA: computed from the last ma_window values.
        if len(last_known) >= ma_window:
            ma_val = np.mean(last_known[-ma_window:])
        else:
            ma_val = np.mean(last_known)
        # Seasonal features: use forecast_section row i if applicable.
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        # Combine in the order: AR lags, MA, seasonal dummies.
        x_row = np.concatenate([ar_vals, [ma_val], np.array(seasonal_vals, dtype=float)])
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = wls_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_decay"] = preds
    forecast_avg = forecast_section["pred_decay"].mean()
    
    def mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    mape_val = mape(forecast_section[S].values, forecast_section["pred_decay"].values)
    
    return forecast_avg, mape_val, decay_param

def decay_weighted_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    decay_function: str = "exponential",
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    Decay-weighted AR(p) forecasting with additional MA and seasonal features.
    The decay factor is learned via minimizing a weighted loss function.
    Returns: (forecasted average, MAPE, learned decay parameter)
    """
    return _decay_weighted_forecast_single(DF, S, K, time_unit, p, ma_window, add_seasonal, verbose)

def normal_ar_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    Normal (unweighted) AR(p) forecasting using standard OLS with additional MA and seasonal features.
    Returns forecasted average over K steps and MAPE.
    """
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window, add_seasonal)
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    if verbose:
        print("\n--- Normal OLS Summary ---")
        print(ols_model.summary())
    
    last_known = list(train_df[S].values.astype(float))
    preds = []
    for i in range(K):
        ar_vals = np.array(last_known[-p:], dtype=float)
        if len(last_known) >= ma_window:
            ma_val = np.mean(last_known[-ma_window:])
        else:
            ma_val = np.mean(last_known)
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        x_row = np.concatenate([ar_vals, [ma_val], np.array(seasonal_vals, dtype=float)])
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = ols_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_normal"] = preds
    forecast_avg = forecast_section["pred_normal"].mean()
    mape_val = np.mean(np.abs((forecast_section[S].values - forecast_section["pred_normal"].values) /
                                (forecast_section[S].values + 1e-9))) * 100
    if verbose:
        print(f"\n[Normal AR] Forecasted Average (over {K} steps): {forecast_avg:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
    
    return forecast_avg, mape_val

def arma_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    ARMA(1,1, cycle) forecasting using statsmodels ARIMA with order (1,0,1).
    If add_seasonal is True, seasonal (month) dummy variables are used as exogenous regressors.
    Returns forecasted average over K steps and MAPE.
    """
    df = DF.copy().dropna(subset=[S, time_unit])
    if not np.issubdtype(df[time_unit].dtype, np.datetime64):
        df[time_unit] = pd.to_datetime(df[time_unit], errors='coerce')
    df.sort_values(by=time_unit, inplace=True)
    df.reset_index(drop=True, inplace=True)

    exog = None
    if add_seasonal:
        df["month"] = df[time_unit].dt.month
        exog = pd.get_dummies(df["month"], prefix="mon", drop_first=True).astype(float)
    
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    
    if add_seasonal:
        exog_train = exog.iloc[:train_size]
        exog_test = exog.iloc[train_size:train_size+K]
    else:
        exog_train = None
        exog_test = None
    
    arma_model = sm.tsa.ARIMA(train_df[S], order=(1,0,1), exog=exog_train).fit()
    if verbose:
        print("\n--- ARMA(1,1, cycle) Model Summary ---")
        print(arma_model.summary())
    
    forecast = arma_model.forecast(steps=K, exog=exog_test)
    forecast_vals = forecast.values.astype(float)
    actual = test_df[S].iloc[:K].values.astype(float)
    mape_val = np.mean(np.abs((actual - forecast_vals) / (actual + 1e-9))) * 100
    forecast_avg = np.mean(forecast_vals)
    
    return forecast_avg, mape_val

def cycle_ar_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    p: int = 1,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    AR with cycle forecasting (no MA) using standard OLS on AR lag features and optional seasonal dummies.
    Returns forecasted average over K steps and MAPE.
    """
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window=3, add_seasonal=add_seasonal)
    if "MA" in predictors:
        predictors.remove("MA")
        df.drop(columns=["MA"], inplace=True)
    
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    if verbose:
        print("\n--- AR with Cycle (no MA) OLS Summary ---")
        print(ols_model.summary())
        
    last_known = list(train_df[S].values.astype(float))
    preds = []
    for i in range(K):
        ar_vals = np.array(last_known[-p:], dtype=float)
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        if add_seasonal:
            x_row = np.concatenate([ar_vals, np.array(seasonal_vals, dtype=float)])
        else:
            x_row = ar_vals
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = ols_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_cycle_ar"] = preds
    forecast_avg = forecast_section["pred_cycle_ar"].mean()
    mape_val = np.mean(np.abs((forecast_section[S].values - forecast_section["pred_cycle_ar"].values) /
                                (forecast_section[S].values + 1e-9))) * 100
    if verbose:
        print(f"\n[AR with Cycle (no MA)] Forecasted Average (over {K} steps): {forecast_avg:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
    
    return forecast_avg, mape_val

# === Execution Code Using Electric_Production_tm.csv ===
if __name__ == "__main__":
    # Load dataset (expects columns: 'DATE', 'IPG2211A2N')
    df = pd.read_csv("Electric_Production_tm.csv")  # file with columns ['DATE','IPG2211A2N']
    
    # Forecast horizons to compare.
    forecast_horizons = [3, 5, 7, 9]
    series_name = "IPG2211A2N"
    time_col = "DATE"
    ar_order = 1         # AR(1)
    ma_window = 3        # Moving average window size
    add_seasonal = True  # Include seasonal (month) dummies
    
    # Containers for results.
    results_decay = {}
    results_normal = {}
    results_arma = {}
    results_cycle_ar = {}
    
    print("=== Decay-Weighted AR Forecasts (with learned decay factor) ===")
    for K in forecast_horizons:
        fc_avg, mape_decay, learned_dp = decay_weighted_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            decay_function="exponential",
            p=ar_order,
            ma_window=ma_window,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=True
        )
        results_decay[K] = (fc_avg, mape_decay, learned_dp)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg:.4f}, MAPE = {mape_decay:.2f}%, learned decay_param = {learned_dp:.4f}\n")
    
    print("=== Normal AR Forecasts (with additional features) ===")
    for K in forecast_horizons:
        fc_avg_norm, mape_norm = normal_ar_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            p=ar_order,
            ma_window=ma_window,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_normal[K] = (fc_avg_norm, mape_norm)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_norm:.4f}, MAPE = {mape_norm:.2f}%\n")
    
    print("=== ARMA(1,1, cycle) Forecasts ===")
    for K in forecast_horizons:
        fc_avg_arma, mape_arma = arma_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_arma[K] = (fc_avg_arma, mape_arma)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_arma:.4f}, MAPE = {mape_arma:.2f}%\n")
    
    print("=== AR with Cycle Forecasts (no MA) ===")
    for K in forecast_horizons:
        fc_avg_cycle, mape_cycle = cycle_ar_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            p=ar_order,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_cycle_ar[K] = (fc_avg_cycle, mape_cycle)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_cycle:.4f}, MAPE = {mape_cycle:.2f}%\n")
    
    # Final comparison.
    print("=== Final Forecast Comparison ===")
    for K in forecast_horizons:
        fc_decay, mape_decay, dp = results_decay[K]
        fc_normal, mape_normal = results_normal[K]
        fc_arma, mape_arma = results_arma[K]
        fc_cycle, mape_cycle = results_cycle_ar[K]
        print(f"For {K} days forecast:")
        print(f"  Decay-Weighted AR: Forecast Avg = {fc_decay:.4f}, MAPE = {mape_decay:.2f}% (decay_param = {dp:.2f})")
        print(f"  Normal AR      : Forecast Avg = {fc_normal:.4f}, MAPE = {mape_normal:.2f}%")
        print(f"  ARMA(1,1, cycle): Forecast Avg = {fc_arma:.4f}, MAPE = {mape_arma:.2f}%")
        print(f"  AR with Cycle (no MA): Forecast Avg = {fc_cycle:.4f}, MAPE = {mape_cycle:.2f}%\n")

