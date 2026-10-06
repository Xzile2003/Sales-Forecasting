"""Sales Forecasting: ARIMA (SARIMAX), Prophet and LSTM on daily retail sales.

Usage:
    python sales_forecast.py                      # synthetic data, ARIMA + Prophet
    python sales_forecast.py --models arima prophet lstm
    python sales_forecast.py --data my_sales.csv  # columns: date, sales, promo
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_synthetic_data(path="data/sales.csv", seed=42):
    """Daily sales with trend + weekly/yearly seasonality + promotions + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    t = np.arange(len(dates))
    trend = 200 + 0.15 * t
    yearly = 30 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    weekly = np.array([0, -5, -5, 0, 10, 35, 25])[dates.dayofweek]
    promo = (rng.random(len(dates)) < 0.08).astype(int)
    sales = trend + yearly + weekly + 60 * promo + rng.normal(0, 10, len(dates))
    df = pd.DataFrame({"date": dates, "sales": sales.round(2), "promo": promo})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path):
    if path is None:
        path = "data/sales.csv"
        if not os.path.exists(path):
            make_synthetic_data(path)
    df = pd.read_csv(path, parse_dates=["date"])
    if "promo" not in df.columns:
        df["promo"] = 0
    return df.sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Models (each returns an array of predictions for the test period)
# ----------------------------------------------------------------------------
def run_arima(train, test):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train.set_index("date")["sales"],
        exog=train.set_index("date")[["promo"]],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),  # weekly seasonality
    ).fit(disp=False)
    pred = model.forecast(len(test), exog=test.set_index("date")[["promo"]])
    return pred.values


def run_prophet(train, test):
    from prophet import Prophet

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    m.add_regressor("promo")
    m.fit(train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", "promo"]])
    future = test.rename(columns={"date": "ds"})[["ds", "promo"]]
    return m.predict(future)["yhat"].values


def run_lstm(train, test, window=28, epochs=30):
    from tensorflow import keras

    lo, hi = train["sales"].min(), train["sales"].max()
    n = len(train)
    s = np.concatenate([(train["sales"].values - lo) / (hi - lo), np.zeros(len(test))])
    p = np.concatenate([train["promo"].values, test["promo"].values]).astype(float)

    def window_at(t):
        # each step = [previous day's sales, today's promo]
        idx = np.arange(t - window + 1, t + 1)
        return np.stack([s[idx - 1], p[idx]], axis=1)

    X = np.array([window_at(i) for i in range(window, n)])
    y = s[window:n]

    model = keras.Sequential(
        [keras.layers.Input((window, 2)), keras.layers.LSTM(32), keras.layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    for t in range(n, n + len(test)):  # recursive day-by-day forecast
        s[t] = model.predict(window_at(t)[None], verbose=0)[0, 0]
    return s[n:] * (hi - lo) + lo


MODELS = {"arima": run_arima, "prophet": run_prophet, "lstm": run_lstm}


# ----------------------------------------------------------------------------
# Evaluation + plot
# ----------------------------------------------------------------------------
def metrics(actual, pred):
    err = actual - pred
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE %": np.mean(np.abs(err / actual)) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with columns: date, sales, promo (optional)")
    ap.add_argument("--models", nargs="+", default=["arima", "prophet"], choices=MODELS)
    ap.add_argument("--test-days", type=int, default=60)
    args = ap.parse_args()

    df = load_data(args.data)
    train, test = df.iloc[: -args.test_days], df.iloc[-args.test_days :]

    plt.figure(figsize=(12, 5))
    plt.plot(train["date"].tail(120), train["sales"].tail(120), color="gray", label="Train")
    plt.plot(test["date"], test["sales"], color="black", linewidth=2, label="Actual")

    results = {}
    for name in args.models:
        print(f"Running {name} ...")
        pred = MODELS[name](train, test)
        results[name] = metrics(test["sales"].values, pred)
        plt.plot(test["date"], pred, linestyle="--", label=f"{name.upper()} forecast")

    promo_days = test[test["promo"] == 1]["date"]
    for d in promo_days:
        plt.axvline(d, color="orange", alpha=0.25)

    plt.title("Forecast vs Actual Sales (orange lines = promotion days)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png", dpi=150)
    print("\n", pd.DataFrame(results).T.round(2))
    print("\nSaved plot to forecast.png")
    plt.show()


if __name__ == "__main__":
    main()
