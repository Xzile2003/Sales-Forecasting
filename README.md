# Sales Forecasting

Predict future retail/e-commerce sales from historical time-series data using **ARIMA (SARIMAX)**, **Prophet**, and **LSTM**, then plot forecast vs. actual sales.

## What's included
- **Trend**: captured by differencing (ARIMA), piecewise trend (Prophet), and learned patterns (LSTM)
- **Seasonality**: weekly (all models) and yearly (Prophet)
- **Promotions**: a `promo` (0/1) column used as an external regressor in all three models
- **Evaluation**: MAE, RMSE, MAPE on the last 60 days + a forecast-vs-actual plot (`forecast.png`)

## Quick start
```bash
pip install -r requirements.txt
python sales_forecast.py                          # ARIMA + Prophet on synthetic data
python sales_forecast.py --models arima prophet lstm
```
Synthetic daily data is generated automatically on first run (`data/sales.csv`).

## Use your own data
Provide a CSV with these columns:

| date       | sales | promo (optional) |
|------------|-------|------------------|
| 2024-01-01 | 245.3 | 0                |

```bash
python sales_forecast.py --data my_sales.csv --test-days 30
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | synthetic | Path to your CSV |
| `--models` | `arima prophet` | Any of `arima`, `prophet`, `lstm` |
| `--test-days` | `60` | Days held out for evaluation |

## Notes
- ARIMA is configured with weekly seasonality only; yearly seasonality is handled by Prophet.
- LSTM needs TensorFlow (large install) and is optional.
- Future promotion dates must be known in advance to forecast with them.
