# `notebooks/` — step-by-step exploration

These notebooks mirror the `scripts/` pipeline but are meant for **reading and
exploration** (plots, intermediate inspection, narrative). Run them **in order** —
each step depends on the artifacts of the previous one.

| Notebook | Step | Purpose |
|---|---|---|
| `01_preprocessing.ipynb` | Preprocessing | Cleaning, target creation, chronological splits. |
| `02_baselines.ipynb` | Baselines | Naive, historical mean, ARIMA, Random Forest references. |
| `03_feature_engineering.ipynb` | Features | Leakage-free momentum / volatility / technical / macro features. |
| `04_hmm_regimes.ipynb` | Regimes | 3-state Gaussian HMM with a causal forward filter. |
| `05_cnn_lstm.ipynb` | Predictor | Compact CNN-LSTM `base12` next-day return model. |
| `06_backtesting.ipynb` | Backtest | Cost-aware strategy metrics and Deflated Sharpe Ratio. |
| `07_risk_layer.ipynb` | Risk | VaR, ES, GARCH volatility, risk regime. |
| `08_strategies.ipynb` | Strategies | HMM-gate, risk-gate, VaR-budget sizing. |
| `09_robustness.ipynb` | Robustness | Subperiods, transaction costs, HMM thresholds, JKM tests. |
| `10_forecast_may_june.ipynb` | Live forecast | Out-of-sample forward forecast demo. |

> The reproducible, non-interactive equivalents are in
> [`../scripts/`](../scripts/). Use those for clean re-runs; use these to *look*
> at what each step does.
