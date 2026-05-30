# Methodology

Hybrid forecasting system for the MASI (Casablanca Stock Exchange) index.
Reference rules and historical constraints live in [`project_spec.md`](project_spec.md).

## 1. Objective

Predict the **next-day MASI log-return** with a realistic frontier-market
protocol: limited data, no VIX, low liquidity, transaction costs, and strict
anti-leakage discipline.

## 2. Architecture

```text
raw data
  -> preprocessing and temporal splits
  -> leakage-free feature engineering
  -> HMM regime detection
  -> CNN-LSTM next-day return prediction
  -> strategy layer: raw signal, HMM-gate, risk-gate, VaR budget
  -> robustness: subperiods, transaction costs, HMM thresholds, JKM tests
  -> CLI export for dashboard/API
```

## 3. Components

| Layer | Technique | Current status |
|---|---|---|
| Data audit | Stationarity, ARCH, missingness, anomalies | `scripts/00_data_audit.py` |
| Preprocessing | Cleaning, target creation, temporal splits | `scripts/01_preprocessing.py` |
| Baselines | Naive, historical mean, ARIMA, Random Forest | `scripts/02_baselines.py` |
| Features | Momentum, volatility, technical, cross-asset, macro | `scripts/03_feature_engineering.py` |
| Regimes | 3-state Gaussian HMM with causal forward filter | `scripts/04_hmm_regimes.py` |
| Predictor | Compact CNN-LSTM `base12` | `scripts/05_cnn_lstm.py` |
| Backtest | Cost-aware strategy metrics and DSR | `scripts/06_backtesting.py` |
| Risk | VaR, ES, GARCH vol, risk regime | `scripts/07_risk_layer.py` |
| Strategies | HMM-gate, risk-gate, budget sizing | `scripts/08_strategies.py` |
| Robustness | Subperiods, costs, thresholds, JKM | `scripts/09_robustness.py` |
| CLI | `predict`, `risk`, `backtest`, `export`, `train` | `src/masi_hybrid_forecasting/pipeline/` |

## 4. Validation Discipline

- **L1–L8 anti-leakage rules** are documented in [`anti_leakage.md`](anti_leakage.md).
- Deep learning is evaluated only after baselines.
- Reported performance uses realistic transaction costs.
- Directional accuracy, Sharpe, Sortino, maximum drawdown, Calmar, DSR, and
  regime-conditional metrics are all reported.
- The final result is treated as **defensible under the current protocol**, not
  proof of live alpha. A later holdout or paper-trading period is still needed.

## 5. Current Verdict

The historical final recommendation is:

```text
CNN-LSTM base12 + HMM-gate
```

The CNN-LSTM provides a small directional edge. The HMM-gate improves the
risk-return profile by avoiding the Neutral regime, where the raw predictor is
weak. Under the original step-8 protocol it achieved Sharpe around `1.71`,
maximum drawdown around `-6%`, and DSR around `0.997`.

After tightening the transaction-cost convention so that a direct `-1 -> +1`
flip pays two one-way trades, the HMM-gate remains attractive but more
conservative: the same TEST predictions give Sharpe around `1.55`, maximum
drawdown around `-6.5%`, final equity around `1.71`, and DSR around `0.992`.
This is the safer number to quote unless all reports are regenerated.

## 6. Known Methodological Limits

- The TEST window is unique: 948 days from 2022-06-28 to 2026-04-17.
- HMM/GARCH are fit on TRAIN and applied causally; full per-fold refit is future
  work.
- DSR is sensitive to the chosen strategy panel and should be explained with the
  same care as in the step-8 and step-9 reports.
- The strategy should be validated on a later holdout or paper-trading period
  before any production claim.
