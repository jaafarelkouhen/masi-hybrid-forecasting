# MASI Hybrid Forecasting

Hybrid forecasting system for the **MASI** (Casablanca Stock Exchange) index :
**HMM** for regime detection + **CNN-LSTM** for next-day log-return prediction,
under a strict walk-forward anti-leakage methodology.

> **Headline result** — CNN-LSTM `base12` + HMM-gate, TEST 2022-06-28 → 2026-04-17
> (948 days) : Sharpe ≈ **+1.71** (historical convention) / **+1.55** (turnover-aware),
> Max Drawdown ≈ **−6 %**, DSR ≈ **0.997**, robust to 5/10/20 bps costs.

![Equity curves — production strategy vs baselines](reports/figures/etape6/etape6_equity_curves.png)

---

## Table of contents

- [Structure](#structure)
- [Install](#install)
- [Run the pipeline](#run-the-pipeline)
- [Methodology](#methodology)
- [Visual tour of the pipeline (screenshots)](#visual-tour-of-the-pipeline-screenshots)
- [Resources & papers used](#resources--papers-used)
- [Documentation site (Read the Docs)](#documentation-site-read-the-docs)
- [Status](#status)

---

## Structure

```
masi-hybrid-forecasting/
├── data/                            # raw, external, interim, processed (features, regimes)
├── notebooks/                       # 00_audit → 09_robustness (pipeline order)
├── src/masi_hybrid_forecasting/     # importable package
│   ├── data/                        # loaders, splits, leak-safe transforms
│   ├── features/                    # feature engineering
│   ├── regimes/                     # HMM
│   ├── models/                      # CNN-LSTM
│   ├── backtesting/                 # walk-forward evaluation
│   └── utils/
├── scripts/                         # CLI entry points (training, backtest, reports)
├── reports/                         # human-facing deliverables (figures, executive_summary)
├── outputs/                         # model artifacts (checkpoints, predictions) — gitignored
├── docs/                            # methodology, data_pipeline, references (PDFs + external repos)
└── tests/
    ├── unit/                        # pure unit tests (no I/O)
    └── integration/                 # touches disk/data
```

**`scripts/` vs `outputs/` vs `reports/`** :
- `scripts/` = reproducible research steps (`00_data_audit.py` → `09_robustness.py`).
- `outputs/` = artefacts produced by the pipeline (model checkpoints, raw predictions, intermediate JSON). **Gitignored.**
- `reports/` = curated deliverables for humans (figures for the report, executive summary, final results table).

## Install

```bash
pip install -e ".[notebooks,dev]"
```

- `notebooks` extra → `jupyter`, `ipykernel`
- `dev` extra → `pytest`

Core deps (numpy/pandas/scipy/scikit-learn, statsmodels/arch/hmmlearn, torch, matplotlib/seaborn) are installed by default.

## Run the pipeline

The production stack (CNN-LSTM `base12` + HMM-gate + optional risk layer) ships
as a 5-command CLI :

```bash
python -m masi_hybrid_forecasting.pipeline predict                            # show TEST predictions
python -m masi_hybrid_forecasting.pipeline risk                               # VaR / ES / risk regime
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate       # production strategy
python -m masi_hybrid_forecasting.pipeline export  --strategy hmm_gate        # canonical CSV for dashboard
python -m masi_hybrid_forecasting.pipeline train                              # re-train CNN-LSTM (~10-15 min)
```

Full subcommand reference : [`src/masi_hybrid_forecasting/pipeline/README.md`](src/masi_hybrid_forecasting/pipeline/README.md).

For step-by-step exploration, run the `scripts/00_data_audit.py` →
`scripts/09_robustness.py` series in order, or open the matching `notebooks/`.

## Methodology

- **Splits** : TRAIN 2007–2020 / VAL 2020–2022 / TEST 2022–2026, 70/10/20 with 8-day gaps.
- **Anti-leakage** : every transform (scaler, HMM fit, feature stats) is fit on the training window only and applied causally. See [`docs/anti_leakage.md`](docs/anti_leakage.md).
- **Architecture** : see [`docs/methodology.md`](docs/methodology.md).

---

## Visual tour of the pipeline (screenshots)

Every diagnostic figure produced by the pipeline lives under
[`reports/figures/`](reports/figures/) — **44 plots** total, grouped by étape.
The full set is shown below.

### Étape 0 — Data audit (4 figures)

Raw MASI series, log-returns, missingness map, plus stationarity / ARCH /
QQ-plot diagnostics that justify the choice of GARCH(1,1) as the volatility
proxy (no high-frequency data available on MASI).

| MASI overview | Returns ACF |
|---|---|
| ![Overview](reports/figures/etape0/audit_plot_1_overview.png) | ![ACF](reports/figures/etape0/audit_plot_2_acf.png) |
| **Rolling stats** | **QQ-plot** |
| ![Rolling stats](reports/figures/etape0/audit_plot_3_rolling_stats.png) | ![QQ-plot](reports/figures/etape0/audit_plot_4_qqplot.png) |

### Étape 1 — Preprocessing & splits (4 figures)

Strict temporal split **TRAIN 2007–2020 / VAL 2020–2022 / TEST 2022–2026** with
8-day gaps between segments — no shuffling, no leakage across boundaries.

| Split overview | Target distribution |
|---|---|
| ![Splits](reports/figures/etape1/etape1_split_overview.png) | ![Target](reports/figures/etape1/etape1_target_distribution.png) |
| **Factor overview** | **Scaler diagnostic** |
| ![Factors](reports/figures/etape1/etape1_factor_overview.png) | ![Scaler](reports/figures/etape1/etape1_scaler_diagnostic.png) |

### Étape 2 — Baselines (3 figures)

Four mandatory baselines before deep learning : Naive (RW), Historical mean,
ARIMA, Random Forest. ARIMA wins on RMSE, sets the floor any CNN-LSTM has to
beat.

| Metrics summary | Predictions VAL |
|---|---|
| ![Metrics](reports/figures/etape2/etape2_metrics_summary.png) | ![VAL](reports/figures/etape2/etape2_predictions_val.png) |

![Predictions TEST](reports/figures/etape2/etape2_predictions_test.png)

### Étape 3 — Feature engineering (4 figures)

24 leakage-free features (momentum, volatility proxies, technical indicators)
— every rolling stat is built with `.shift(1).rolling(...)` so the value at
day `t` only uses information from `≤ t-1`.

| Feature overview | Correlation heatmap |
|---|---|
| ![Features](reports/figures/etape3/etape3_feature_overview.png) | ![Corr](reports/figures/etape3/etape3_corr_heatmap.png) |
| **Random Forest importance** | **Volatility proxies** |
| ![RF importance](reports/figures/etape3/etape3_rf_importance.png) | ![Vol proxies](reports/figures/etape3/etape3_volatility_proxies.png) |

### Étape 4 — HMM regimes (5 figures)

3-state Gaussian HMM (`Bear / Neutral / Bull`) fit on TRAIN only and applied
causally on VAL+TEST via forward filter. Coverage on TEST :
**Bull 409 / Neutral 371 / Bear 168**.

| Regime timeline | Transition matrix |
|---|---|
| ![Timeline](reports/figures/etape4/etape4_regime_timeline.png) | ![Transitions](reports/figures/etape4/etape4_transition_matrix.png) |
| **Regime characteristics** | **Model selection** |
| ![Characteristics](reports/figures/etape4/etape4_regime_characteristics.png) | ![Model selection](reports/figures/etape4/etape4_model_selection.png) |

![Specification comparison](reports/figures/etape4/etape4_spec_comparison.png)

### Étape 5 — CNN-LSTM `base12` (5 figures)

Compact architecture (~5k params, well under the 10k overfitting threshold for
MASI's small data regime). Directional accuracy **0.556** on TEST, stable
across folds.

| Cumulative TEST P&L | Walk-forward fold stability |
|---|---|
| ![Cumulative](reports/figures/etape5/etape5_cumulative.png) | ![Folds](reports/figures/etape5/etape5_fold_stability.png) |
| **Baseline comparison** | **Threshold rule** |
| ![Baselines](reports/figures/etape5/etape5_baseline_comparison.png) | ![Threshold](reports/figures/etape5/etape5_threshold_rule.png) |

![L scan (input window)](reports/figures/etape5/etape5_lscan.png)

### Étape 6 — Backtest & DSR (5 figures)

Cost-aware backtest (5 / 10 / 20 bps), regime-conditional Sharpe heatmap,
Deflated Sharpe Ratio. The CNN-LSTM survives 20 bps with Sharpe ≈ +0.42 — the
only baseline that does.

| Equity curves | Drawdowns |
|---|---|
| ![Equity](reports/figures/etape6/etape6_equity_curves.png) | ![Drawdowns](reports/figures/etape6/etape6_drawdowns.png) |
| **Cost sensitivity** | **Regime heatmap** |
| ![Cost](reports/figures/etape6/etape6_cost_sensitivity.png) | ![Regime heatmap](reports/figures/etape6/etape6_regime_heatmap.png) |

![Final summary](reports/figures/etape6/etape6_final_summary.png)

### Étape 7 — Risk layer (4 figures)

Causal VaR / ES (parametric + historical) and a 3-state risk regime built from
TRAIN+VAL quantiles. Kupiec POF passes (5.80 %, p=0.27), Christoffersen
independence rejected — honest negative finding documented in
`outputs/etape7/report.md`.

| Vol cone (GARCH) | VaR breaches (Kupiec) |
|---|---|
| ![Vol cone](reports/figures/etape7/etape7_vol_cone.png) | ![VaR](reports/figures/etape7/etape7_var_breaches.png) |
| **MDD comparison** | **Return distribution** |
| ![MDD](reports/figures/etape7/etape7_mdd_comparison.png) | ![Return dist](reports/figures/etape7/etape7_return_distribution.png) |

### Étape 8 — Combined strategies (5 figures, HMM-gate wins)

Seven strategies compared (raw signal, HMM-gate, risk-gate, VaR-budget,
combinations). **HMM-gate is the winner** : Sharpe +1.71, MDD −6 %, DSR 0.997
under the historical protocol.

| DSR summary | Equity curves |
|---|---|
| ![DSR](reports/figures/etape8/etape8_dsr_summary.png) | ![Equity étape 8](reports/figures/etape8/etape8_equity_curves.png) |
| **Drawdowns** | **Regime heatmap** |
| ![Drawdowns](reports/figures/etape8/etape8_drawdowns.png) | ![Regime heatmap](reports/figures/etape8/etape8_regime_heatmap.png) |

![Sharpe vs MDD scatter](reports/figures/etape8/etape8_sharpe_mdd_scatter.png)

### Étape 9 — Robustness (5 figures, 5 axes)

Temporal stability (P1 ≈ P2 ≈ 1.7), cost robustness up to 20 bps, HMM
threshold insensitivity, and Jobson-Korkie-Memmel pairwise tests.

| Subperiod Sharpe (P1 vs P2) | Cost sensitivity |
|---|---|
| ![Subperiod](reports/figures/etape9/etape9_subperiod_sharpe.png) | ![Cost](reports/figures/etape9/etape9_cost_sensitivity.png) |
| **Dynamic HMM threshold** | **JKM heatmap** |
| ![HMM threshold](reports/figures/etape9/etape9_dynamic_hmm_threshold.png) | ![JKM](reports/figures/etape9/etape9_jkm_heatmap.png) |

![Robustness scorecard](reports/figures/etape9/etape9_robustness_scorecard.png)

---

## Resources & papers used

The methodology is grounded in 10 peer-reviewed papers. Full synthesis,
critique of leakage risks paper by paper, and the resulting decisions for the
MASI pipeline are documented in
[`docs/literature_review.md`](docs/literature_review.md).

### P1 — Anti-leakage & validation

- **Albelali & Ahmed (2025)** — *Hidden Leaks in LSTM Time Series Forecasting*.
  arXiv : [2512.12924](https://arxiv.org/abs/2512.12924). Motivates the strict
  "split first, build windows second" discipline used in Étape 1.
- **Deep, Deep & Lamptey (2025)** — *Walk-Forward Validation Framework*.
  arXiv : [2512.06932](https://arxiv.org/abs/2512.06932). Backbone of the
  walk-forward protocol and the "signal at t → execute at OPEN of t+1" rule.

### P2 — Moroccan market (MASI / MSI20)

- **Oukhouya & El Himdi (2024)** — *ML Forecasting MASI : XGBoost / SVR / LSTM*.
  In *Lecture Notes in Networks and Systems*, vol. 1037.
  [doi:10.1007/978-3-031-68628-3_6](https://doi.org/10.1007/978-3-031-68628-3_6).
- **Oukhouya & El Himdi (2023)** — *SVR / XGBoost / LSTM / MLP on MSI 20*.
  *Computer Sciences & Mathematics Forum* 7(1), 39.
  [doi:10.3390/cmsf2023007039](https://doi.org/10.3390/cmsf2023007039).
- **Touzani & Douzi (2021)** — *LSTM + GRU Trading Strategy for the Moroccan Market*.
  *Journal of Big Data* 8, 126.
  [doi:10.1186/s40537-021-00512-z](https://doi.org/10.1186/s40537-021-00512-z).

### P3 — HMM + Deep Learning architecture

- **Sivakumar (2025)** — *HMM-LSTM Fusion Model for Economic Forecasting*.
  arXiv : [2501.02002](https://arxiv.org/abs/2501.02002). Source of the
  "regime as feature" idea used to inject the HMM state into the CNN-LSTM input.
- **Monteiro (2025)** — *HMM + Neural Networks for Algorithmic Trading*.
  arXiv : [2407.19858](https://arxiv.org/abs/2407.19858). Source of the
  consensus-gating idea behind the HMM-gate strategy in Étape 8.

### P4 — Frontier & emerging markets

- **Korley & Giouvris (2021)** — *Regime-Switching in Frontier African Markets*.
  *Journal of Risk and Financial Management* 14(3), 122.
  [doi:10.3390/jrfm14030122](https://doi.org/10.3390/jrfm14030122).
  Justifies the 3-state asymmetric regime model for MASI.

### P5 — Volatility proxies (no VIX on MASI)

- **Hansen, Huang, Tong & Wang (2021)** — *Realized GARCH, CBOE VIX, and the
  Volatility Risk Premium*. arXiv : [2112.05302](https://arxiv.org/abs/2112.05302).
  Establishes why MASI must fall back to GARCH(1,1) on daily data (no
  intraday data available).

### P6 — Deep learning on financial time series

- **Fozap (2025)** — *Hybrid LSTM-CNN with Technical Indicators*.
  *Journal of Risk and Financial Management* 18(4), 201.
  [doi:10.3390/jrfm18040201](https://doi.org/10.3390/jrfm18040201).
  Reference architecture for the CNN→LSTM ordering used in Étape 5.

> The PDFs of these papers and the cloned reference repos live under
> `docs/references/`, which is **gitignored** (private working bibliography).
> Only the citations and the synthesis are public.

---

## Documentation site (Read the Docs)

A Sphinx documentation site is configured in [`docs/`](docs/). To build it
locally :

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

The same configuration ([`.readthedocs.yaml`](.readthedocs.yaml)) powers the
hosted version on Read the Docs — every push to `main` triggers a rebuild.

## Status

**v1.0.0 — pipeline migrated** : scripts 00→09, `src/masi_hybrid_forecasting/pipeline/`
CLI (5 subcommands), notebooks 01→09, reports + executive summary all in place.

Verdict from étape 9 (robustness) : **CNN-LSTM `base12` + HMM-gate = production
stack**, DSR ≈ 0.997 under the historical protocol, robust to 5/10/20 bps costs,
stable P1 ≈ P2.

Current artifacts live under `outputs/`, with figures in `reports/figures/`.
See [`docs/pipeline_index.md`](docs/pipeline_index.md) for the current step-by-step index.
