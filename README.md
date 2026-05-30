# MASI Hybrid Forecasting

Hybrid forecasting system for the MASI (Casablanca Stock Exchange) index :
**HMM** for regime detection + **CNN-LSTM** for next-day log-return prediction,
under a strict walk-forward anti-leakage methodology.

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

## Status

**v1.0.0 — pipeline migrated** : scripts 00→09, `src/masi_hybrid_forecasting/pipeline/`
CLI (5 subcommands), notebooks 01→09, reports + executive summary all in place.

Verdict from étape 9 (robustness) : **CNN-LSTM `base12` + HMM-gate = production
stack**, DSR ≈ 0.997 under the historical protocol, robust to 5/10/20 bps costs,
stable P1 ≈ P2.

Current artifacts live under `outputs/`, with figures in `reports/figures/`.
See [`docs/INDEX.md`](docs/INDEX.md) for the current step-by-step index and
[`docs/migration_plan.md`](docs/migration_plan.md) for the historical migration notes.
