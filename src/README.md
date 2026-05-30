# `src/` — the importable package

`src/masi_hybrid_forecasting/` is the installable Python package (`pip install -e .`).
It holds the **clean, reusable code** behind the research, as opposed to `scripts/`
which holds the step-by-step reproducible runs.

## Layout

```
masi_hybrid_forecasting/
├── data/          # loaders, temporal splits, leak-safe transforms
├── features/      # feature engineering (momentum, volatility, technical, macro)
├── regimes/       # HMM regime detection
├── models/        # CNN-LSTM architecture
├── backtesting/   # walk-forward evaluation utilities
├── utils/         # shared helpers
└── pipeline/      # the production CLI (entry point of the whole engine)
```

Most modelling logic is consumed through **`pipeline/`**, which orchestrates the
other submodules into five commands.

## The CLI

```bash
python -m masi_hybrid_forecasting.pipeline <command>
```

| Command | What it does |
|---|---|
| `predict` | Produce / show the CNN-LSTM next-day TEST predictions. |
| `risk` | Compute the risk layer (VaR, ES, GARCH vol, risk regime). |
| `backtest` | Apply a strategy and report metrics (Sharpe, Sortino, DSR…). |
| `export` | Write the canonical CSV consumed by the dashboard. |
| `train` | Re-train the CNN-LSTM walk-forward (~10–15 min). |

Full reference: [`masi_hybrid_forecasting/pipeline/README.md`](masi_hybrid_forecasting/pipeline/README.md).

The production strategy is **`hmm_gate`** (CNN-LSTM `base12` + HMM gate). Other
strategies and the cost grid are defined in `pipeline/config.py`.
