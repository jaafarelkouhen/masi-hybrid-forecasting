# `scripts/` — reproducible pipeline steps

Each script is one **reproducible research step**, numbered in execution order.
Run them sequentially: every step reads the artifacts written by the previous one
into [`../outputs/`](../outputs/).

```bash
python scripts/00_data_audit.py
python scripts/01_preprocessing.py
...
python scripts/09_robustness.py
```

| Script | Produces |
|---|---|
| `00_data_audit.py` | `data/interim/masi_merged.csv`, `outputs/etape0/` |
| `01_preprocessing.py` | temporal splits → `outputs/etape1/splits/` |
| `02_baselines.py` | baseline metrics + predictions → `outputs/etape2/` |
| `03_feature_engineering.py` | engineered features → `outputs/etape3/features/` |
| `04_hmm_regimes.py` | HMM regimes → `outputs/etape4/regimes/` |
| `05_cnn_lstm.py` | CNN-LSTM TEST predictions → `outputs/etape5/` |
| `06_backtesting.py` | backtest metrics → `outputs/etape6/` |
| `07_risk_layer.py` | risk metrics → `outputs/etape7/` |
| `08_strategies.py` | strategy comparison → `outputs/etape8/` |
| `09_robustness.py` | robustness checks → `outputs/etape9/` |

**`scripts/` vs `src/`:** scripts are the *runnable steps*; the clean, importable
logic lives in [`../src/`](../src/) and is exposed through the
`python -m masi_hybrid_forecasting.pipeline` CLI, which is the recommended way to
run the production stack once you understand the steps.

Each step also writes a human-readable `report.md` in its `outputs/etapeN/`
folder. See [`../docs/INDEX.md`](../docs/INDEX.md) for the full status table.
