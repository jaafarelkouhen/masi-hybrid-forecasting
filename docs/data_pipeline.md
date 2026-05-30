# Data Pipeline

End-to-end flow from raw sources to backtest-ready strategy outputs.

## 1. Sources

| File | Span | Role |
|---|---:|---|
| `data/raw/masi_raw.csv` | 2007–2026 | MASI daily close / OHLC source |
| `data/raw/master_dataset.csv` and `.xlsx` | 2007–2026 | Cross-asset and macro panel |

The raw files are ignored by git. A clean clone needs either a data handoff or a
regeneration step before integration tests can run without skips.

## 2. Current Flow

```text
data/raw/
  └─ scripts/00_data_audit.py
       -> data/interim/masi_merged.csv
       -> outputs/etape0/

data/interim/masi_merged.csv
  └─ scripts/01_preprocessing.py
       -> outputs/etape1/splits/masi_{clean_full,train,val,test}.csv

outputs/etape1/splits/
  ├─ scripts/02_baselines.py
  │    -> outputs/etape2/
  └─ scripts/03_feature_engineering.py
       -> outputs/etape3/features/

outputs/etape3/features/
  └─ scripts/04_hmm_regimes.py
       -> outputs/etape4/regimes/

outputs/etape4/regimes/
  └─ scripts/05_cnn_lstm.py
       -> outputs/etape5/predictions_test.csv

outputs/etape5/ + outputs/etape4/
  ├─ scripts/06_backtesting.py
  │    -> outputs/etape6/
  ├─ scripts/07_risk_layer.py
  │    -> outputs/etape7/
  ├─ scripts/08_strategies.py
  │    -> outputs/etape8/
  └─ scripts/09_robustness.py
       -> outputs/etape9/
```

The production-facing CLI in `src/masi_hybrid_forecasting/pipeline/` orchestrates
the generated artifacts:

```powershell
python -m masi_hybrid_forecasting.pipeline predict
python -m masi_hybrid_forecasting.pipeline risk
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate
python -m masi_hybrid_forecasting.pipeline export --strategy hmm_gate
```

## 3. Canonical Splits

| Split | Period | Rows | Role |
|---|---:|---:|---|
| TRAIN | 2007–2020 | ~3.3k | Fit scalers, feature stats, GARCH, HMM, model weights |
| VAL | 2020–2022 | 478 | Model selection / early stopping |
| TEST | 2022–2026 | 948 | Out-of-sample evaluation |

The pipeline keeps positive calendar gaps between TRAIN→VAL and VAL→TEST.

## 4. Leakage Controls

- `StandardScaler` and feature statistics are fit on TRAIN only.
- Rolling features use `shift(1)` or equivalent past-only windows.
- Cross-asset and macro features are lagged conservatively.
- HMM regimes exported as features use a causal forward filter, not full-sequence
  Viterbi smoothing.
- GARCH and HMM refit is currently **partial**: parameters are fit on TRAIN and
  applied forward causally. Full per-fold GARCH+HMM refit is future work.

Full details: [`anti_leakage.md`](anti_leakage.md).
