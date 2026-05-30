# ÉTAPE 3 — Feature Engineering & Leakage Audit
## MASI Hybrid Forecasting System
**Generated:** 2026-05-21
**Input:** `outputs/etape1/splits/masi_clean_full.csv` + canonical split dates
**Script:** `scripts/03_feature_engineering.py`
**Notebook:** `notebooks/03_feature_engineering.ipynb`

> Per `prompt.md` Étape 2 decisions: build derivative features (lags, rolling
> stats, GARCH, RSI, MACD, Bollinger) with strict `shift(>=1)` discipline (L3),
> then retrain the Random Forest as a sanity-check before adding HMM/CNN-LSTM.

---

## 1. Methodology

### 1.1 Compute-then-split (D1)

All causal features are computed on the **full clean series** (4,784 rows,
2007–2026), *then* re-split using Étape 1's exact TRAIN/VAL/TEST date indices.

This is the **correct leakage-free order** for causal features: each row uses
only its own past, so computing on the full series and slicing afterwards is
identical to computing per-split — but without destroying the VAL/TEST warm-up
rows. Split-*then*-compute would force each of VAL and TEST to discard its first
~26 rows (or, worse, borrow them from the previous split → leakage).

### 1.2 Single contemporaneity rule (D2)

| Feature class | Timing | Justification |
|---------------|--------|---------------|
| `log_return` | **contemporaneous** (`r_t`) | MASI's own return is fully settled at the MASI close — known at decision time `t` |
| Lagged returns `ret_lag{1,2,3,5}` | strictly past | explicit `shift(k)`, `k ≥ 1` |
| Rolling stats (mean, vol, RSI, MACD, Bollinger) | strictly past | `rolling(w).shift(1)` — window ends at `t-1`, no centered windows (L3) |
| `garch_vol` | strictly past | recursion `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` uses returns up to `t-1` |
| Cross-asset returns (Brent, Gold, EUR/MAD, 4 stocks) | **lagged 1 day** | foreign / commodity markets may not be settled at MASI's 15:30 Casablanca close → a contemporaneous cross-asset return is a soft look-ahead (L5 spirit) |
| Macro levels (`gpr_index`, `bam_policy_rate`) | lagged 1 day | published with reporting lag — conservative |

**Result:** exactly **one** feature (`log_return`) is contemporaneous; all 23
others use strictly past data. This single, auditable rule is what the empirical
test in §3 verifies.

### 1.3 Volatility proxies — Moroccan constraint C1 (no VIX) (D3)

| Proxy | Feature | Spec |
|-------|---------|------|
| P1 | `roll_vol_5`, `roll_vol_21` | rolling std of log-returns, `shift(1)` |
| P2 | `garch_vol` | GARCH(1,1) conditional volatility |
| P4 | `downside_semidev_21` | Sortino-style downside semi-deviation, `shift(1)` |

**GARCH(1,1) discipline (L8-partial, by design):** parameters are estimated on
**TRAIN log-returns only**; the conditional-variance recursion is then run
forward over the full series with **frozen** parameters (the L1/L2 discipline
applied to GARCH). Full per-walk-forward-window refit is not implemented in the
migrated pipeline; L8 is therefore **partial by design** and listed as future
work.

GARCH(1,1) TRAIN fit (returns scaled ×100 for numerical stability):

| Param | Value |
|-------|-------|
| μ (mean) | 0.0005 |
| ω (omega) | 0.05409 |
| α (alpha) | 0.2370 |
| β (beta) | 0.6643 |
| **α + β (persistence)** | **0.9012** |
| Log-likelihood | per `garch_params_train.json` |

α + β = 0.90 < 1 → covariance-stationary ✅, and the high persistence confirms
the **strong volatility clustering** found in the Étape 0 ARCH test. `garch_vol`
is therefore a genuine regime signal, not noise.

### 1.4 ATR / OHLC features — REJECTED (D4, RULE 6)

`masi_high` and `masi_low` are populated only from **2016-04 onward** (they come
from `masi_raw.csv`). Of 4,784 rows, **2,284 have NaN high/low** — i.e. nearly
all of the pre-2016 TRAIN period (TRAIN starts 2007).

An ATR feature would be NaN for ~70 % of TRAIN. Models requiring complete cases
(RF, CNN-LSTM) cannot use it; imputing 9 years of synthetic OHLC would itself be
a fabrication. Per `prompt.md` RULE 6, ATR is **explicitly rejected** (not
silently dropped). The volatility-proxy requirement is fully met by P1/P2/P4.

---

## 2. Feature Catalogue (24 features)

| Group | Features | Count |
|-------|----------|-------|
| **A — Momentum** | `log_return`, `ret_lag1/2/3/5`, `roll_mean_5`, `roll_mean_21` | 7 |
| **B — Volatility** | `roll_vol_5`, `roll_vol_21`, `garch_vol`, `downside_semidev_21` | 4 |
| **C — Technical** | `rsi_14`, `macd_hist`, `bb_pctb`, `bb_width` | 4 |
| **D — Cross-asset** (lagged) | `atw/iam/lhm/mng_ret_lag1`, `brent/gold/eurmad_ret_lag1` | 7 |
| **E — Macro** (lagged) | `gpr_lag1`, `bam_policy_rate_lag1` | 2 |

Target: `target_y_next = ln(P_{t+1}/P_t)` (inherited from Étape 1, L4-compliant).

---

## 3. Anti-Leakage Audit (L3) — Empirical Causality Test (D5)

`engineer_features()` is a **pure function**: row `t` of its output depends only
on raw rows `≤ t` and on the frozen TRAIN GARCH parameters. The test recomputes
the feature matrix on truncated series `full[:T]` and checks that **every row
before the cut is bit-identical** to the full-series computation. Any
future-peeking feature (centered window, leaky `fillna`, etc.) would change
earlier rows when the tail is removed.

| Cut index | Cut date | Rows compared | max \|diff\| | NaN pattern | Verdict |
|-----------|----------|---------------|-------------|-------------|---------|
| 1913 | 2014-10-01 | 1,883 | 0.00e+00 | identical | **PASS** |
| 2870 | 2018-08-02 | 2,840 | 0.00e+00 | identical | **PASS** |
| 3827 | 2022-06-14 | 3,797 | 0.00e+00 | identical | **PASS** |

**All cuts passed with exact zero difference** → the 24-feature matrix is
provably leakage-free. The script `assert`s this; a failure would halt Étape 3.

### Leakage-rule compliance summary

| Rule | Status | Evidence |
|------|--------|----------|
| L1 — Scaler stats fit on TRAIN only | ✅ ENFORCED | `scaler_stats_etape3_train_only.json`, computed on 3,297 TRAIN rows |
| L3 — `shift(≥1)`, no centered windows | ✅ ENFORCED + **TESTED** | §3 truncation test, max diff = 0.0 |
| L4 — Target built after features | ✅ INHERITED | `target_y_next` from Étape 1 |
| L6 — Temporal split + 8-day gaps | ✅ INHERITED | Étape 1 split dates reused verbatim |
| L8 — GARCH per window | 🟡 PARTIAL (by design) | params from TRAIN only; full per-window refit remains future work |

---

## 4. Feature Matrix & Splits

| Stage | Rows |
|-------|------|
| Full clean series | 4,784 |
| After warm-up drop (longest look-back = 30) | 4,733 (−51, **all at the 2007 series start**) |

| Split | Date range | N (Étape 1) | N (Étape 3) | Note |
|-------|------------|-------------|-------------|------|
| TRAIN | 2007-04-16 → 2020-07-09 | 3,348 | **3,297** | −51 warm-up rows |
| VAL | 2020-07-17 → 2022-06-20 | 478 | **478** | all rows kept ✅ |
| TEST | 2022-06-28 → 2026-04-17 | 948 | **948** | all rows kept ✅ |

The warm-up loss is **entirely confined to the TRAIN start** — VAL and TEST keep
every observation (asserted in code). This confirms the compute-then-split
design (§1.1) works as intended.

---

## 5. Random Forest Sanity-Check (D7)

Same RF spec as Étape 2 (100 trees, `min_samples_leaf=5`, `max_features=sqrt`,
`seed=42`), retrained on the **24 engineered features** instead of the 11 raw
inputs. Metrics use the **identical** definitions as Étape 2 (5 bps cost,
10,000-resample bootstrap DA CI).

| Metric | Étape 2 RF (11 raw) | Étape 3 RF (24 engineered) | Δ |
|--------|--------------------|-----------------------------|---|
| VAL — RMSE | 0.006326 | 0.006323 | ≈0 |
| VAL — DA | 0.5052 | **0.5178** | +0.0126 |
| VAL — Sharpe | +0.216 | **+0.494** | +0.278 |
| **TEST — DA** | 0.5327 `[0.500, 0.564]` | **0.5422 `[0.5105, 0.5738]`** | **+0.0095** |
| TEST — Sharpe | +0.714 | **+0.970** | +0.256 |
| TEST — MDD | −0.238 | **−0.163** | +0.075 (better) |
| TEST — trades | 210 | 348 | — |

### Interpretation

1. **Feature engineering helps — modestly but consistently.** TEST DA rises
   +0.95 pp and TEST Sharpe rises from 0.71 to 0.97. The improvement appears on
   both VAL and TEST, in the same direction → not a single-window artefact.

2. **The TEST DA confidence interval is now strictly above 0.50.** Étape 2 RF
   had a CI lower bound of *exactly* 0.5000 (marginal). The engineered RF gives
   `[0.5105, 0.5738]` — the lower bound clears 0.50, so directional skill on
   out-of-sample data is now **statistically significant at 95 %** under this
   protocol.

3. **Honest caveat.** This remains a *small* edge (DA ≈ 54 %) on a single TEST
   window. It does not yet meet the Étape 5 CNN-LSTM gate (DA ≥ 0.55,
   Sharpe ≥ 1.30). It establishes that the **feature set is sound** and that
   added model complexity (HMM regimes, CNN-LSTM) now has a fair chance to add
   value on top of informative inputs.

### RF feature importance — top 10 of 24

| Rank | Feature | Importance | Group |
|------|---------|-----------|-------|
| 1 | `log_return` | 0.1037 | A momentum |
| 2 | `roll_mean_5` | 0.0585 | A momentum |
| 3 | `macd_hist` | 0.0511 | C technical |
| 4 | `roll_mean_21` | 0.0500 | A momentum |
| 5 | `garch_vol` | 0.0488 | B volatility |
| 6 | `downside_semidev_21` | 0.0456 | B volatility |
| 7 | `ret_lag1` | 0.0450 | A momentum |
| 8 | `atw_ret_lag1` | 0.0430 | D cross-asset |
| 9 | `rsi_14` | 0.0419 | C technical |
| 10 | `roll_vol_5` | 0.0401 | B volatility |

`log_return` remains rank #1 (autocorrelation signal, as in Étape 2). The new
derivatives — momentum (`roll_mean_5/21`), `macd_hist`, and the volatility
proxies (`garch_vol`, `downside_semidev_21`) — populate ranks 2–6, confirming
they carry independent signal. `bam_policy_rate_lag1` is least important
(0.0117) — the policy rate is too slow-moving for daily prediction.

---

## 6. Recommended CNN-LSTM Core-15 Feature Set

Constraint C3 requires `F ≤ 15` inputs for the CNN-LSTM (only ~2,000–2,500
effective observations). The top-15 features by RF importance:

```
 1. log_return            9. rsi_14
 2. roll_mean_5          10. roll_vol_5
 3. macd_hist            11. roll_vol_21
 4. roll_mean_21         12. lhm_ret_lag1
 5. garch_vol            13. ret_lag3
 6. downside_semidev_21  14. bb_pctb
 7. ret_lag1             15. ret_lag2
 8. atw_ret_lag1
```

Balanced across groups: 7 momentum, 4 volatility, 2 technical, 2 cross-asset.
Note `roll_vol_5`/`roll_vol_21` are correlated (see heatmap) — Étape 5 may prune
one if a smaller `F` is needed. This list is a **recommendation**; Étape 5 will
test it against window sizes `L ∈ {10, 15, 20}`.

---

## 7. Output Artifacts

| Artifact | Path |
|----------|------|
| Engineered features — TRAIN | `outputs/etape3/features/masi_features_train.csv` |
| Engineered features — VAL | `outputs/etape3/features/masi_features_val.csv` |
| Engineered features — TEST | `outputs/etape3/features/masi_features_test.csv` |
| Scaler stats (TRAIN-only, L1) | `outputs/etape3/features/scaler_stats_etape3_train_only.json` |
| Frozen GARCH parameters | `outputs/etape3/features/garch_params_train.json` |
| RF sanity-check metrics | `outputs/etape3/features/rf_sanitycheck_metrics.json` |
| Feature metadata / catalogue | `outputs/etape3/features/feature_metadata.json` |
| Plot — correlation heatmap | `reports/figures/etape3/etape3_corr_heatmap.png` |
| Plot — RF feature importance | `reports/figures/etape3/etape3_rf_importance.png` |
| Plot — volatility proxies | `reports/figures/etape3/etape3_volatility_proxies.png` |
| Plot — technical/momentum overview | `reports/figures/etape3/etape3_feature_overview.png` |

---

## 8. Decisions for Étape 4 (HMM Regime Detection)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Train HMM on **TRAIN engineered features only** (L2); forward-predict regimes on VAL/TEST | No regime label may use future data |
| 2 | Candidate HMM inputs: `log_return`, `garch_vol`, `roll_vol_21`, `downside_semidev_21` | Return + volatility define Bull/Neutral/Bear regimes (Sivakumar 2024) |
| 3 | Use **3 states** (Bull / Neutral / Bear) | Matches project objective and the 2008 + COVID Bears in TRAIN |
| 4 | Keep `garch_vol` as TRAIN-frozen in this version; test full per-window refit as future work | Documents L8 as partial instead of overstating implementation |
| 5 | One-hot encode the predicted regime → append as CNN-LSTM feature (Regime-as-Feature) | Core project innovation |

---

*End of Étape 3 Report — generated by `scripts/03_feature_engineering.py`*
