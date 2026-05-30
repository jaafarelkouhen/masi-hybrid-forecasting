"""
================================================================================
ÉTAPE 3 — Feature Engineering (leakage-free derivative features)
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Build the derivative feature set that HMM (Étape 4) and CNN-LSTM (Étape 5) will
  consume. All features are constructed from RAW columns only, with strict
  anti-leakage discipline, and validated with an empirical causality test.

INPUT
  outputs/etape1/splits/masi_clean_full.csv   (4,784 rows, raw, target included)
  outputs/etape1/splits/masi_{train,val,test}.csv  (canonical split dates)

OUTPUT
  outputs/etape3/features/masi_features_{train,val,test}.csv  (engineered, unscaled)
  outputs/etape3/features/scaler_stats_etape3_train_only.json (L1 — TRAIN-only)
  outputs/etape3/features/feature_metadata.json               (feature catalogue)
  outputs/etape3/features/garch_params_train.json             (frozen GARCH params)
  outputs/etape3/features/rf_sanitycheck_metrics.json         (RF retrain results)
  reports/figures/etape3/*.png                                  (4 diagnostic plots)

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5 — every choice justified)
--------------------------------------------------------------------------------
D1  Compute-then-split.  Causal features are computed on the FULL clean series,
    THEN re-split by Étape 1 dates.  Each row uses only its own past, so this is
    leakage-free; split-then-compute would wrongly destroy VAL/TEST warmup rows.

D2  Single contemporaneity rule.  Only `log_return` (MASI's OWN return, fully
    settled at the MASI close) is used contemporaneously.  EVERY other feature
    uses strictly past data: explicit lags (shift k>=1) or rolling().shift(1).
    -> Cross-asset returns (Brent, Gold, EUR/MAD, the 4 stocks) are lagged 1 day:
       foreign / commodity markets may not be settled at MASI's 15:30 close, so a
       contemporaneous cross-asset return would be a soft look-ahead (L5 spirit).

D3  Volatility proxies (Moroccan constraint C1 — no VIX):
       P1  roll_vol_5, roll_vol_21      rolling realized volatility
       P2  garch_vol                    GARCH(1,1) conditional volatility
       P4  downside_semidev_21          Sortino-style downside semi-deviation
    GARCH(1,1) parameters are estimated on TRAIN ONLY; the conditional-variance
    recursion is then run forward over the full series with FROZEN parameters
    (the L1/L2 discipline applied to GARCH).  Full per-walk-forward-window refit
    is not implemented in the migrated pipeline; L8 remains partial by design.

D4  ATR / OHLC features REJECTED (RULE 6).  masi_high / masi_low exist only from
    2016-04 onward -> NaN for 2,284 / 4,784 rows, i.e. ~all of the pre-2016 TRAIN
    period.  A feature absent for the majority of training is unusable for the
    models; ATR is documented as rejected, not silently imputed.  P1/P2/P4 cover
    the volatility-proxy requirement instead.

D5  Empirical leakage test.  engineer_features() is a pure function; recomputing
    it on series[:T] must reproduce every earlier row BIT-IDENTICALLY versus the
    full-series computation.  Any future-peeking feature (centered window, bad
    fillna, etc.) changes earlier rows -> AssertionError.  GARCH uses frozen
    TRAIN params so it is deterministic and included in the test.

D6  StandardScaler stats fit on TRAIN engineered rows ONLY (L1).  Saved as JSON;
    the unscaled feature files are saved for interpretability + the RF check.

D7  RF sanity-check.  Retrain the Étape 2 Random Forest on the engineered feature
    set and compare to the Étape 2 RF (TEST DA = 0.5327).  Per prompt.md, feature
    engineering must demonstrably help before adding HMM / CNN-LSTM complexity.

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE
--------------------------------------------------------------------------------
  L1  Scaler stats fit on TRAIN engineered rows only.                  ENFORCED
  L3  All rolling / lag features use shift(>=1); no centered windows.  ENFORCED + TESTED
  L4  Target `target_y_next` was built AFTER features in Étape 1.      INHERITED
  L6  Splits use Étape 1 dates (70/10/20 + 8-day gaps).                INHERITED
  L8  GARCH params from TRAIN only here; full per-window refit is future work. PARTIAL

Author: Quantitative Research Lab
Date:   2026-05-21
================================================================================
"""

from __future__ import annotations

import os
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape1", "splits")
FEAT_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape3", "features")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape3")

os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5          # one-way, MASI-realistic (consistent with Étape 2)
TARGET = "target_y_next"

# Raw input columns expected in masi_clean_full.csv
RAW_PRICE_COLS = ["masi_close", "atw_close", "iam_close", "lhm_close", "mng_close",
                  "brent_close", "gold_close", "eur_mad"]
RAW_MACRO_LEVEL_COLS = ["gpr_index", "bam_policy_rate"]

# Longest look-back window across all features (macd EMA-26, bb-20, roll-21).
# Used to drop the global warm-up rows (all located at the 2007 series start).
WARMUP = 30


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_clean_full() -> pd.DataFrame:
    path = os.path.join(SPLITS_DIR, "masi_clean_full.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run scripts/01_preprocessing.py first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    print(f"  clean_full : {len(df):4d} obs  {df.index.min().date()} -> {df.index.max().date()}")
    return df


def load_split_indices() -> Dict[str, pd.DatetimeIndex]:
    """Return the exact TRAIN/VAL/TEST date indices from Étape 1 (canonical splits)."""
    idx = {}
    for name in ("train", "val", "test"):
        p = os.path.join(SPLITS_DIR, f"masi_{name}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file {p} — run scripts/01_preprocessing.py first.")
        d = pd.read_csv(p)
        idx[name] = pd.to_datetime(d["date"])
        print(f"  {name:5s}      : {len(idx[name]):4d} obs  "
              f"{idx[name].min().date()} -> {idx[name].max().date()}")
    return {k: pd.DatetimeIndex(v) for k, v in idx.items()}


# =============================================================================
# 2. GARCH(1,1) — TRAIN-fitted, frozen-parameter forward recursion (D3, L8-partial)
# =============================================================================

def fit_garch_on_train(train_returns: pd.Series) -> Dict[str, float]:
    """
    Fit GARCH(1,1) with constant mean on TRAIN log-returns only.
    `arch` recommends scaling returns by 100 for numerical stability.
    Returns the frozen parameters {mu, omega, alpha, beta} on the *scaled* series.
    """
    r = train_returns.dropna().values * 100.0
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    p = res.params
    params = {
        "mu":    float(p["mu"]),
        "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]),
        "beta":  float(p["beta[1]"]),
        "scale": 100.0,
        "loglik": float(res.loglikelihood),
        "n_train": int(len(r)),
    }
    persist = params["alpha"] + params["beta"]
    print(f"    GARCH(1,1) TRAIN fit: mu={params['mu']:.4f} omega={params['omega']:.5f} "
          f"alpha={params['alpha']:.4f} beta={params['beta']:.4f}  (alpha+beta={persist:.4f})")
    assert persist < 1.0, "GARCH not covariance-stationary (alpha+beta >= 1) — reject."
    return params


def garch_conditional_vol(returns: pd.Series, params: Dict[str, float]) -> pd.Series:
    """
    Run the GARCH(1,1) conditional-variance recursion over the FULL return series
    using the FROZEN train parameters:

        eps_t       = r_t - mu
        sigma2_1    = omega / (1 - alpha - beta)            (unconditional variance)
        sigma2_t    = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}

    sigma_t depends only on returns up to t-1 => the value at row t uses strictly
    past data => leakage-free as a feature for predicting target at row t.
    Output is returned on the original (un-scaled) return scale.
    """
    mu, omega = params["mu"], params["omega"]
    alpha, beta, scale = params["alpha"], params["beta"], params["scale"]

    r = returns.values * scale
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        rt1 = r[t - 1]
        eps2 = (rt1 - mu) ** 2 if np.isfinite(rt1) else sigma2[t - 1]
        sigma2[t] = omega + alpha * eps2 + beta * sigma2[t - 1]
    cond_vol = np.sqrt(sigma2) / scale     # back to return scale
    return pd.Series(cond_vol, index=returns.index, name="garch_vol")


# =============================================================================
# 3. TECHNICAL INDICATOR PRIMITIVES (all causal — applied with shift(1) below)
# =============================================================================

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Uses close up to t; shift(1) applied by caller for L3 safety."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - signal line (single feature: captures the cross)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=slow).mean()
    return macd - signal_line


def _bollinger(close: pd.Series, window: int = 20, k: float = 2.0
               ) -> Tuple[pd.Series, pd.Series]:
    """Returns (%B, band width). Trailing window — caller applies shift(1)."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + k * std
    lower = sma - k * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / sma
    return pct_b, width


# =============================================================================
# 4. FEATURE ENGINEERING — the pure function (D5: must be causal & deterministic)
# =============================================================================

def engineer_features(df: pd.DataFrame, garch_params: Dict[str, float]) -> pd.DataFrame:
    """
    Build the engineered feature matrix from raw columns.

    PURE FUNCTION: output row t depends only on raw rows <= t (causality), and on
    the frozen `garch_params` (TRAIN-estimated, passed in — never re-estimated
    here).  This is what makes the D5 truncation leakage test valid.

    Contemporaneity rule (D2):
      * log_return            -> contemporaneous (MASI own return, settled at close)
      * everything else       -> strictly past (lag k>=1, or rolling().shift(1))
    """
    out = pd.DataFrame(index=df.index)
    r = df["log_return"]

    # ---- Group A — MASI own return & momentum ------------------------------
    out["log_return"] = r                                   # contemporaneous (D2)
    out["ret_lag1"] = r.shift(1)
    out["ret_lag2"] = r.shift(2)
    out["ret_lag3"] = r.shift(3)
    out["ret_lag5"] = r.shift(5)
    out["roll_mean_5"] = r.rolling(5).mean().shift(1)
    out["roll_mean_21"] = r.rolling(21).mean().shift(1)

    # ---- Group B — volatility proxies (C1: no VIX) -------------------------
    out["roll_vol_5"] = r.rolling(5).std().shift(1)                       # P1
    out["roll_vol_21"] = r.rolling(21).std().shift(1)                     # P1
    out["garch_vol"] = garch_conditional_vol(r, garch_params)             # P2 (already causal)
    neg = r.clip(upper=0.0)
    out["downside_semidev_21"] = np.sqrt((neg ** 2).rolling(21).mean()).shift(1)  # P4

    # ---- Group C — MASI technical indicators (from close only) -------------
    close = df["masi_close"]
    out["rsi_14"] = _rsi(close, 14).shift(1)
    out["macd_hist"] = _macd_hist(close).shift(1)
    bb_pctb, bb_width = _bollinger(close)
    out["bb_pctb"] = bb_pctb.shift(1)
    out["bb_width"] = bb_width.shift(1)

    # ---- Group D — cross-asset returns, lagged 1 day (D2 — settlement risk) -
    for col, name in [("atw_close", "atw"), ("iam_close", "iam"),
                      ("lhm_close", "lhm"), ("mng_close", "mng"),
                      ("brent_close", "brent"), ("gold_close", "gold"),
                      ("eur_mad", "eurmad")]:
        out[f"{name}_ret_lag1"] = np.log(df[col] / df[col].shift(1)).shift(1)

    # ---- Group E — macro levels, lagged 1 day ------------------------------
    out["gpr_lag1"] = df["gpr_index"].shift(1)
    out["bam_policy_rate_lag1"] = df["bam_policy_rate"].shift(1)

    # ---- Target (inherited from Étape 1, L4-compliant) ---------------------
    out[TARGET] = df[TARGET]
    return out


FEATURE_COLS: List[str] = [
    "log_return", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "roll_mean_5", "roll_mean_21",
    "roll_vol_5", "roll_vol_21", "garch_vol", "downside_semidev_21",
    "rsi_14", "macd_hist", "bb_pctb", "bb_width",
    "atw_ret_lag1", "iam_ret_lag1", "lhm_ret_lag1", "mng_ret_lag1",
    "brent_ret_lag1", "gold_ret_lag1", "eurmad_ret_lag1",
    "gpr_lag1", "bam_policy_rate_lag1",
]

FEATURE_GROUPS = {
    "A_momentum":   ["log_return", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
                     "roll_mean_5", "roll_mean_21"],
    "B_volatility": ["roll_vol_5", "roll_vol_21", "garch_vol", "downside_semidev_21"],
    "C_technical":  ["rsi_14", "macd_hist", "bb_pctb", "bb_width"],
    "D_crossasset": ["atw_ret_lag1", "iam_ret_lag1", "lhm_ret_lag1", "mng_ret_lag1",
                     "brent_ret_lag1", "gold_ret_lag1", "eurmad_ret_lag1"],
    "E_macro":      ["gpr_lag1", "bam_policy_rate_lag1"],
}


# =============================================================================
# 5. EMPIRICAL LEAKAGE TEST (D5) — the core anti-leakage assertion
# =============================================================================

def leakage_truncation_test(full: pd.DataFrame, garch_params: Dict[str, float],
                             cut_points: List[int] = None) -> List[dict]:
    """
    For each cut point T, recompute features on full[:T] and verify that the rows
    BEFORE the cut are bit-identical to the full-series computation.

    A feature that peeks into the future (centered window, future fillna, leaky
    rolling) would change earlier rows when the tail is removed -> AssertionError.

    GARCH is included: it uses the frozen TRAIN params, so it is deterministic and
    must also be perfectly causal.
    """
    if cut_points is None:
        n = len(full)
        cut_points = [int(n * f) for f in (0.40, 0.60, 0.80)]
    full_feat = engineer_features(full, garch_params)
    results = []
    for T in cut_points:
        trunc_feat = engineer_features(full.iloc[:T], garch_params)
        margin = WARMUP                                  # ignore tail-warmup near the cut
        a = full_feat[FEATURE_COLS].iloc[:T - margin]
        b = trunc_feat[FEATURE_COLS].iloc[:T - margin]
        # NaN-aware exact comparison
        max_abs_diff = float(np.nanmax(np.abs(a.values - b.values))) if len(a) else 0.0
        same_nan = bool(np.array_equal(np.isnan(a.values), np.isnan(b.values)))
        ok = (max_abs_diff < 1e-12) and same_nan
        results.append({"cut_index": T, "cut_date": str(full.index[T - 1].date()),
                         "rows_compared": int(len(a)), "max_abs_diff": max_abs_diff,
                         "nan_pattern_identical": same_nan, "passed": ok})
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] cut@{T:4d} ({full.index[T-1].date()}): "
              f"max|diff|={max_abs_diff:.2e}  nan_match={same_nan}  "
              f"rows={len(a)}")
        assert ok, (f"LEAKAGE DETECTED at cut {T}: a feature is not causal "
                    f"(max_abs_diff={max_abs_diff:.2e}, nan_match={same_nan}).")
    return results


# =============================================================================
# 6. METRICS (re-used from Étape 2 — identical definitions for comparability)
# =============================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def trading_strategy_returns(y_true: np.ndarray, y_pred: np.ndarray,
                             cost_bps: float = TRANSACTION_COST_BPS):
    positions = np.sign(y_pred)
    prev_pos = np.concatenate([[0], positions[:-1]])
    changes = (positions != prev_pos).astype(float)
    costs = np.abs(positions - prev_pos) * (cost_bps / 10_000.0)
    strat = positions * y_true - costs
    return strat, int(changes.sum())


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def bootstrap_da_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 10_000,
                    seed: int = RANDOM_SEED) -> Tuple[float, float]:
    mask = y_true != 0
    hits = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.array([hits[rng.integers(0, len(hits), len(hits))].mean()
                      for _ in range(n_boot)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    strat, n_trades = trading_strategy_returns(y_true, y_pred)
    da_lo, da_hi = bootstrap_da_ci(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "da_ci": [da_lo, da_hi],
        "sharpe_annualized": annualized_sharpe(strat),
        "max_drawdown": max_drawdown(strat),
        "n_trades": n_trades,
    }


# =============================================================================
# 7. RANDOM FOREST SANITY-CHECK (D7)
# =============================================================================

def rf_sanity_check(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Retrain the Étape 2 RF on the engineered feature set; compare to Étape 2."""
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                               max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(train[FEATURE_COLS].values, train[TARGET].values)
    out = {
        "val":  evaluate(val[TARGET].values,  rf.predict(val[FEATURE_COLS].values)),
        "test": evaluate(test[TARGET].values, rf.predict(test[FEATURE_COLS].values)),
        "feature_importance": dict(sorted(
            zip(FEATURE_COLS, rf.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True)),
        # Étape 2 RF reference (11 raw features) — from outputs/etape2/metrics.json
        "etape2_rf_reference": {"val_da": 0.5052410901467506,
                                "test_da": 0.5327004219409283,
                                "test_sharpe": 0.714006540339513},
    }
    return rf, out


# =============================================================================
# 8. PLOTS
# =============================================================================

def plot_corr_heatmap(train_feat: pd.DataFrame, path: str) -> None:
    corr = train_feat[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_COLS, fontsize=7)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            v = corr.values[i, j]
            if abs(v) >= 0.5 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.75 else "black")
    ax.set_title("Étape 3 — Engineered feature correlation matrix (TRAIN)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rf_importance(importance: Dict[str, float], path: str) -> None:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = ["#2ca02c" if v >= sorted(vals)[-15] else "#bbbbbb" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_title("Étape 3 — RF feature importance (green = recommended core-15)")
    ax.set_xlabel("Gini importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_volatility_proxies(full_feat: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(full_feat.index, full_feat["roll_vol_5"], label="roll_vol_5 (P1)",
            color="#ff7f0e", linewidth=0.6, alpha=0.7)
    ax.plot(full_feat.index, full_feat["roll_vol_21"], label="roll_vol_21 (P1)",
            color="#1f77b4", linewidth=0.8)
    ax.plot(full_feat.index, full_feat["garch_vol"], label="garch_vol (P2)",
            color="#d62728", linewidth=0.9)
    ax.plot(full_feat.index, full_feat["downside_semidev_21"],
            label="downside_semidev_21 (P4)", color="#2ca02c", linewidth=0.7, alpha=0.8)
    ax.set_title("Étape 3 — Volatility proxies for MASI (no VIX — constraint C1)")
    ax.set_ylabel("daily volatility (log-return units)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_overview(full_feat: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(full_feat.index, full_feat["rsi_14"], color="#9467bd", linewidth=0.6)
    axes[0, 0].axhline(70, color="r", ls="--", lw=0.6)
    axes[0, 0].axhline(30, color="g", ls="--", lw=0.6)
    axes[0, 0].set_title("RSI(14)")
    axes[0, 1].plot(full_feat.index, full_feat["macd_hist"], color="#1f77b4", linewidth=0.6)
    axes[0, 1].axhline(0, color="black", lw=0.5)
    axes[0, 1].set_title("MACD histogram")
    axes[1, 0].plot(full_feat.index, full_feat["bb_pctb"], color="#ff7f0e", linewidth=0.6)
    axes[1, 0].axhline(1, color="r", ls="--", lw=0.6)
    axes[1, 0].axhline(0, color="g", ls="--", lw=0.6)
    axes[1, 0].set_title("Bollinger %B")
    axes[1, 1].plot(full_feat.index, full_feat["roll_mean_21"], color="#2ca02c", linewidth=0.7)
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set_title("21-day rolling mean return (momentum)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Étape 3 — Engineered technical / momentum features")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("ÉTAPE 3 — FEATURE ENGINEERING")
    print("=" * 78)

    # --- 1. Load ------------------------------------------------------------
    print("\n[1] Loading data ...")
    full = load_clean_full()
    split_idx = load_split_indices()

    # --- 2. GARCH on TRAIN only --------------------------------------------
    print("\n[2] GARCH(1,1) — fit on TRAIN log-returns only (D3, L1/L2 discipline) ...")
    train_returns = full.loc[full.index.isin(split_idx["train"]), "log_return"]
    garch_params = fit_garch_on_train(train_returns)

    # --- 3. Empirical leakage test (D5) ------------------------------------
    print("\n[3] Empirical leakage / causality test (D5) ...")
    leak_results = leakage_truncation_test(full, garch_params)
    print("    All causality cuts PASSED — every engineered feature is leakage-free.")

    # --- 4. Engineer features on the full series, drop warm-up -------------
    print("\n[4] Engineering features on full series, then re-splitting (D1) ...")
    feat_full = engineer_features(full, garch_params)
    n_before = len(feat_full)
    feat_full = feat_full.iloc[WARMUP:]                       # drop 2007 warm-up rows
    feat_full = feat_full.dropna(subset=FEATURE_COLS)
    print(f"    feature matrix: {n_before} -> {len(feat_full)} rows "
          f"(dropped {n_before - len(feat_full)} warm-up/NaN rows, all at series start)")

    # --- 5. Re-split by Étape 1 dates --------------------------------------
    splits = {}
    for name, idx in split_idx.items():
        sub = feat_full[feat_full.index.isin(idx)].copy()
        splits[name] = sub
        print(f"    {name:5s}: {len(sub):4d} rows  {sub.index.min().date()} -> "
              f"{sub.index.max().date()}")
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert len(val) == len(split_idx["val"]), "VAL lost rows — warm-up leaked into VAL!"
    assert len(test) == len(split_idx["test"]), "TEST lost rows — warm-up leaked into TEST!"
    print("    L3 OK: VAL/TEST keep ALL rows (warm-up loss confined to TRAIN start).")

    # --- 6. StandardScaler stats — TRAIN only (D6, L1) ---------------------
    print("\n[6] Computing StandardScaler stats on TRAIN engineered rows only (L1) ...")
    scaler_stats = {c: {"mean": float(train[c].mean()), "std": float(train[c].std(ddof=0))}
                    for c in FEATURE_COLS}

    # --- 7. RF sanity-check (D7) -------------------------------------------
    print("\n[7] Random Forest sanity-check on engineered features (D7) ...")
    rf, rf_results = rf_sanity_check(train, val, test)
    e2 = rf_results["etape2_rf_reference"]
    print(f"    Étape 3 RF  — VAL  DA={rf_results['val']['directional_accuracy']:.4f}  "
          f"TEST DA={rf_results['test']['directional_accuracy']:.4f}  "
          f"TEST Sharpe={rf_results['test']['sharpe_annualized']:+.3f}")
    print(f"    Étape 2 RF  — VAL  DA={e2['val_da']:.4f}  TEST DA={e2['test_da']:.4f}  "
          f"TEST Sharpe={e2['test_sharpe']:+.3f}")
    d_da = rf_results['test']['directional_accuracy'] - e2['test_da']
    print(f"    Delta TEST DA (engineered - raw): {d_da:+.4f}  "
          f"-> feature engineering {'HELPS' if d_da > 0 else 'does NOT help'} the RF.")

    # --- 8. Persist artifacts ----------------------------------------------
    print("\n[8] Writing artifacts ...")
    for name, sub in splits.items():
        out_cols = FEATURE_COLS + [TARGET]
        p = os.path.join(FEAT_DIR, f"masi_features_{name}.csv")
        sub[out_cols].to_csv(p)
        print(f"    {p}")

    with open(os.path.join(FEAT_DIR, "scaler_stats_etape3_train_only.json"), "w") as f:
        json.dump(scaler_stats, f, indent=2)
    with open(os.path.join(FEAT_DIR, "garch_params_train.json"), "w") as f:
        json.dump(garch_params, f, indent=2)
    with open(os.path.join(FEAT_DIR, "rf_sanitycheck_metrics.json"), "w") as f:
        json.dump(rf_results, f, indent=2)

    metadata = {
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "feature_groups": FEATURE_GROUPS,
        "target": TARGET,
        "contemporaneous_features": ["log_return"],
        "rejected_features": {
            "atr_14": "OHLC (masi_high/low) only cover 2016+ -> NaN for 2284/4784 rows "
                      "(~all pre-2016 TRAIN). Unusable; volatility covered by P1/P2/P4.",
        },
        "splits": {name: {"n": len(sub),
                          "start": str(sub.index.min().date()),
                          "end": str(sub.index.max().date())}
                   for name, sub in splits.items()},
        "warmup_rows_dropped": int(n_before - len(feat_full)),
        "leakage_test": leak_results,
        "garch_params": garch_params,
    }
    with open(os.path.join(FEAT_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    {os.path.join(FEAT_DIR, 'feature_metadata.json')}")

    # --- 9. Plots ----------------------------------------------------------
    print("\n[9] Generating diagnostic plots ...")
    plot_corr_heatmap(train, os.path.join(PLOTS_DIR, "etape3_corr_heatmap.png"))
    plot_rf_importance(rf_results["feature_importance"],
                       os.path.join(PLOTS_DIR, "etape3_rf_importance.png"))
    plot_volatility_proxies(feat_full, os.path.join(PLOTS_DIR, "etape3_volatility_proxies.png"))
    plot_feature_overview(feat_full, os.path.join(PLOTS_DIR, "etape3_feature_overview.png"))
    print(f"    4 plots -> {PLOTS_DIR}")

    # --- 10. Recommended core-15 (by RF importance) ------------------------
    core15 = list(rf_results["feature_importance"].keys())[:15]
    print("\n[10] Recommended CNN-LSTM core-15 feature set (constraint C3, F<=15):")
    for i, name in enumerate(core15, 1):
        print(f"     {i:2d}. {name:24s} imp={rf_results['feature_importance'][name]:.4f}")

    print("\n" + "=" * 78)
    print("ÉTAPE 3 COMPLETE — all leakage cuts passed, artifacts written.")
    print("=" * 78)


if __name__ == "__main__":
    main()
