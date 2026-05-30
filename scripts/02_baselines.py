"""
================================================================================
ÉTAPE 2 — Mandatory Baselines (BEFORE any deep learning)
MASI Hybrid Forecasting System
================================================================================

Per prompt.md RULE 8: DL is FORBIDDEN until these 4 baselines are evaluated:
  1. Random Walk (naive: prediction = 0, no drift)
  2. Historical Mean (prediction = train mean return)
  3. ARIMA (auto-selected via AIC on small grid; fixed-coef forecast on val/test)
  4. Random Forest (multi-factor inputs, 100 trees, no leakage)

INPUT:
  outputs/etape1/splits/masi_{train,val,test}.csv
  (created by scripts/01_preprocessing.py)

OUTPUT:
  outputs/etape2/predictions_{train,val,test}.csv  — all 4 baselines side-by-side
  outputs/etape2/metrics.json                       — full metrics + bootstrap CI
  reports/figures/etape2/*.png                                — diagnostic plots

Metrics computed for each baseline on VAL and TEST:
  - RMSE, MAE                    (regression error)
  - Directional Accuracy (DA)    (% correct sign predictions)
  - Annualized Sharpe ratio      (with 5 bps round-trip transaction cost)
  - Maximum Drawdown (MDD)       (worst peak-to-trough of strategy cumulative returns)
  - Bootstrap 95% CI             (10,000 resamples for RMSE and DA)

Anti-leakage:
  L1 ✓ — Scaler from Étape 1 (TRAIN only); we use raw + log_return as features
  L3 ✓ — All input features are values at time t (no future info)
  L4 ✓ — Target = target_y_next was constructed AFTER features in Étape 1
  L5 ✓ — Signal at t → return realized at t+1 (1-step ahead)
  L6 ✓ — Strict TRAIN/VAL/TEST partitions, no peeking

Author: Quantitative Research Lab
Date:   2026-05-20
================================================================================
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape1", "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape2")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape2")
REPORT_FILE = os.path.join(PROJECT_ROOT, "outputs", "etape2", "report.md")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5         # one-way; 10 bps round-trip (MASI realistic, Deep 2025)
BOOTSTRAP_RESAMPLES = 10_000     # Deep 2025 standard
RANDOM_SEED = 42

# Feature columns for Random Forest (values at time t — known, no leakage)
RF_FEATURES = [
    "masi_close",
    "atw_close", "iam_close", "lhm_close", "mng_close",
    "brent_close", "gold_close", "eur_mad", "gpr_index", "bam_policy_rate",
    "log_return",  # contemporaneous return = ln(P_t / P_{t-1}) — fine as input
]
TARGET = "target_y_next"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": os.path.join(SPLITS_DIR, "masi_train.csv"),
        "val":   os.path.join(SPLITS_DIR, "masi_val.csv"),
        "test":  os.path.join(SPLITS_DIR, "masi_test.csv"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}\nRun scripts/01_preprocessing.py first.")
    out = []
    for name, p in paths.items():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        out.append(df)
        print(f"  {name:5s}: {len(df):4d} obs  {df.index.min().date()} -> {df.index.max().date()}")
    return out[0], out[1], out[2]


# =============================================================================
# BASELINE 1: RANDOM WALK
# =============================================================================

def baseline_random_walk(val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Naive: prediction = 0 (efficient market — expected return = 0)."""
    return {
        "val":  np.zeros(len(val_df)),
        "test": np.zeros(len(test_df)),
    }


# =============================================================================
# BASELINE 2: HISTORICAL MEAN
# =============================================================================

def baseline_historical_mean(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Constant prediction = mean of TRAIN target."""
    mu = float(train_df[TARGET].mean())
    print(f"    Historical mean (train): {mu:+.6f}  ({mu*TRADING_DAYS*100:+.2f}% annualized)")
    return {
        "val":  np.full(len(val_df),  mu),
        "test": np.full(len(test_df), mu),
        "_mu":  mu,
    }


# =============================================================================
# BASELINE 3: ARIMA (auto-selected on TRAIN)
# =============================================================================

def baseline_arima(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
                    ) -> Dict[str, np.ndarray]:
    """
    ARIMA on log_return (the contemporaneous return; target is the lead-1 of this).
    Strategy: select (p, q) via AIC on small grid; fit on TRAIN; 1-step ahead forecasts
    on VAL and TEST using rolling extension (re-fit not needed for forecast — coefs fixed).
    """
    y_train = train_df["log_return"].dropna().values
    y_val = val_df["log_return"].dropna().values
    y_test = test_df["log_return"].dropna().values

    # Grid search on (p, q) for d=0 (returns are I(0))
    best = (None, np.inf)
    for p in range(0, 3):
        for q in range(0, 3):
            if p == 0 and q == 0:
                continue
            try:
                m = ARIMA(y_train, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
                if m.aic < best[1]:
                    best = ((p, 0, q), m.aic)
            except Exception:
                continue
    best_order = best[0] or (1, 0, 1)
    print(f"    ARIMA selected order: {best_order}  (AIC={best[1]:.2f})")

    # Refit on TRAIN with best order
    model = ARIMA(y_train, order=best_order).fit(method_kwargs={"warn_convergence": False})

    # 1-step-ahead forecasts: extend the model with each new observation (no re-fit, fixed coefs)
    def rolling_forecast(history_init: np.ndarray, future: np.ndarray) -> np.ndarray:
        """Use fitted params; predict next return given the full series so far."""
        history = list(history_init)
        preds = np.empty(len(future))
        for i in range(len(future)):
            # Apply fitted params to the current history
            m_fit = model.apply(np.asarray(history), refit=False)
            f = m_fit.forecast(steps=1)
            preds[i] = float(f[0])
            history.append(future[i])
        return preds

    print(f"    Rolling 1-step-ahead on VAL ({len(y_val)} steps)...")
    pred_val = rolling_forecast(y_train, y_val)
    print(f"    Rolling 1-step-ahead on TEST ({len(y_test)} steps)...")
    pred_test = rolling_forecast(np.concatenate([y_train, y_val]), y_test)

    # Convert ARIMA forecasts of log_return at t+1 to target_y_next at t
    # Our target is y_t = ln(P_{t+1}/P_t) = log_return[t+1] — exact match
    return {
        "val":  pred_val,
        "test": pred_test,
        "_order": best_order,
        "_aic": best[1],
    }


# =============================================================================
# BASELINE 4: RANDOM FOREST (multi-factor)
# =============================================================================

def baseline_random_forest(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Random Forest with 100 trees on the 11 raw multi-factor inputs at time t.
    Trained on TRAIN, predicts on VAL and TEST.
    """
    features = [c for c in RF_FEATURES if c in train_df.columns]
    print(f"    Features: {features}")
    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_val = val_df[features].values
    X_test = test_df[features].values

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=None,
        min_samples_leaf=5, max_features="sqrt",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return {
        "val":  rf.predict(X_val),
        "test": rf.predict(X_test),
        "_feature_importance": dict(zip(features, rf.feature_importances_.tolist())),
    }


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class Metrics:
    rmse: float
    mae: float
    directional_accuracy: float
    sharpe_annualized: float
    max_drawdown: float
    n_trades: int
    rmse_ci: Tuple[float, float] = None
    da_ci: Tuple[float, float] = None


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def trading_strategy_returns(
    y_true: np.ndarray, y_pred: np.ndarray, cost_bps: float = TRANSACTION_COST_BPS,
) -> Tuple[np.ndarray, int]:
    """
    Trading rule: position[t] = sign(prediction[t])  ∈ {-1, 0, +1}.
    Returns[t] = position[t] * y_true[t]  (because y_true = ln(P_{t+1}/P_t))
    Transaction cost: one-way cost_bps × |Δposition|.
    A direct flip -1 → +1 closes one unit and opens one unit, so it costs 2×.
    """
    positions = np.sign(y_pred)
    prev_pos = np.concatenate([[0], positions[:-1]])
    position_changes = (positions != prev_pos).astype(float)
    cost = np.abs(positions - prev_pos) * (cost_bps / 10_000.0)
    strategy_returns = positions * y_true - cost
    n_trades = int(position_changes.sum())
    return strategy_returns, n_trades


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = BOOTSTRAP_RESAMPLES,
                  alpha: float = 0.05, seed: int = RANDOM_SEED) -> Tuple[float, float]:
    """Generic percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_ci: bool = True) -> Metrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    strat_ret, n_trades = trading_strategy_returns(y_true, y_pred)
    sharpe = annualized_sharpe(strat_ret)
    mdd = max_drawdown(strat_ret)
    m = Metrics(rmse=rmse, mae=mae, directional_accuracy=da,
                sharpe_annualized=sharpe, max_drawdown=mdd, n_trades=n_trades)
    if with_ci:
        # Bootstrap residuals for RMSE CI
        residuals = y_true - y_pred
        def rmse_stat(x): return float(np.sqrt(np.mean(x ** 2)))
        m.rmse_ci = bootstrap_ci(residuals, rmse_stat)
        # Bootstrap correct-sign indicators for DA CI
        mask = y_true != 0
        if mask.sum() > 0:
            hits = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).astype(float)
            m.da_ci = bootstrap_ci(hits, lambda x: float(x.mean()))
    return m


# =============================================================================
# PLOTS
# =============================================================================

def plot_predictions(y_true: pd.Series, preds: Dict[str, np.ndarray], title: str,
                     save_path: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    # Top: actual vs predictions (zoomed sample to be readable)
    axes[0].plot(y_true.index, y_true.values, label="Actual", color="black", linewidth=0.8, alpha=0.7)
    colors = {"random_walk": "#7f7f7f", "historical_mean": "#ff7f0e",
              "arima": "#1f77b4", "random_forest": "#2ca02c"}
    for name, p in preds.items():
        axes[0].plot(y_true.index, p, label=name, color=colors.get(name, "red"), linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.4)
    axes[0].set_title(f"{title} — predictions vs actual")
    axes[0].set_ylabel("log-return")
    axes[0].legend(fontsize=8, ncol=5)
    axes[0].grid(alpha=0.3)

    # Bottom: cumulative strategy returns per model
    for name, p in preds.items():
        sr, _ = trading_strategy_returns(y_true.values, p)
        axes[1].plot(y_true.index, np.exp(np.cumsum(sr)), label=name, color=colors.get(name, "red"))
    axes[1].axhline(1, color="black", linewidth=0.5)
    axes[1].set_title("Cumulative strategy returns (5 bps transaction cost)")
    axes[1].set_ylabel("equity (starting at 1.0)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


def plot_metrics_summary(metrics_per_partition: dict, save_path: str) -> str:
    """Bar chart: RMSE / DA / Sharpe across 4 baselines on val and test."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metric_names = ("rmse", "directional_accuracy", "sharpe_annualized")
    titles = ("RMSE (lower=better)", "Directional Accuracy (>0.5=skill)", "Annualized Sharpe")

    for row_i, partition in enumerate(("val", "test")):
        for col_i, (mname, title) in enumerate(zip(metric_names, titles)):
            models = list(metrics_per_partition[partition].keys())
            vals = [metrics_per_partition[partition][m][mname] for m in models]
            colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]
            axes[row_i, col_i].bar(models, vals, color=colors[:len(models)])
            axes[row_i, col_i].set_title(f"{partition.upper()} — {title}")
            axes[row_i, col_i].grid(alpha=0.3, axis="y")
            if mname == "directional_accuracy":
                axes[row_i, col_i].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random=0.5")
                axes[row_i, col_i].legend(fontsize=7)
            elif mname == "sharpe_annualized":
                axes[row_i, col_i].axhline(0, color="red", linestyle="--", linewidth=0.8)
            for j, v in enumerate(vals):
                axes[row_i, col_i].text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_all(verbose: bool = True) -> dict:
    print("=" * 72)
    print(" ÉTAPE 2 — Mandatory Baselines (RW, Hist Mean, ARIMA, RF)")
    print(" Per prompt.md RULE 8: DL FORBIDDEN until all 4 reported")
    print("=" * 72)

    # --- Load splits
    print("\n[1/5] Loading splits...")
    train_df, val_df, test_df = load_splits()

    y_val = val_df[TARGET].values
    y_test = test_df[TARGET].values

    # --- Run baselines
    print("\n[2/5] Running baselines...")

    print("  (1/4) Random Walk...")
    rw = baseline_random_walk(val_df, test_df)

    print("  (2/4) Historical Mean...")
    hm = baseline_historical_mean(train_df, val_df, test_df)

    print("  (3/4) ARIMA...")
    ar = baseline_arima(train_df, val_df, test_df)

    print("  (4/4) Random Forest...")
    rf = baseline_random_forest(train_df, val_df, test_df)

    predictions = {
        "random_walk":     {"val": rw["val"], "test": rw["test"]},
        "historical_mean": {"val": hm["val"], "test": hm["test"]},
        "arima":           {"val": ar["val"], "test": ar["test"]},
        "random_forest":   {"val": rf["val"], "test": rf["test"]},
    }

    # --- Metrics
    print("\n[3/5] Computing metrics (with bootstrap CI, 10k resamples)...")
    metrics = {"val": {}, "test": {}}
    for name, preds in predictions.items():
        print(f"  {name}...")
        metrics["val"][name] = asdict(compute_metrics(y_val, preds["val"]))
        metrics["test"][name] = asdict(compute_metrics(y_test, preds["test"]))

    # --- Persist predictions + metrics
    print("\n[4/5] Saving artifacts...")
    pred_val_df = pd.DataFrame({"date": val_df.index, "actual": y_val,
                                 **{n: predictions[n]["val"] for n in predictions}})
    pred_test_df = pd.DataFrame({"date": test_df.index, "actual": y_test,
                                  **{n: predictions[n]["test"] for n in predictions}})
    pred_val_df.to_csv(os.path.join(RESULTS_DIR, "predictions_val.csv"), index=False)
    pred_test_df.to_csv(os.path.join(RESULTS_DIR, "predictions_test.csv"), index=False)

    metrics_serializable = {
        "val": metrics["val"], "test": metrics["test"],
        "arima_order": list(ar["_order"]),
        "arima_aic": ar["_aic"],
        "historical_mean_train": hm["_mu"],
        "rf_feature_importance": rf["_feature_importance"],
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)

    # --- Plots
    print("\n[5/5] Generating plots...")
    p1 = plot_predictions(
        val_df[TARGET], {n: predictions[n]["val"] for n in predictions},
        "VAL set", os.path.join(PLOTS_DIR, "etape2_predictions_val.png"))
    p2 = plot_predictions(
        test_df[TARGET], {n: predictions[n]["test"] for n in predictions},
        "TEST set", os.path.join(PLOTS_DIR, "etape2_predictions_test.png"))
    p3 = plot_metrics_summary(metrics, os.path.join(PLOTS_DIR, "etape2_metrics_summary.png"))
    for p in (p1, p2, p3):
        print(f"  saved: {os.path.relpath(p, PROJECT_ROOT)}")

    # --- Console summary
    print("\n" + "=" * 72)
    print(" RESULTS SUMMARY")
    print("=" * 72)
    print(f"\n{'Baseline':<20s}{'RMSE':>10s}{'MAE':>10s}{'DA':>10s}{'Sharpe':>10s}{'MDD':>10s}{'Trades':>9s}")
    for part in ("val", "test"):
        print(f"\n[{part.upper()}]")
        for name in predictions:
            m = metrics[part][name]
            print(f"  {name:<18s}{m['rmse']:>10.6f}{m['mae']:>10.6f}"
                  f"{m['directional_accuracy']:>10.4f}{m['sharpe_annualized']:>10.3f}"
                  f"{m['max_drawdown']:>10.3f}{m['n_trades']:>9d}")
    print("\n" + "=" * 72)

    return {
        "train": train_df, "val": val_df, "test": test_df,
        "predictions": predictions, "metrics": metrics,
        "arima_info": {"order": ar["_order"], "aic": ar["_aic"]},
        "rf_importance": rf["_feature_importance"],
        "plot_paths": [p1, p2, p3],
    }


if __name__ == "__main__":
    artifacts = run_all(verbose=True)
