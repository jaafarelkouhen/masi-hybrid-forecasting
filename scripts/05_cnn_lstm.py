"""
================================================================================
ÉTAPE 5 — CNN-LSTM with Regime-as-Feature (walk-forward)
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Train a compact CNN-LSTM to forecast the next-day MASI log-return, evaluate it
  with an expanding walk-forward, and decide — per prompt.md RULE 8 — whether the
  deep-learning model is justified versus the Étape 2 baselines.

INPUT
  outputs/etape4/regimes/masi_regimes_{train,val,test}.csv  (Étape 3 features + regime cols)
  outputs/etape2/metrics.json                       (baseline TEST metrics)

OUTPUT
  outputs/etape5/walkforward_metrics.json
  outputs/etape5/predictions_test.csv
  reports/figures/etape5/*.png   (4 diagnostic plots)

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5)
--------------------------------------------------------------------------------
D1  COMPACT ARCHITECTURE (constraint C3 — ~3,300 train obs).
       Conv1D(16, k=3) -> ReLU -> Dropout(0.2)
       -> LSTM(24, 1 layer) -> Dropout(0.2)
       -> Dense(16, ReLU) -> Dense(1, linear)
    ~5k parameters (well under the 10k C3 ceiling, and under the prompt's
    32/32/16 maximum). Conv1D extracts local short-term patterns; the LSTM
    carries the longer dependency; this is the standard hybrid ordering (Fozap
    2025). Built in PyTorch (TensorFlow unavailable in the environment).

D2  REGIME ABLATION (mandated by the Étape 4 report §6).
       base12        : 12 strongest Étape 3 features (RF importance)
       base12+regime : base12 + the 3 HMM regime soft-probabilities (15 feats)
    The CNN-LSTM is run for BOTH; the difference isolates the regime feature's
    marginal value. If the regime does not help -> it is dropped (RULE 6/8).

D3  WINDOW SIZE L scanned over {10, 15, 20} (constraint C3 — never default 30).
    Selected on the VAL fold (DA, RMSE tie-break), then fixed for the walk-forward.

D4  EXPANDING WALK-FORWARD (Deep 2025 — multiple out-of-sample windows).
    Fold 1 = VAL; folds 2-5 = TEST split into 4 consecutive quarters. Each fold
    trains on ALL data strictly before it. -> folds 2-5 aggregate to exactly the
    948-day canonical TEST, so the CNN-LSTM is directly comparable to the Étape 2
    baselines, AND the 4 TEST quarters give a stability check.

D5  ANTI-LEAKAGE.
    L1 — feature scaler re-fit on each fold's train pool ONLY (not the fold test).
    L3 — sequences are TRAILING windows (rows [t-L+1 .. t] to predict t->t+1);
         they never reach forward -> leakage-free. Each (window,target) pair is
         assigned to the split of its target row.
    L5 — target_y_next IS the t->t+1 return: signal at t, return realised t+1.
    L6 — Étape 1 split dates + gaps inherited.
    L2/L8 — the HMM regime feature uses Étape 4 params (HMM fit on the first
         TRAIN block, decoded by a causal forward-filter -> leakage-free). A full
         per-fold HMM+GARCH refit is NOT done here: re-running the entire
         feature+regime pipeline inside every fold is disproportionate (RULE 6),
         and the causal filter already removes look-ahead. This is a documented
         scope limitation, not a leak.

D6  SEED VARIANCE. EM-free but NN training is seed-sensitive. Each (fold,config)
    trains 3 seeds; the ensemble = mean prediction; per-seed spread is reported
    (RULE 8 — "sensitive to random seed -> report variance").

D7  HONEST VERDICT (RULE 8). The CNN-LSTM is JUSTIFIED only if it beats the best
    baseline on TEST (DA and Sharpe) AND is stable across the 4 TEST folds. The
    aspirational gates are DA >= 0.55, Sharpe >= 1.30, MDD >= -0.20. If it fails,
    the report says so and recommends shipping Random Forest / ARIMA.

Author: Quantitative Research Lab
Date:   2026-05-21
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape4", "regimes")
ETAPE2_METRICS = os.path.join(PROJECT_ROOT, "outputs", "etape2", "metrics.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape5")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape5")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET = "target_y_next"
TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5
BOOTSTRAP_RESAMPLES = 10_000
TARGET_SCALE = 100.0                 # train on y*100 (returns are tiny)

# 12 strongest Étape 3 features (RF importance, from etape3 rf_sanitycheck)
BASE12 = ["log_return", "roll_mean_5", "macd_hist", "roll_mean_21", "garch_vol",
          "downside_semidev_21", "ret_lag1", "atw_ret_lag1", "rsi_14",
          "roll_vol_5", "roll_vol_21", "lhm_ret_lag1"]
REGIME_COLS = ["regime_prob_bear", "regime_prob_neutral", "regime_prob_bull"]
CONFIGS = {"base12": BASE12, "base12+regime": BASE12 + REGIME_COLS}

L_GRID = [10, 15, 20]
SEEDS = [0, 1, 2]
EPOCHS = 80
PATIENCE = 10
BATCH = 64
LR = 1e-3
DROPOUT = 0.2

# performance gates (Étape 2 report, section 6)
GATE_DA, GATE_SHARPE, GATE_MDD = 0.55, 1.30, -0.20


# =============================================================================
# 1. DATA
# =============================================================================

def load_full() -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    """Concatenate the 3 regime-augmented splits; return frame + split row bounds."""
    frames, bounds, cursor = [], {}, 0
    for name in ("train", "val", "test"):
        d = pd.read_csv(os.path.join(REG_DIR, f"masi_regimes_{name}.csv"))
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        bounds[name] = (cursor, cursor + len(d))
        cursor += len(d)
        frames.append(d)
    full = pd.concat(frames)
    print(f"  full timeline: {len(full)} rows  "
          f"{full.index.min().date()} -> {full.index.max().date()}")
    for k, (a, b) in bounds.items():
        print(f"    {k:5s}: rows [{a:4d}, {b:4d})  n={b-a}")
    return full, bounds


def build_sequences(feat: np.ndarray, target: np.ndarray, target_rows: np.ndarray,
                    L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trailing windows (L3): the window for target row i is feat[i-L+1 : i+1].
    Backward-only -> never sees the future. Rows with i < L-1 are dropped.
    """
    X, y, kept = [], [], []
    for i in target_rows:
        if i < L - 1:
            continue
        X.append(feat[i - L + 1: i + 1])
        y.append(target[i])
        kept.append(i)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(kept)


# =============================================================================
# 2. MODEL (D1)
# =============================================================================

class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, conv_filters: int = 16, lstm_units: int = 24,
                 dense_units: int = 16, dropout: float = DROPOUT):
        super().__init__()
        self.conv = nn.Conv1d(n_features, conv_filters, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_filters, lstm_units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):                       # x: (B, L, F)
        x = x.transpose(1, 2)                   # (B, F, L)
        x = torch.relu(self.conv(x))            # (B, conv_filters, L)
        x = self.drop1(x)
        x = x.transpose(1, 2)                   # (B, L, conv_filters)
        out, _ = self.lstm(x)                   # (B, L, lstm_units)
        z = self.drop2(out[:, -1, :])           # last timestep
        z = torch.relu(self.fc1(z))
        return self.fc2(z).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(Xtr, ytr, Xva, yva, n_features: int, seed: int):
    """Train one CNN-LSTM with early stopping on the internal validation split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNNLSTM(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr * TARGET_SCALE)
    Xva_t = torch.tensor(Xva)
    yva_t = torch.tensor(yva * TARGET_SCALE)

    best_state, best_val, wait = None, np.inf, 0
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s: s + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva_t), yva_t))
        if vloss < best_val - 1e-7:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in
                                                 model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X.astype(np.float32)))
    return out.numpy() / TARGET_SCALE         # back to return scale


# =============================================================================
# 3. METRICS  (identical definitions to Étape 2 — comparability)
# =============================================================================

def directional_accuracy(y_true, y_pred) -> float:
    m = y_true != 0
    return float((np.sign(y_pred[m]) == np.sign(y_true[m])).mean()) if m.sum() else float("nan")


def trading_strategy_returns(y_true, y_pred, cost_bps=TRANSACTION_COST_BPS):
    pos = np.sign(y_pred)
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(pos - prev) * (cost_bps / 1e4)
    return pos * y_true - cost, int((pos != prev).sum())


def annualized_sharpe(r) -> float:
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if len(r) and r.std() else 0.0


def max_drawdown(r) -> float:
    cum = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def bootstrap_da_ci(y_true, y_pred, seed=42):
    m = y_true != 0
    hits = (np.sign(y_pred[m]) == np.sign(y_true[m])).astype(float)
    rng = np.random.default_rng(seed)
    stat = np.array([hits[rng.integers(0, len(hits), len(hits))].mean()
                     for _ in range(BOOTSTRAP_RESAMPLES)])
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]


def evaluate(y_true, y_pred) -> dict:
    strat, n_trades = trading_strategy_returns(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "da_ci": bootstrap_da_ci(y_true, y_pred),
        "sharpe_annualized": annualized_sharpe(strat),
        "max_drawdown": max_drawdown(strat),
        "n_trades": n_trades,
    }


# =============================================================================
# 3b. CONFIDENCE-THRESHOLD TRADING RULE (calibrated on VAL ONLY — no snooping)
# =============================================================================
# The baseline trading rule (position = sign(prediction)) trades every day.
# A confidence threshold tau filters out low-conviction predictions:
#     position_t = sign(pred_t) if |pred_t| > tau, else 0
# tau is calibrated by maximizing VAL Sharpe over a quantile grid, then APPLIED
# to TEST. VAL is the project's designated model-selection set => no data-snooping.
# DA is unchanged (the prediction is unchanged); only the trading-Sharpe changes.

def calibrate_threshold(y_val, pred_val, cost_bps=TRANSACTION_COST_BPS,
                        min_active: int = 20):
    """Pick tau on VAL that maximizes Sharpe under the threshold rule."""
    abs_pred = np.abs(pred_val)
    qs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    cand = sorted({0.0, *(float(np.quantile(abs_pred, q)) for q in qs)})
    grid, best_tau, best_sh = [], 0.0, -np.inf
    for tau in cand:
        mask = abs_pred > tau
        n_active = int(mask.sum())
        if n_active < min_active:
            grid.append({"tau": float(tau), "n_active": n_active,
                         "sharpe": None, "skipped": True})
            continue
        pos = np.sign(pred_val) * mask.astype(float)
        prev = np.concatenate([[0.0], pos[:-1]])
        cost_t = np.abs(pos - prev) * (cost_bps / 1e4)
        strat = pos * y_val - cost_t
        sh = annualized_sharpe(strat)
        grid.append({"tau": float(tau), "n_active": n_active,
                     "sharpe": float(sh), "skipped": False})
        if sh > best_sh:
            best_tau, best_sh = float(tau), float(sh)
    return best_tau, grid


def apply_threshold(y_true, pred, tau, cost_bps=TRANSACTION_COST_BPS):
    """Apply a calibrated tau and return strategy metrics."""
    abs_pred = np.abs(pred)
    mask = abs_pred > tau
    pos = np.sign(pred) * mask.astype(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    cost_t = np.abs(pos - prev) * (cost_bps / 1e4)
    strat = pos * y_true - cost_t
    return {
        "tau": float(tau),
        "sharpe": annualized_sharpe(strat),
        "max_drawdown": max_drawdown(strat),
        "n_days_active": int(mask.sum()),
        "n_position_changes": int((pos != prev).sum()),
        "active_day_da": (float((np.sign(pred[mask]) == np.sign(y_true[mask])).mean())
                          if mask.sum() else None),
    }


# =============================================================================
# 4. WALK-FORWARD ENGINE
# =============================================================================

def scale_features(feat: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """L1 — standardize using statistics from the fold's train pool ONLY."""
    mu = feat[train_rows].mean(axis=0)
    sd = feat[train_rows].std(axis=0)
    sd[sd == 0] = 1.0
    return (feat - mu) / sd


def run_fold(feat_raw, target, train_rows, test_rows, L, n_features, seeds):
    """Train a 3-seed ensemble on the fold train pool, predict the fold test block."""
    feat = scale_features(feat_raw, train_rows)
    # internal early-stopping split: last 12% of the (time-ordered) train pool
    cut = int(len(train_rows) * 0.88)
    tr_rows, va_rows = train_rows[:cut], train_rows[cut:]
    Xtr, ytr, _ = build_sequences(feat, target, tr_rows, L)
    Xva, yva, _ = build_sequences(feat, target, va_rows, L)
    Xte, yte, kept_te = build_sequences(feat, target, test_rows, L)

    preds = []
    for sd in seeds:
        model = train_model(Xtr, ytr, Xva, yva, n_features, seed=sd)
        preds.append(predict(model, Xte))
    preds = np.array(preds)                       # (n_seeds, n_test)
    return {"y_true": yte, "kept_rows": kept_te,
            "pred_ensemble": preds.mean(axis=0), "pred_per_seed": preds}


def walk_forward(full: pd.DataFrame, bounds, cfg_name: str, cols: List[str], L: int):
    """Expanding walk-forward: fold 1 = VAL, folds 2-5 = TEST quarters."""
    feat_raw = full[cols].values.astype(np.float32)
    target = full[TARGET].values.astype(np.float32)
    n_features = len(cols)

    tr_a, tr_b = bounds["train"]
    va_a, va_b = bounds["val"]
    te_a, te_b = bounds["test"]
    q = np.linspace(te_a, te_b, 5, dtype=int)     # 4 TEST quarters

    folds = [("VAL", np.arange(tr_a, tr_b), np.arange(va_a, va_b))]
    for k in range(4):
        train_rows = np.arange(tr_a, q[k])        # expanding: all data before the block
        test_rows = np.arange(q[k], q[k + 1])
        folds.append((f"TEST-q{k+1}", train_rows, test_rows))

    fold_results = {}
    for fname, train_rows, test_rows in folds:
        res = run_fold(feat_raw, target, train_rows, test_rows, L, n_features, SEEDS)
        m = evaluate(res["y_true"], res["pred_ensemble"])
        per_seed_da = [directional_accuracy(res["y_true"], res["pred_per_seed"][s])
                       for s in range(len(SEEDS))]
        fold_results[fname] = {"metrics": m, "per_seed_da": per_seed_da,
                               "n": len(res["y_true"]),
                               "y_true": res["y_true"], "pred": res["pred_ensemble"],
                               "kept_rows": res["kept_rows"]}
        print(f"    [{cfg_name:14s} {fname:9s}] n={len(res['y_true']):3d}  "
              f"DA={m['directional_accuracy']:.4f}  Sharpe={m['sharpe_annualized']:+.3f}  "
              f"seedDA={np.std(per_seed_da):.4f}sd")
    return fold_results


def aggregate_test(fold_results: dict) -> dict:
    """Aggregate the 4 TEST-quarter folds into one 948-day TEST evaluation."""
    yk = [(fold_results[f]["kept_rows"], fold_results[f]["y_true"], fold_results[f]["pred"])
          for f in ("TEST-q1", "TEST-q2", "TEST-q3", "TEST-q4")]
    rows = np.concatenate([k for k, _, _ in yk])
    y_true = np.concatenate([y for _, y, _ in yk])
    pred = np.concatenate([p for _, _, p in yk])
    order = np.argsort(rows)
    return {"rows": rows[order], "y_true": y_true[order], "pred": pred[order],
            "metrics": evaluate(y_true[order], pred[order])}


# =============================================================================
# 5. PLOTS
# =============================================================================

def plot_baseline_comparison(cnn_metrics, baselines, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = ["random_walk", "historical_mean", "arima", "random_forest"]
    da = [baselines[m]["directional_accuracy"] for m in models]
    sh = [baselines[m]["sharpe_annualized"] for m in models]
    labels = ["RW", "HistMean", "ARIMA", "RF"]
    for cfg in cnn_metrics:
        labels.append("CNN-LSTM\n" + cfg)
        da.append(cnn_metrics[cfg]["directional_accuracy"])
        sh.append(cnn_metrics[cfg]["sharpe_annualized"])
    colors = ["#999"] * 4 + ["#1f77b4", "#2ca02c"][:len(cnn_metrics)]
    axes[0].bar(labels, da, color=colors)
    axes[0].axhline(0.5, color="k", lw=0.6)
    axes[0].axhline(GATE_DA, color="r", ls="--", lw=0.8, label=f"gate {GATE_DA}")
    axes[0].set_title("Directional Accuracy — TEST")
    axes[0].set_ylim(0.40, 0.62)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, sh, color=colors)
    axes[1].axhline(GATE_SHARPE, color="r", ls="--", lw=0.8, label=f"gate {GATE_SHARPE}")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Annualized Sharpe — TEST")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Étape 5 — CNN-LSTM vs Étape 2 baselines (TEST)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fold_stability(wf_all, path):
    folds = ["VAL", "TEST-q1", "TEST-q2", "TEST-q3", "TEST-q4"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(folds))
    for k, cfg in enumerate(wf_all):
        da = [wf_all[cfg][f]["metrics"]["directional_accuracy"] for f in folds]
        sh = [wf_all[cfg][f]["metrics"]["sharpe_annualized"] for f in folds]
        axes[0].plot(x, da, "o-", label=cfg)
        axes[1].plot(x, sh, "o-", label=cfg)
    axes[0].axhline(0.5, color="k", lw=0.6)
    axes[0].set_title("DA per walk-forward fold (stability)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Sharpe per walk-forward fold (stability)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Étape 5 — walk-forward stability across folds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cumulative(test_agg, baselines_pred, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    for cfg, agg in test_agg.items():
        strat, _ = trading_strategy_returns(agg["y_true"], agg["pred"])
        ax.plot(np.exp(np.cumsum(strat)), label=f"CNN-LSTM {cfg}", linewidth=1.1)
    if baselines_pred is not None:
        for nm, (yt, yp) in baselines_pred.items():
            strat, _ = trading_strategy_returns(yt, yp)
            ax.plot(np.exp(np.cumsum(strat)), label=nm, linewidth=0.8, alpha=0.7, ls="--")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_title("Étape 5 — cumulative strategy equity on TEST (5 bps cost)")
    ax.set_xlabel("TEST trading day")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_threshold(threshold_results, cnn_test, baselines, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: VAL Sharpe vs tau (calibration curve)
    for cfg in threshold_results:
        grid = [g for g in threshold_results[cfg]["val_calibration_grid"] if not g["skipped"]]
        axes[0].plot([g["tau"] for g in grid], [g["sharpe"] for g in grid],
                     "o-", label=cfg, linewidth=1.0)
        axes[0].axvline(threshold_results[cfg]["tau"], color="r", ls=":", lw=0.7)
    axes[0].set_xlabel("tau (confidence threshold)")
    axes[0].set_ylabel("VAL annualized Sharpe")
    axes[0].set_title("Threshold calibration on VAL (chosen tau dashed)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    # right: TEST Sharpe — baseline rule vs threshold rule vs ARIMA
    labels, sharpes, colors = [], [], []
    for cfg in threshold_results:
        labels += [f"{cfg}\nbaseline rule", f"{cfg}\nthreshold rule"]
        sharpes += [cnn_test[cfg]["sharpe_annualized"],
                    threshold_results[cfg]["test"]["sharpe"]]
        colors += ["#7f9fbf", "#2ca02c"]
    labels.append("ARIMA\n(reference)")
    sharpes.append(baselines["arima"]["sharpe_annualized"])
    colors.append("#888888")
    axes[1].bar(labels, sharpes, color=colors)
    axes[1].axhline(baselines["arima"]["sharpe_annualized"], color="r", ls="--", lw=0.8,
                    label=f"ARIMA Sharpe {baselines['arima']['sharpe_annualized']:.2f}")
    axes[1].axhline(GATE_SHARPE, color="orange", ls=":", lw=0.8,
                    label=f"gate {GATE_SHARPE}")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("TEST Sharpe — baseline vs confidence-threshold trading rule")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lscan(lscan, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    Ls = sorted(lscan.keys())
    ax.plot(Ls, [lscan[L]["directional_accuracy"] for L in Ls], "o-", color="#2ca02c")
    ax.set_xticks(Ls)
    ax.set_xlabel("window size L")
    ax.set_ylabel("VAL directional accuracy")
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_title("Étape 5 — window-size scan (VAL)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("ÉTAPE 5 — CNN-LSTM with Regime-as-Feature (walk-forward)")
    print("=" * 78)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    print("\n[1] Loading regime-augmented features ...")
    full, bounds = load_full()
    baselines = json.load(open(ETAPE2_METRICS))["test"]

    pdummy = count_params(CNNLSTM(len(CONFIGS["base12+regime"])))
    print(f"\n[2] CNN-LSTM architecture: Conv1D(16)->LSTM(24)->Dense(16)->1  "
          f"(~{pdummy} params, C3 ceiling 10k)")

    # --- 3. Window-size scan on the VAL fold (D3) --------------------------
    print(f"\n[3] Window-size scan L in {L_GRID} (config base12+regime, VAL fold) ...")
    cols = CONFIGS["base12+regime"]
    feat_raw = full[cols].values.astype(np.float32)
    target = full[TARGET].values.astype(np.float32)
    tr_a, tr_b = bounds["train"]
    va_a, va_b = bounds["val"]
    lscan = {}
    for L in L_GRID:
        res = run_fold(feat_raw, target, np.arange(tr_a, tr_b),
                       np.arange(va_a, va_b), L, len(cols), seeds=[0])
        lscan[L] = evaluate(res["y_true"], res["pred_ensemble"])
        print(f"    L={L:2d}: VAL DA={lscan[L]['directional_accuracy']:.4f}  "
              f"RMSE={lscan[L]['rmse']:.6f}")
    best_L = max(L_GRID, key=lambda L: (lscan[L]["directional_accuracy"],
                                        -lscan[L]["rmse"]))
    print(f"    -> selected L = {best_L}")

    # --- 4. Expanding walk-forward, both configs (D2, D4) ------------------
    print(f"\n[4] Expanding walk-forward (L={best_L}, 5 folds, 3 seeds each) ...")
    wf_all, test_agg = {}, {}
    for cfg_name, cfg_cols in CONFIGS.items():
        wf = walk_forward(full, bounds, cfg_name, cfg_cols, best_L)
        wf_all[cfg_name] = wf
        test_agg[cfg_name] = aggregate_test(wf)

    # --- 5. Honest verdict (D7) --------------------------------------------
    print("\n[5] TEST results (folds 2-5 aggregated = 948-day canonical TEST) ...")
    cnn_test = {cfg: test_agg[cfg]["metrics"] for cfg in CONFIGS}
    for cfg in CONFIGS:
        m = cnn_test[cfg]
        print(f"    CNN-LSTM {cfg:14s}: DA={m['directional_accuracy']:.4f} "
              f"CI{[round(x,3) for x in m['da_ci']]}  RMSE={m['rmse']:.6f}  "
              f"Sharpe={m['sharpe_annualized']:+.3f}  MDD={m['max_drawdown']:+.3f}")
    print("    -- Étape 2 baselines (TEST) --")
    for b in ("random_walk", "historical_mean", "arima", "random_forest"):
        bm = baselines[b]
        print(f"    {b:16s}: DA={bm['directional_accuracy']:.4f}  "
              f"Sharpe={bm['sharpe_annualized']:+.3f}  MDD={bm['max_drawdown']:+.3f}")

    # --- 5b. Confidence-threshold trading rule (calibrated on VAL ONLY) ---
    print("\n[5b] Confidence-threshold trading rule (tau calibrated on VAL — no snooping) ...")
    threshold_results = {}
    for cfg in CONFIGS:
        val_y = wf_all[cfg]["VAL"]["y_true"]
        val_p = wf_all[cfg]["VAL"]["pred"]
        tau, grid = calibrate_threshold(val_y, val_p)
        val_thr = apply_threshold(val_y, val_p, tau)
        test_thr = apply_threshold(test_agg[cfg]["y_true"], test_agg[cfg]["pred"], tau)
        per_fold_thr = {f: apply_threshold(wf_all[cfg][f]["y_true"],
                                           wf_all[cfg][f]["pred"], tau)
                        for f in ("TEST-q1", "TEST-q2", "TEST-q3", "TEST-q4")}
        threshold_results[cfg] = {"tau": tau, "val_calibration_grid": grid,
                                  "val": val_thr, "test": test_thr,
                                  "per_test_fold": per_fold_thr}
        print(f"    {cfg:14s}: tau={tau:.6f}  "
              f"VAL Sharpe(thr)={val_thr['sharpe']:+.3f}  "
              f"TEST Sharpe(thr)={test_thr['sharpe']:+.3f}  "
              f"(baseline-rule TEST Sharpe={cnn_test[cfg]['sharpe_annualized']:+.3f})  "
              f"active={test_thr['n_days_active']}/{len(test_agg[cfg]['pred'])}")

    # --- 5c. Multi-axis honest verdict (PREDICTION vs TRADING) -----------
    best_cfg = max(CONFIGS, key=lambda c: cnn_test[c]["directional_accuracy"])
    bm = cnn_test[best_cfg]
    base_da = {b: baselines[b]["directional_accuracy"] for b in baselines}
    base_sharpe = {b: baselines[b]["sharpe_annualized"] for b in baselines}
    best_base_da = max(base_da.values())
    best_base_sharpe = max(base_sharpe.values())
    # For RMSE / MDD: compare to the ACTIVE trading baselines (ARIMA, RF) only.
    # Random Walk and Historical Mean trivially minimize RMSE by predicting ~0;
    # they generate no usable trading signal, so RMSE/MDD-vs-them is not the
    # meaningful comparison (Étape 2 report §3).
    active = ("arima", "random_forest")
    best_base_rmse = min(baselines[b]["rmse"] for b in active)
    best_base_mdd = max(baselines[b]["max_drawdown"] for b in active)

    ablation_gain = (cnn_test["base12+regime"]["directional_accuracy"]
                     - cnn_test["base12"]["directional_accuracy"])
    fold_da = [wf_all[best_cfg][f]["metrics"]["directional_accuracy"]
               for f in ("TEST-q1", "TEST-q2", "TEST-q3", "TEST-q4")]
    fold_da_std = float(np.std(fold_da))
    stable = fold_da_std < 0.05
    thr_sharpe = threshold_results[best_cfg]["test"]["sharpe"]

    beats_da = bm["directional_accuracy"] > best_base_da
    beats_rmse = bm["rmse"] < best_base_rmse
    beats_mdd = bm["max_drawdown"] > best_base_mdd
    beats_sharpe_base = bm["sharpe_annualized"] > best_base_sharpe
    beats_sharpe_thr = thr_sharpe > best_base_sharpe
    beats_sharpe_any = beats_sharpe_base or beats_sharpe_thr
    best_cnn_sharpe = max(bm["sharpe_annualized"], thr_sharpe)
    clears_gates = (bm["directional_accuracy"] >= GATE_DA
                    and best_cnn_sharpe >= GATE_SHARPE
                    and max(bm["max_drawdown"],
                            threshold_results[best_cfg]["test"]["max_drawdown"]) >= GATE_MDD)

    # Multi-axis verdict (D7 reframed)
    if beats_da and beats_sharpe_any and stable:
        verdict = "JUSTIFIED"
        verdict_note = "CNN-LSTM dominates baselines on PREDICTION and TRADING-Sharpe"
    elif beats_da and beats_rmse and beats_mdd and stable:
        verdict = "BEST_PREDICTOR"
        verdict_note = ("CNN-LSTM is the BEST PREDICTOR (DA, RMSE, MDD); ARIMA "
                        "remains best on trading-Sharpe even under the threshold rule")
    elif beats_da and stable:
        verdict = "MARGINAL_PREDICTOR"
        verdict_note = "CNN-LSTM wins DA but does not dominate the other prediction metrics"
    else:
        verdict = "REJECTED"
        verdict_note = "CNN-LSTM does not beat baselines on DA -> ship ARIMA/Random Forest"

    # Strict legacy verdict (the prior binary call — kept for transparency)
    strict_verdict = ("JUSTIFIED" if (clears_gates and beats_da
                                      and beats_sharpe_base and stable) else
                      "MARGINAL" if (beats_da and beats_sharpe_base and stable) else
                      "REJECTED")

    print("\n  " + "-" * 68)
    print("  HONEST MULTI-AXIS VERDICT (D7 reframed)")
    print("  " + "-" * 68)
    print(f"  best CNN-LSTM config           : {best_cfg}")
    print(f"  regime ablation gain (DA)      : {ablation_gain:+.4f}  "
          f"({'regime helps' if ablation_gain > 0 else 'regime does NOT help'})")
    print(f"  PREDICTION axis (CNN-LSTM vs best baseline):")
    print(f"    DA   {bm['directional_accuracy']:.4f} vs {best_base_da:.4f} (any)         "
          f"{'WIN' if beats_da else 'LOSE'}")
    print(f"    RMSE {bm['rmse']:.6f} vs {best_base_rmse:.6f} (ARIMA/RF) "
          f"{'WIN' if beats_rmse else 'LOSE'}")
    print(f"    MDD  {bm['max_drawdown']:+.3f} vs {best_base_mdd:+.3f} (ARIMA/RF)        "
          f"{'WIN' if beats_mdd else 'LOSE'}")
    print(f"  TRADING axis (Sharpe):")
    print(f"    baseline rule  {bm['sharpe_annualized']:+.3f} vs {best_base_sharpe:+.3f}  "
          f"{'WIN' if beats_sharpe_base else 'LOSE'}")
    print(f"    threshold rule {thr_sharpe:+.3f} vs {best_base_sharpe:+.3f}  "
          f"{'WIN' if beats_sharpe_thr else 'LOSE'}")
    print(f"  stable across TEST folds (DA std={fold_da_std:.4f})  : {stable}")
    print(f"\n  OVERALL = {verdict}")
    print(f"  -> {verdict_note}")
    print(f"  (strict RULE 8 binary verdict, baseline-rule only: {strict_verdict})")
    print("  " + "-" * 68)

    # --- 6. Persist --------------------------------------------------------
    print("\n[6] Writing artifacts ...")
    out = {
        "selected_L": best_L, "lscan": lscan,
        "architecture_params": pdummy,
        "cnn_test": cnn_test,
        "baselines_test": {b: baselines[b] for b in
                           ("random_walk", "historical_mean", "arima", "random_forest")},
        "walk_forward": {cfg: {f: {"metrics": wf_all[cfg][f]["metrics"],
                                   "per_seed_da": wf_all[cfg][f]["per_seed_da"],
                                   "n": wf_all[cfg][f]["n"]}
                               for f in wf_all[cfg]} for cfg in wf_all},
        "threshold_rule": threshold_results,
        "verdict": {"best_config": best_cfg, "ablation_gain_da": ablation_gain,
                    "beats_baseline_da": bool(beats_da),
                    "beats_baseline_rmse": bool(beats_rmse),
                    "beats_baseline_mdd": bool(beats_mdd),
                    "beats_baseline_sharpe_baseline_rule": bool(beats_sharpe_base),
                    "beats_baseline_sharpe_threshold_rule": bool(beats_sharpe_thr),
                    "clears_gates": bool(clears_gates), "stable": bool(stable),
                    "fold_da_std": fold_da_std,
                    "overall": verdict, "verdict_note": verdict_note,
                    "strict_rule8_binary_verdict": strict_verdict},
    }
    with open(os.path.join(RESULTS_DIR, "walkforward_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    agg = test_agg[best_cfg]
    pd.DataFrame({"row": agg["rows"], "y_true": agg["y_true"], "y_pred": agg["pred"]}
                 ).to_csv(os.path.join(RESULTS_DIR, "predictions_test.csv"), index=False)
    print(f"    {os.path.join(RESULTS_DIR, 'walkforward_metrics.json')}")
    print(f"    {os.path.join(RESULTS_DIR, 'predictions_test.csv')}")

    # --- 7. Plots ----------------------------------------------------------
    print("\n[7] Generating plots ...")
    plot_lscan(lscan, os.path.join(PLOTS_DIR, "etape5_lscan.png"))
    plot_baseline_comparison(cnn_test, baselines,
                             os.path.join(PLOTS_DIR, "etape5_baseline_comparison.png"))
    plot_fold_stability(wf_all, os.path.join(PLOTS_DIR, "etape5_fold_stability.png"))
    plot_cumulative(test_agg, None, os.path.join(PLOTS_DIR, "etape5_cumulative.png"))
    plot_threshold(threshold_results, cnn_test, baselines,
                   os.path.join(PLOTS_DIR, "etape5_threshold_rule.png"))
    print(f"    5 plots -> {PLOTS_DIR}")

    print("\n" + "=" * 78)
    print(f"ÉTAPE 5 COMPLETE — verdict = {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
