"""
================================================================================
ÉTAPE 6 — Final Backtest with Deflated Sharpe Ratio
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Final, mémoire-ready backtest of all candidate models on the 948-day canonical
  TEST window, with realistic Moroccan transaction costs, the Deflated Sharpe
  Ratio (Bailey & López de Prado 2014), regime-conditional decomposition, and a
  cost-sensitivity analysis.

INPUT
  outputs/etape2/predictions_test.csv      (baselines)
  outputs/etape5/predictions_test.csv      (CNN-LSTM base12)
  outputs/etape4/regimes/masi_regimes_test.csv     (regime labels)

OUTPUT
  outputs/etape6/backtest_metrics.json     (full per-model metrics)
  outputs/etape6/equity_curves.csv         (per-day equity per model)
  reports/figures/etape6/*.png                       (5 diagnostic plots)

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5)
--------------------------------------------------------------------------------
D1  Models backtested (5):  Random Walk, Historical Mean, ARIMA(2,0,2),
    Random Forest, CNN-LSTM `base12`.  The HMM regime feature was dropped after
    the Étape 5 ablation (−0.84 pp DA) — it is used only as a CONDITIONING
    variable for the regime-conditional reporting.

D2  Trading rule:  position_t = sign(prediction_t) ∈ {−1, 0, +1}.
    Strategy return_t = position_t · y_true_t  −  cost_bps if position changes.
    Primary cost = 5 bps one-way (10 bps round-trip), MASI-realistic per Touzani
    & Douzi (2021) and Deep et al. (2025).  Sensitivity at 5/10/20 bps.

D3  Deflated Sharpe Ratio (Bailey & López de Prado 2014).
    Standard Sharpe is biased upward when many strategies have been screened.
    DSR = PSR(SR0) where SR0 is the expected maximum Sharpe under the null given
    N trials, and PSR adjusts for the non-normality of returns:
        SR0  = sqrt(V) · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]
        PSR(x) = Φ( (SR̂ − x)·sqrt(T−1) / sqrt(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) )
    γ ≈ 0.5772 (Euler-Mascheroni), V = variance of the N trial Sharpes,
    γ₃ = skew, γ₄ = Pearson kurtosis of the strategy daily returns.
    DSR ≥ 0.95 is a typical "publishable" threshold; DSR ≥ 0.99 = very strong.

D4  Regime-conditional decomposition. The Étape 4 (causal) regime labels are
    joined day-by-day. Each model's Sharpe + DA is reported per regime
    (Bear / Neutral / Bull) — answers Deep 2025's "regime-dependent performance"
    requirement and exposes whether a model only works in one regime.

D5  Honest reporting (RULE 8 spirit, user instruction). DSR < 0.95 is flagged;
    cost-sensitivity failures are flagged. Final recommendation grounded in the
    full evidence, not the single Sharpe point estimate.

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE
--------------------------------------------------------------------------------
  L5  Signal at t → return realised at t+1 (target = y_t = ln(P_{t+1}/P_t)).  OK
  L1/L2/L3/L4/L6/L8 — all inherited; predictions on disk are leakage-free.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, norm

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E2_PRED = os.path.join(PROJECT_ROOT, "outputs", "etape2", "predictions_test.csv")
E5_PRED = os.path.join(PROJECT_ROOT, "outputs", "etape5", "predictions_test.csv")
E4_REG = os.path.join(PROJECT_ROOT, "outputs", "etape4", "regimes", "masi_regimes_test.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape6")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape6")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRADING_DAYS = 252
PRIMARY_COST_BPS = 5
SENSITIVITY_COSTS_BPS = [5, 10, 20]      # 10-40 bps round-trip range
EULER_M = 0.5772156649

MODELS = ["random_walk", "historical_mean", "arima", "random_forest", "cnn_lstm_base12"]
MODEL_LABELS = {"random_walk": "Random Walk", "historical_mean": "Hist. Mean",
                "arima": "ARIMA(2,0,2)", "random_forest": "Random Forest",
                "cnn_lstm_base12": "CNN-LSTM base12"}
MODEL_COLORS = {"random_walk": "#888888", "historical_mean": "#bcbd22",
                "arima": "#1f77b4", "random_forest": "#2ca02c",
                "cnn_lstm_base12": "#d62728"}
REGIME_NAMES = ["Bear", "Neutral", "Bull"]
REGIME_COLORS = {"Bear": "#d62728", "Neutral": "#bdbd62", "Bull": "#2ca02c"}


# =============================================================================
# 1. DATA LOADING & ALIGNMENT
# =============================================================================

def load_predictions() -> pd.DataFrame:
    """Align baselines + CNN-LSTM + regime by date on the canonical TEST window."""
    e2 = pd.read_csv(E2_PRED)
    e2["date"] = pd.to_datetime(e2["date"])
    e2 = e2.set_index("date").sort_index().rename(columns={"actual": "y_true"})

    reg = pd.read_csv(E4_REG)
    reg["date"] = pd.to_datetime(reg["date"])
    reg = reg.set_index("date").sort_index()

    # Étape 5 predictions use a full-timeline row index (TRAIN+VAL+TEST). The
    # TEST block starts at row 3775 (= len(TRAIN)+len(VAL)). Map to dates via
    # the regime file (sorted TEST dates in order).
    e5 = pd.read_csv(E5_PRED)
    test_dates = reg.index.values
    base_row = 3775
    e5["date"] = pd.to_datetime([test_dates[r - base_row] for r in e5["row"]])
    e5 = e5.set_index("date").sort_index().rename(columns={"y_pred": "cnn_lstm_base12"})

    df = e2.join(e5[["cnn_lstm_base12"]], how="left")
    df = df.join(reg[["regime", "regime_name"]], how="left")
    df = df.dropna(subset=["y_true"] + MODELS + ["regime"])
    print(f"  aligned TEST frame: {len(df)} rows  "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return df


# =============================================================================
# 2. CORE BACKTEST METRICS
# =============================================================================

def strategy_returns(y: np.ndarray, pred: np.ndarray, cost_bps: float
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
    pos = np.sign(pred)
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(pos - prev) * (cost_bps / 1e4)
    return pos * y - cost, pos, int((pos != prev).sum())


def equity_curve(strat: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(strat))


def max_drawdown(strat: np.ndarray) -> float:
    cum = equity_curve(strat)
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def backtest_metrics(y: np.ndarray, pred: np.ndarray, cost_bps: float) -> dict:
    strat, pos, ntr = strategy_returns(y, pred, cost_bps)
    mean_d, std_d = float(strat.mean()), float(strat.std())
    sh = mean_d / std_d * np.sqrt(TRADING_DAYS) if std_d > 0 else 0.0
    neg = strat[strat < 0]
    down_std = float(neg.std()) if len(neg) else 0.0
    sortino = mean_d / down_std * np.sqrt(TRADING_DAYS) if down_std > 0 else 0.0
    mdd = max_drawdown(strat)
    ann_ret = mean_d * TRADING_DAYS
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    mask = y != 0
    da = float((np.sign(pred[mask]) == np.sign(y[mask])).mean()) if mask.sum() else float("nan")
    turnover = float(np.abs(pos[1:] - pos[:-1]).mean()) if len(pos) > 1 else 0.0
    return {
        "ann_return": ann_ret, "ann_vol": std_d * np.sqrt(TRADING_DAYS),
        "sharpe": sh, "sortino": sortino,
        "max_drawdown": mdd, "calmar": calmar,
        "directional_accuracy": da,
        "n_trades": ntr, "turnover": turnover,
        "best_day": float(strat.max()), "worst_day": float(strat.min()),
        "skew_strat": float(skew(strat)),
        "kurt_strat_pearson": float(kurtosis(strat, fisher=False)),
        "final_equity": float(equity_curve(strat)[-1]),
        "strategy_returns": strat,        # kept for plotting / DSR
    }


# =============================================================================
# 3. DEFLATED SHARPE RATIO (Bailey & López de Prado 2014)
# =============================================================================

def psr(sr_hat: float, sr_threshold: float, T: int,
        skew_r: float, kurt_p: float) -> float:
    """Probabilistic Sharpe Ratio at a threshold (daily-scale SR)."""
    denom_sq = max(1e-12, 1.0 - skew_r * sr_hat + ((kurt_p - 1.0) / 4.0) * sr_hat ** 2)
    z = (sr_hat - sr_threshold) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe(strat: np.ndarray, all_daily_sharpes: List[float]) -> dict:
    """Deflated Sharpe = PSR(SR0) with SR0 from the trial-Sharpe variance."""
    T = len(strat)
    sr_d = strat.mean() / strat.std() if strat.std() > 0 else 0.0
    sk = float(skew(strat))
    kp = float(kurtosis(strat, fisher=False))
    N = len(all_daily_sharpes)
    if N > 1:
        V = float(np.var(all_daily_sharpes, ddof=1))
        sr0 = float(np.sqrt(V) * ((1.0 - EULER_M) * norm.ppf(1.0 - 1.0 / N)
                                  + EULER_M * norm.ppf(1.0 - 1.0 / (N * math.e))))
    else:
        V, sr0 = 0.0, 0.0
    return {"sr_daily": sr_d, "sr0_threshold": sr0, "v_trial_sharpes": V,
            "n_trials": N, "skew_strat": sk, "kurt_pearson": kp,
            "deflated_sharpe_psr": psr(sr_d, sr0, T, sk, kp),
            "psr_vs_zero": psr(sr_d, 0.0, T, sk, kp)}


# =============================================================================
# 4. REGIME-CONDITIONAL DECOMPOSITION
# =============================================================================

def regime_conditional(y: np.ndarray, pred: np.ndarray, regime: np.ndarray,
                       cost_bps: float) -> dict:
    """Per-regime Sharpe and DA (cost applied; small partitions noted)."""
    out = {}
    for i, nm in enumerate(REGIME_NAMES):
        m = regime == i
        if m.sum() < 5:
            out[nm] = {"n": int(m.sum()), "sharpe": None, "da": None,
                       "mean_return_ann": None}
            continue
        strat, _, _ = strategy_returns(y[m], pred[m], cost_bps)
        sh = float(strat.mean() / strat.std() * np.sqrt(TRADING_DAYS)) if strat.std() > 0 else 0.0
        nz = y[m] != 0
        da = (float((np.sign(pred[m][nz]) == np.sign(y[m][nz])).mean())
              if nz.sum() else float("nan"))
        out[nm] = {"n": int(m.sum()), "sharpe": sh, "da": da,
                   "mean_return_ann": float(strat.mean() * TRADING_DAYS)}
    return out


# =============================================================================
# 5. PLOTS
# =============================================================================

def plot_equity_curves(eq: Dict[str, np.ndarray], dates: pd.DatetimeIndex, path: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    for m, curve in eq.items():
        ax.plot(dates, curve, label=MODEL_LABELS[m], color=MODEL_COLORS[m], linewidth=1.1)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_title("Étape 6 — Cumulative equity on TEST (5 bps one-way cost)")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drawdowns(eq: Dict[str, np.ndarray], dates: pd.DatetimeIndex, path: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for m, curve in eq.items():
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak
        ax.plot(dates, dd, label=MODEL_LABELS[m], color=MODEL_COLORS[m], linewidth=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Étape 6 — Drawdowns on TEST")
    ax.set_ylabel("drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_regime_heatmap(regime_table: Dict[str, Dict[str, dict]], path: str):
    M = [m for m in MODELS]
    rows = []
    for m in M:
        rows.append([(regime_table[m][r]["sharpe"] if regime_table[m][r]["sharpe"] is not None
                      else float("nan")) for r in REGIME_NAMES])
    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(REGIME_NAMES)))
    ax.set_xticklabels(REGIME_NAMES)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in M])
    for i in range(len(M)):
        for j in range(len(REGIME_NAMES)):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 1.2 else "black", fontsize=9)
    ax.set_title("Étape 6 — Sharpe per (model × regime)  [TEST, 5 bps]")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cost_sensitivity(cost_sens: Dict[int, Dict[str, float]], path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    costs = sorted(cost_sens.keys())
    for m in MODELS:
        ax.plot(costs, [cost_sens[c][m] for c in costs], "o-",
                label=MODEL_LABELS[m], color=MODEL_COLORS[m], linewidth=1.1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(costs)
    ax.set_xlabel("one-way transaction cost (bps)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Étape 6 — Sharpe vs cost (robustness)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_final_summary(metrics: dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sh = [metrics[m]["sharpe"] for m in MODELS]
    dsr = [metrics[m]["deflated"]["deflated_sharpe_psr"] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]
    axes[0].bar([MODEL_LABELS[m] for m in MODELS], sh, color=colors)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Sharpe annualized (TEST, 5 bps)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar([MODEL_LABELS[m] for m in MODELS], dsr, color=colors)
    axes[1].axhline(0.95, color="r", ls="--", lw=0.8, label="DSR 0.95 (publishable)")
    axes[1].axhline(0.99, color="darkred", ls=":", lw=0.8, label="DSR 0.99 (strong)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Deflated Sharpe Ratio = PSR(SR0)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("ÉTAPE 6 — FINAL BACKTEST (Deflated Sharpe + regime-conditional)")
    print("=" * 78)

    print("\n[1] Loading & aligning predictions ...")
    df = load_predictions()
    y = df["y_true"].values
    regime = df["regime"].values.astype(int)
    dates = df.index

    # --- 2. Per-model backtest at primary cost ---------------------------
    print(f"\n[2] Backtesting {len(MODELS)} models at primary cost = {PRIMARY_COST_BPS} bps ...")
    metrics = {}
    for m in MODELS:
        pred = df[m].values
        metrics[m] = backtest_metrics(y, pred, PRIMARY_COST_BPS)
        em = metrics[m]
        print(f"    {MODEL_LABELS[m]:18s}  DA={em['directional_accuracy']:.4f}  "
              f"Sharpe={em['sharpe']:+.3f}  Sortino={em['sortino']:+.3f}  "
              f"MDD={em['max_drawdown']:+.3f}  Calmar={em['calmar']:+.2f}  "
              f"equity={em['final_equity']:.3f}")

    # --- 3. Deflated Sharpe Ratio (Bailey & López de Prado 2014) ---------
    print(f"\n[3] Deflated Sharpe (N={len(MODELS)} trials) ...")
    all_daily_sharpes = []
    for m in MODELS:
        s = metrics[m]["strategy_returns"]
        all_daily_sharpes.append(float(s.mean() / s.std()) if s.std() > 0 else 0.0)
    for m in MODELS:
        d = deflated_sharpe(metrics[m]["strategy_returns"], all_daily_sharpes)
        metrics[m]["deflated"] = d
        flag = ("STRONG" if d["deflated_sharpe_psr"] >= 0.99 else
                "PUBLISHABLE" if d["deflated_sharpe_psr"] >= 0.95 else
                "WEAK")
        print(f"    {MODEL_LABELS[m]:18s}  SR_daily={d['sr_daily']:+.4f}  "
              f"SR0={d['sr0_threshold']:+.4f}  DSR={d['deflated_sharpe_psr']:.4f}  [{flag}]")

    # --- 4. Regime-conditional ------------------------------------------
    print(f"\n[4] Regime-conditional decomposition (Bear / Neutral / Bull) ...")
    regime_table = {}
    for m in MODELS:
        rc = regime_conditional(y, df[m].values, regime, PRIMARY_COST_BPS)
        regime_table[m] = rc
        line = f"    {MODEL_LABELS[m]:18s}  "
        for r in REGIME_NAMES:
            sh = rc[r]["sharpe"]
            sh_str = f"{sh:+.2f}" if sh is not None else "  nan"
            line += f"{r}: Sh={sh_str}  "
        print(line)

    # --- 5. Cost sensitivity --------------------------------------------
    print(f"\n[5] Cost sensitivity (one-way bps in {SENSITIVITY_COSTS_BPS}) ...")
    cost_sens = {}
    for cb in SENSITIVITY_COSTS_BPS:
        cost_sens[cb] = {m: backtest_metrics(y, df[m].values, cb)["sharpe"] for m in MODELS}
        line = f"    {cb} bps:  " + "  ".join(
            f"{MODEL_LABELS[m]}={cost_sens[cb][m]:+.2f}" for m in MODELS)
        print(line)

    # --- 6. Final verdict / ranking -------------------------------------
    print(f"\n[6] Ranking on TEST (primary cost {PRIMARY_COST_BPS} bps) ...")
    rank_sharpe = sorted(MODELS, key=lambda m: metrics[m]["sharpe"], reverse=True)
    rank_dsr = sorted(MODELS, key=lambda m: metrics[m]["deflated"]["deflated_sharpe_psr"],
                      reverse=True)
    rank_da = sorted(MODELS, key=lambda m: metrics[m]["directional_accuracy"], reverse=True)
    print(f"    Sharpe  : {' > '.join(MODEL_LABELS[m] for m in rank_sharpe)}")
    print(f"    DSR     : {' > '.join(MODEL_LABELS[m] for m in rank_dsr)}")
    print(f"    DA      : {' > '.join(MODEL_LABELS[m] for m in rank_da)}")

    publishable = [m for m in MODELS
                   if metrics[m]["deflated"]["deflated_sharpe_psr"] >= 0.95]
    print(f"\n    Models with DSR >= 0.95 (publishable): "
          f"{[MODEL_LABELS[m] for m in publishable] or '(none)'}")

    # --- 7. Persist ------------------------------------------------------
    print(f"\n[7] Writing artifacts ...")
    out = {
        "primary_cost_bps": PRIMARY_COST_BPS,
        "n_test_days": int(len(df)),
        "test_range": [str(dates.min().date()), str(dates.max().date())],
        "models": {m: {k: v for k, v in metrics[m].items() if k != "strategy_returns"}
                   for m in MODELS},
        "regime_conditional": regime_table,
        "cost_sensitivity": cost_sens,
        "rankings": {"by_sharpe": rank_sharpe, "by_dsr": rank_dsr, "by_da": rank_da},
        "publishable_models_dsr_0p95": publishable,
    }
    with open(os.path.join(RESULTS_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    eq = {m: equity_curve(metrics[m]["strategy_returns"]) for m in MODELS}
    eq_df = pd.DataFrame({MODEL_LABELS[m]: eq[m] for m in MODELS}, index=dates)
    eq_df.to_csv(os.path.join(RESULTS_DIR, "equity_curves.csv"))
    print(f"    {os.path.join(RESULTS_DIR, 'backtest_metrics.json')}")
    print(f"    {os.path.join(RESULTS_DIR, 'equity_curves.csv')}")

    # --- 8. Plots --------------------------------------------------------
    print(f"\n[8] Generating diagnostic plots ...")
    plot_equity_curves(eq, dates, os.path.join(PLOTS_DIR, "etape6_equity_curves.png"))
    plot_drawdowns(eq, dates, os.path.join(PLOTS_DIR, "etape6_drawdowns.png"))
    plot_regime_heatmap(regime_table, os.path.join(PLOTS_DIR, "etape6_regime_heatmap.png"))
    plot_cost_sensitivity(cost_sens, os.path.join(PLOTS_DIR, "etape6_cost_sensitivity.png"))
    plot_final_summary(metrics, os.path.join(PLOTS_DIR, "etape6_final_summary.png"))
    print(f"    5 plots -> {PLOTS_DIR}")

    print("\n" + "=" * 78)
    top_dsr = rank_dsr[0]
    print(f"ÉTAPE 6 COMPLETE — top DSR = {MODEL_LABELS[top_dsr]} "
          f"(DSR={metrics[top_dsr]['deflated']['deflated_sharpe_psr']:.4f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
