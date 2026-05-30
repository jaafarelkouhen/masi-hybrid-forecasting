"""
================================================================================
ÉTAPE 9 — Robustesse (sous-périodes, coûts, seuils HMM, JKM, stress proxy)
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Tester la robustesse des 7 stratégies de l'étape 8 selon 5 axes :
    A. Sous-périodes : TEST coupé en deux (~474 jours chacun)
    B. Cost sensitivity : 5 / 10 / 20 bps (alignement étape 6)
    C. Seuils HMM dynamiques : gate sur P(regime) > T pour différents T
    D. Jobson-Korkie-Memmel (1981, 2003) : test pairwise différence de Sharpe
    E. Stress proxy alternatif : strat 7 avec risk_regime=high au lieu de HMM=Neutral

INPUT
  outputs/etape6/etape6_final_predictions.csv         (canonique)
  outputs/etape7/risk_metrics_test.csv    (VaR, ES, vol_garch, risk_regime)
  outputs/etape4/regimes/masi_regimes_test.csv (regime_prob_*)

OUTPUT
  outputs/etape9/robustness_metrics.json
  outputs/etape9/subperiod_metrics.csv
  outputs/etape9/cost_sensitivity.csv
  outputs/etape9/dynamic_hmm_sweep.csv
  outputs/etape9/jkm_pvalue_matrix.csv
  reports/figures/etape9/etape9_subperiod_sharpe.png
  reports/figures/etape9/etape9_cost_sensitivity.png
  reports/figures/etape9/etape9_dynamic_hmm_threshold.png
  reports/figures/etape9/etape9_jkm_heatmap.png
  reports/figures/etape9/etape9_robustness_scorecard.png

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5)
--------------------------------------------------------------------------------
D1  Sous-périodes : split au midpoint TEST (jour 474 = ~2024-06).
    Justification : équilibre entre les deux moitiés pour comparaison fair.
    Pas de découpage par régime macro (qui serait data-snooping).

D2  Cost sensitivity : on conserve EXACTEMENT le même setup étape 8 :
    coût one-way proportionnel au turnover |Δposition|, et on ne change que
    la valeur `COST_DEC`. Permet comparaison directe étape 6 § 4.

D3  Seuils HMM dynamiques : utilise les `regime_prob_*` causales étape 4.
    Signal actif si max(p_bear, p_bull) > T. T ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.
    T=0.5 ≈ argmax classique. T=0.7+ = exigence de conviction forte.

D4  Jobson-Korkie-Memmel (JKM) :
      Δ = SR_A − SR_B  (Sharpe daily)
      SE² = (1/T) · [2(1−ρ) + 0.5·(SR_A² + SR_B² − 2·SR_A·SR_B·ρ²)]
      z = Δ / SE       ~ N(0,1)  sous H0 : SR_A = SR_B
    Corrige le DM étape 8 qui testait les MOYENNES (pas les ratios Sharpe).

D5  Stress proxy alternatif : strat 7 réplique avec stress = (risk_regime=high)
    au lieu de (HMM=Neutral). Si la performance est similaire → robuste au
    choix du proxy.

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE  (L1-L8)
--------------------------------------------------------------------------------
L1  Sous-périodes : pas de réoptimisation de paramètres entre P1 et P2.
L2  Régimes HMM consommés étape 4 v2 (causaux).
L3  Rolling causals depuis étape 7/8 (inchangés).
L4  y_true jamais utilisé pour décider position.
L5  Position t → strategy_return t (= y_true_t = ln(P_{t+1}/P_t)).
L6/L7  inhérents.
L8  VaR/GARCH déjà fit TRAIN-only.
================================================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTES (alignées étape 8)
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "outputs" / "etape9"
PLOTS_DIR = ROOT / "reports" / "figures" / "etape9"
RES_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

PPY = 252
COST_BPS_PRIMARY = 5.0
COST_LEVELS = [5.0, 10.0, 20.0]
COST_DEC_PRIMARY = COST_BPS_PRIMARY / 10_000

ROLL_BUDGET = 60
B_STD = 0.01
Q_STRESS = 0.30
Q_NORMAL = 0.70
EPS_VAR = 1e-6

HMM_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# ============================================================================
# CHARGEMENT
# ============================================================================
def load_inputs():
    risk = pd.read_csv(ROOT / "outputs" / "etape7" / "risk_metrics_test.csv",
                       parse_dates=["date"])
    regs = pd.read_csv(ROOT / "outputs" / "etape4" / "regimes" / "masi_regimes_test.csv",
                       parse_dates=["date"])
    # Join regime probas
    df = risk.merge(
        regs[["date", "regime_prob_bear", "regime_prob_neutral", "regime_prob_bull"]],
        on="date", how="left"
    )
    assert len(df) == 948, f"Attendu 948 jours, reçu {len(df)}"
    return df


# ============================================================================
# STRATEGY RETURNS
# ============================================================================
def strat_returns_binary(positions, y_true, cost_dec):
    positions = positions.astype(float)
    prev = np.concatenate([[0.0], positions[:-1]])
    c = cost_dec * np.abs(positions - prev)
    return positions * y_true - c


def strat_returns_continuous(positions, y_true, cost_dec):
    positions = positions.astype(float)
    prev = np.concatenate([[0.0], positions[:-1]])
    c = cost_dec * np.abs(positions - prev)
    return positions * y_true - c


# ============================================================================
# CONSTRUCTION 7 STRATÉGIES + 1 VARIANTE STRESS PROXY
# ============================================================================
def build_all_strategies(df: pd.DataFrame, cost_dec: float = COST_DEC_PRIMARY) -> dict:
    """Construit 7 stratégies étape 8 + 1 variante stress proxy = risk_regime."""
    y_true = df["actual_return"].values
    y_pred = df["predicted_return"].values
    signal = np.sign(y_pred).astype(int)
    regime = df["regime_name"].values
    risk_reg = df["risk_regime"].values
    var_abs = np.abs(df["var_param_5"].values)

    out = {}

    # 1. Buy & Hold
    pos = np.ones(len(df))
    out["1_buy_hold"] = {"label": "Buy & Hold", "mode": "binary",
                          "position": pos,
                          "strategy_return": strat_returns_binary(pos, y_true, cost_dec)}

    # 2. CNN-LSTM nu
    pos = signal.astype(float)
    out["2_cnn_lstm_nu"] = {"label": "CNN-LSTM nu", "mode": "binary",
                             "position": pos,
                             "strategy_return": strat_returns_binary(pos, y_true, cost_dec)}

    # 3. CNN-LSTM + HMM-gate
    pos = np.where(np.isin(regime, ["Bear", "Bull"]), signal, 0).astype(float)
    out["3_cnn_lstm_hmm_gate"] = {"label": "CNN-LSTM + HMM-gate", "mode": "binary",
                                    "position": pos,
                                    "strategy_return": strat_returns_binary(pos, y_true, cost_dec)}

    # 4. CNN-LSTM + risk-gate
    pos = np.where(risk_reg != "high", signal, 0).astype(float)
    out["4_cnn_lstm_risk_gate"] = {"label": "CNN-LSTM + risk-gate", "mode": "binary",
                                     "position": pos,
                                     "strategy_return": strat_returns_binary(pos, y_true, cost_dec)}

    # 5. CNN-LSTM + HMM + risk
    pos = np.where(
        np.isin(regime, ["Bear", "Bull"]) & (risk_reg != "high"),
        signal, 0
    ).astype(float)
    out["5_cnn_lstm_hmm_risk"] = {"label": "CNN-LSTM + HMM + risk-gate", "mode": "binary",
                                    "position": pos,
                                    "strategy_return": strat_returns_binary(pos, y_true, cost_dec)}

    # 6. VaR-budget (B fixe)
    w_raw = np.minimum(1.0, B_STD / np.maximum(var_abs, EPS_VAR))
    pos = signal.astype(float) * w_raw
    out["6_cnn_lstm_var_budget"] = {"label": "CNN-LSTM × VaR-budget", "mode": "continuous",
                                      "position": pos,
                                      "strategy_return": strat_returns_continuous(pos, y_true, cost_dec)}

    # 7. HMM-conditional budget (stress = HMM Neutral)
    var_series = pd.Series(var_abs)
    shifted = var_series.shift(1)
    q_stress = shifted.rolling(ROLL_BUDGET, min_periods=ROLL_BUDGET).quantile(Q_STRESS)
    q_normal = shifted.rolling(ROLL_BUDGET, min_periods=ROLL_BUDGET).quantile(Q_NORMAL)

    is_stress_hmm = (regime == "Neutral")
    B_t = np.where(is_stress_hmm, q_stress.values, q_normal.values)
    B_t = np.where(np.isnan(B_t), B_STD, B_t)
    w_t = np.minimum(1.0, B_t / np.maximum(var_abs, EPS_VAR))
    pos = signal.astype(float) * w_t
    out["7_cnn_lstm_hmm_budget"] = {"label": "CNN-LSTM × HMM-conditional budget",
                                      "mode": "continuous",
                                      "position": pos,
                                      "strategy_return": strat_returns_continuous(pos, y_true, cost_dec)}

    # 7-bis. HMM-cond budget BUT stress = risk_regime=high (variante axe E)
    is_stress_risk = (risk_reg == "high")
    B_t_alt = np.where(is_stress_risk, q_stress.values, q_normal.values)
    B_t_alt = np.where(np.isnan(B_t_alt), B_STD, B_t_alt)
    w_t_alt = np.minimum(1.0, B_t_alt / np.maximum(var_abs, EPS_VAR))
    pos_alt = signal.astype(float) * w_t_alt
    out["7b_cnn_lstm_riskreg_budget"] = {
        "label": "CNN-LSTM × risk-regime-conditional budget (variante)",
        "mode": "continuous",
        "position": pos_alt,
        "strategy_return": strat_returns_continuous(pos_alt, y_true, cost_dec),
    }

    return out


# ============================================================================
# METRICS (compact, focus Sharpe/MDD pour robustesse)
# ============================================================================
def equity_from_log_returns(r):
    return np.exp(np.cumsum(np.asarray(r, dtype=float)))


def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def sharpe_ann(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def sharpe_daily(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return float(r.mean() / sd) if sd > 0 else 0.0


def compute_compact_metrics(strat_ret, positions, mode):
    r = np.asarray(strat_ret, dtype=float)
    p = np.asarray(positions, dtype=float)
    eq = equity_from_log_returns(r)
    sr = sharpe_ann(r)
    mdd = max_drawdown(eq)
    mean_d = float(r.mean())
    ann_ret = float(np.exp(mean_d * PPY) - 1.0)
    calmar = float(ann_ret / abs(mdd)) if mdd != 0 else 0.0
    prev = np.concatenate([[0.0], p[:-1]])
    if mode == "binary":
        n_trades = int((p != prev).sum())
    else:
        n_trades = int((np.abs(p - prev) > 1e-9).sum())
    return {"sharpe": sr, "mdd": mdd, "calmar": calmar,
            "final_equity": float(eq[-1]), "n_trades": n_trades,
            "ann_return": ann_ret}


# ============================================================================
# A. SOUS-PÉRIODES
# ============================================================================
def subperiod_analysis(df: pd.DataFrame, strategies: dict, mid: int = None) -> pd.DataFrame:
    """Découpe TEST en 2 sous-périodes ~474j chacune."""
    if mid is None:
        mid = len(df) // 2
    rows = []
    for k, s in strategies.items():
        for label, sl in [("FULL", slice(None)),
                          ("P1 (2022-06→2024-06)", slice(0, mid)),
                          ("P2 (2024-06→2026-04)", slice(mid, None))]:
            r = s["strategy_return"][sl]
            p = s["position"][sl]
            m = compute_compact_metrics(r, p, s["mode"])
            rows.append({
                "strategy_id": k,
                "label": s["label"],
                "period": label,
                "n_days": len(r),
                **m,
            })
    return pd.DataFrame(rows)


# ============================================================================
# B. COST SENSITIVITY
# ============================================================================
def cost_sensitivity(df: pd.DataFrame, cost_levels=COST_LEVELS) -> pd.DataFrame:
    rows = []
    for cb in cost_levels:
        strategies = build_all_strategies(df, cost_dec=cb / 10_000)
        for k, s in strategies.items():
            m = compute_compact_metrics(s["strategy_return"], s["position"], s["mode"])
            rows.append({
                "strategy_id": k,
                "label": s["label"],
                "cost_bps": cb,
                **m,
            })
    return pd.DataFrame(rows)


# ============================================================================
# C. SEUILS HMM DYNAMIQUES
# ============================================================================
def dynamic_hmm_threshold_sweep(df: pd.DataFrame, thresholds=HMM_THRESHOLDS) -> pd.DataFrame:
    """Pour chaque threshold T, signal actif si max(p_bear, p_bull) > T."""
    y_true = df["actual_return"].values
    y_pred = df["predicted_return"].values
    signal = np.sign(y_pred).astype(int)
    p_bear = df["regime_prob_bear"].values
    p_bull = df["regime_prob_bull"].values
    max_directional = np.maximum(p_bear, p_bull)

    rows = []
    for T in thresholds:
        active = max_directional > T
        pos = np.where(active, signal, 0).astype(float)
        r = strat_returns_binary(pos, y_true, COST_DEC_PRIMARY)
        m = compute_compact_metrics(r, pos, mode="binary")
        rows.append({
            "threshold": T,
            "pct_days_active": float(active.mean()),
            **m,
        })
    return pd.DataFrame(rows)


# ============================================================================
# D. JOBSON-KORKIE-MEMMEL (1981, Memmel 2003)
# ============================================================================
def jkm_test(r_a: np.ndarray, r_b: np.ndarray) -> dict:
    """
    Test différence de Sharpe pairwise.
      Δ = SR_A − SR_B  (Sharpe daily)
      SE² = (1/T) · [2(1−ρ) + 0.5·(SR_A² + SR_B² − 2·SR_A·SR_B·ρ²)]
      z = Δ / SE  ~ N(0,1) sous H0 : SR_A = SR_B
    """
    a = np.asarray(r_a, dtype=float)
    b = np.asarray(r_b, dtype=float)
    T = len(a)
    if T < 30: return {"sr_a": None, "sr_b": None, "diff": None,
                      "se": None, "z": None, "pvalue": None}

    sd_a = a.std()
    sd_b = b.std()
    if sd_a == 0 or sd_b == 0:
        return {"sr_a": None, "sr_b": None, "diff": None,
                "se": None, "z": None, "pvalue": None}

    sr_a = a.mean() / sd_a
    sr_b = b.mean() / sd_b
    rho = float(np.corrcoef(a, b)[0, 1]) if T > 1 else 0.0

    se_sq = (1.0 / T) * (2.0 * (1.0 - rho)
                          + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2.0 * sr_a * sr_b * rho ** 2))
    if se_sq <= 0:
        return {"sr_a": float(sr_a), "sr_b": float(sr_b),
                "diff": float(sr_a - sr_b), "se": None, "z": None, "pvalue": None}
    se = np.sqrt(se_sq)
    z = (sr_a - sr_b) / se
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return {"sr_a": float(sr_a), "sr_b": float(sr_b),
            "diff": float(sr_a - sr_b), "se": float(se),
            "z": float(z), "pvalue": p}


def jkm_matrix(strategies: dict) -> pd.DataFrame:
    """Matrice carrée des p-values JKM entre toutes les paires."""
    keys = list(strategies.keys())
    n = len(keys)
    mat = np.full((n, n), np.nan)
    diff_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                diff_mat[i, j] = 0.0
                continue
            t = jkm_test(strategies[keys[i]]["strategy_return"],
                         strategies[keys[j]]["strategy_return"])
            if t["pvalue"] is not None:
                mat[i, j] = t["pvalue"]
                diff_mat[i, j] = t["diff"]
    p_df = pd.DataFrame(mat, index=keys, columns=keys)
    diff_df = pd.DataFrame(diff_mat, index=keys, columns=keys)
    return p_df, diff_df


# ============================================================================
# PLOTS
# ============================================================================
def plot_subperiod_sharpe(sub_df: pd.DataFrame, fname: Path):
    pivot = sub_df.pivot(index="label", columns="period", values="sharpe")
    pivot = pivot[["P1 (2022-06→2024-06)", "P2 (2024-06→2026-04)", "FULL"]]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(pivot) + 1.5))
    x = np.arange(len(pivot))
    width = 0.27
    for i, col in enumerate(pivot.columns):
        ax.barh(x + (i - 1) * width, pivot[col].values, height=width, label=col)
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Sharpe ann.")
    ax.set_title("Étape 9 — Sharpe par sous-période (axe A robustesse)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_cost_sensitivity(cost_df: pd.DataFrame, fname: Path):
    pivot = cost_df.pivot(index="label", columns="cost_bps", values="sharpe")
    pivot = pivot.sort_values(by=5.0, ascending=False)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(pivot) + 1.5))
    x = np.arange(len(pivot))
    width = 0.27
    for i, col in enumerate(pivot.columns):
        ax.barh(x + (i - 1) * width, pivot[col].values, height=width,
                label=f"{int(col)} bps")
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Sharpe ann.")
    ax.set_title("Étape 9 — Sensibilité aux coûts (axe B robustesse)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_dynamic_hmm_threshold(thr_df: pd.DataFrame, fname: Path):
    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    color_sr = "#2E86AB"
    ax1.plot(thr_df["threshold"], thr_df["sharpe"], "o-", color=color_sr, lw=2, label="Sharpe")
    ax1.set_xlabel("Seuil T sur max(p_bear, p_bull)")
    ax1.set_ylabel("Sharpe ann.", color=color_sr)
    ax1.tick_params(axis="y", labelcolor=color_sr)
    ax1.axvline(0.5, color="grey", ls=":", alpha=0.5, label="T=0.5 (argmax classique)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_act = "#D62828"
    ax2.plot(thr_df["threshold"], thr_df["pct_days_active"] * 100, "s--",
             color=color_act, lw=1.5, label="% jours actifs")
    ax2.set_ylabel("% jours actifs", color=color_act)
    ax2.tick_params(axis="y", labelcolor=color_act)

    ax1.set_title("Étape 9 — Sweep seuil HMM dynamique (axe C robustesse)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88), fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_jkm_heatmap(p_df: pd.DataFrame, labels: dict, fname: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    M = p_df.values
    im = ax.imshow(np.clip(M, 0, 0.2), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.2)
    ax.set_xticks(range(len(p_df.columns)))
    ax.set_xticklabels([labels.get(c, c) for c in p_df.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(p_df.index)))
    ax.set_yticklabels([labels.get(c, c) for c in p_df.index], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = "—" if i == j else f"{v:.2f}"
            color = "white" if v < 0.05 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Étape 9 — JKM p-values pairwise (axe D)\n(vert = p > 0.05 = SR égaux ; rouge = p < 0.05 = SR diffèrent)")
    fig.colorbar(im, ax=ax, label="p-value (clip 0.2)")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_robustness_scorecard(sub_df, cost_df, labels, fname):
    """Tableau visuel : pour chaque stratégie, le Sharpe full + range cost + delta P1-P2."""
    keys = sub_df["strategy_id"].unique()
    rows = []
    for k in keys:
        sub_k = sub_df[sub_df["strategy_id"] == k]
        cost_k = cost_df[cost_df["strategy_id"] == k]
        sr_full = sub_k[sub_k["period"] == "FULL"]["sharpe"].iloc[0]
        sr_p1 = sub_k[sub_k["period"].str.startswith("P1")]["sharpe"].iloc[0]
        sr_p2 = sub_k[sub_k["period"].str.startswith("P2")]["sharpe"].iloc[0]
        sr_5 = cost_k[cost_k["cost_bps"] == 5.0]["sharpe"].iloc[0]
        sr_20 = cost_k[cost_k["cost_bps"] == 20.0]["sharpe"].iloc[0]
        rows.append({
            "key": k, "label": labels.get(k, k),
            "sr_full": sr_full,
            "delta_p1_p2": sr_p1 - sr_p2,
            "delta_5_20": sr_5 - sr_20,
            "sr_5": sr_5, "sr_20": sr_20,
        })
    rows = sorted(rows, key=lambda r: r["sr_full"], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 2))
    y = np.arange(len(rows))
    sr_full = [r["sr_full"] for r in rows]
    sr_20 = [r["sr_20"] for r in rows]
    ax.barh(y, sr_full, color="#2E86AB", alpha=0.7, label="Sharpe @ 5 bps (full TEST)")
    ax.barh(y, sr_20, color="#D62828", alpha=0.5, label="Sharpe @ 20 bps")
    for i, r in enumerate(rows):
        ax.text(max(r["sr_full"], r["sr_20"]) + 0.05, i,
                f"ΔP1-P2={r['delta_p1_p2']:+.2f}  Δcost5-20={r['delta_5_20']:+.2f}",
                fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Sharpe ann.")
    ax.set_title("Étape 9 — Scorecard robustesse (Sharpe full + range cost + delta P1-P2)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


# ============================================================================
# UTILS
# ============================================================================
def _json_safe(v):
    if isinstance(v, dict): return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
    if isinstance(v, (np.integer, int)): return int(v)
    if v is None: return None
    return v


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 78)
    print("ÉTAPE 9 — Robustesse (sous-périodes, coûts, seuils, JKM, stress proxy)")
    print("=" * 78)

    df = load_inputs()
    print(f"[IN]  {len(df)} jours TEST, {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    strategies = build_all_strategies(df, cost_dec=COST_DEC_PRIMARY)
    labels = {k: s["label"] for k, s in strategies.items()}

    # ---- A. SOUS-PÉRIODES ----
    print("\n[A/E] Sous-périodes (split au midpoint)...")
    sub_df = subperiod_analysis(df, strategies)
    sub_df.to_csv(RES_DIR / "subperiod_metrics.csv", index=False)
    print(f"[OUT] subperiod_metrics.csv")
    # afficher pivot Sharpe
    pivot_sr = sub_df.pivot(index="label", columns="period", values="sharpe")
    pivot_sr = pivot_sr[["P1 (2022-06→2024-06)", "P2 (2024-06→2026-04)", "FULL"]]
    print(pivot_sr.round(2).to_string())

    # ---- B. COST SENSITIVITY ----
    print("\n[B/E] Cost sensitivity (5/10/20 bps)...")
    cost_df = cost_sensitivity(df)
    cost_df.to_csv(RES_DIR / "cost_sensitivity.csv", index=False)
    print(f"[OUT] cost_sensitivity.csv")
    pivot_cost = cost_df.pivot(index="label", columns="cost_bps", values="sharpe")
    print(pivot_cost.round(2).to_string())

    # ---- C. SEUILS HMM DYNAMIQUES ----
    print("\n[C/E] Seuils HMM dynamiques (sweep)...")
    thr_df = dynamic_hmm_threshold_sweep(df)
    thr_df.to_csv(RES_DIR / "dynamic_hmm_sweep.csv", index=False)
    print(f"[OUT] dynamic_hmm_sweep.csv")
    print(thr_df.round(3).to_string(index=False))

    # ---- D. JOBSON-KORKIE-MEMMEL ----
    print("\n[D/E] Jobson-Korkie-Memmel pairwise...")
    p_df, diff_df = jkm_matrix(strategies)
    p_df.to_csv(RES_DIR / "jkm_pvalue_matrix.csv")
    diff_df.to_csv(RES_DIR / "jkm_sharpe_diff_matrix.csv")
    print(f"[OUT] jkm_pvalue_matrix.csv, jkm_sharpe_diff_matrix.csv")
    # Print les paires significatives
    keys = list(strategies.keys())
    print("\n  Paires avec différence de Sharpe SIGNIFICATIVE (p < 0.05) :")
    n_sig = 0
    for i, ka in enumerate(keys):
        for j, kb in enumerate(keys):
            if i >= j: continue
            p = p_df.loc[ka, kb]
            d = diff_df.loc[ka, kb]
            if p is not None and not np.isnan(p) and p < 0.05:
                n_sig += 1
                print(f"    {labels[ka]:<40s} vs {labels[kb]:<40s} "
                      f"ΔSR={d:+.4f} p={p:.4f}")
    if n_sig == 0:
        print("    (aucune)")

    # ---- E. STRESS PROXY ALTERNATIF ----
    print("\n[E/E] Comparaison stress proxy : HMM=Neutral vs risk_regime=high ...")
    m7 = compute_compact_metrics(strategies["7_cnn_lstm_hmm_budget"]["strategy_return"],
                                  strategies["7_cnn_lstm_hmm_budget"]["position"], "continuous")
    m7b = compute_compact_metrics(strategies["7b_cnn_lstm_riskreg_budget"]["strategy_return"],
                                   strategies["7b_cnn_lstm_riskreg_budget"]["position"], "continuous")
    print(f"  Strat 7  (stress=HMM Neutral)   : Sharpe={m7['sharpe']:+.3f} | "
          f"MDD={m7['mdd']:+.2%} | eq_finale={m7['final_equity']:.3f}")
    print(f"  Strat 7b (stress=risk_regime hi): Sharpe={m7b['sharpe']:+.3f} | "
          f"MDD={m7b['mdd']:+.2%} | eq_finale={m7b['final_equity']:.3f}")

    # ---- Save JSON consolidé ----
    full = {
        "n_test_days": int(len(df)),
        "test_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "primary_cost_bps": COST_BPS_PRIMARY,
        "axes": ["A_subperiods", "B_cost_sensitivity",
                  "C_dynamic_hmm_thresholds", "D_jkm_pairwise", "E_stress_proxy"],
        "subperiods": sub_df.to_dict(orient="records"),
        "cost_sensitivity": cost_df.to_dict(orient="records"),
        "dynamic_hmm_sweep": thr_df.to_dict(orient="records"),
        "jkm_pvalue_matrix": p_df.to_dict(),
        "jkm_diff_matrix": diff_df.to_dict(),
        "stress_proxy_comparison": {
            "hmm_neutral_proxy": m7,
            "risk_regime_proxy": m7b,
        },
    }
    json_path = RES_DIR / "robustness_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(full), f, indent=2, default=str)
    print(f"\n[OUT] {json_path.name}")

    # ---- Plots ----
    print("\n[PLOTS] Génération 5 PNG ...")
    plot_subperiod_sharpe(sub_df, PLOTS_DIR / "etape9_subperiod_sharpe.png")
    plot_cost_sensitivity(cost_df, PLOTS_DIR / "etape9_cost_sensitivity.png")
    plot_dynamic_hmm_threshold(thr_df, PLOTS_DIR / "etape9_dynamic_hmm_threshold.png")
    plot_jkm_heatmap(p_df, labels, PLOTS_DIR / "etape9_jkm_heatmap.png")
    plot_robustness_scorecard(sub_df, cost_df, labels,
                               PLOTS_DIR / "etape9_robustness_scorecard.png")

    print("\n" + "=" * 78)
    print("ÉTAPE 9 — TERMINÉE")
    print("=" * 78)
    return full


if __name__ == "__main__":
    main()
