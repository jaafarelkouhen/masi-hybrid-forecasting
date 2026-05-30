"""
================================================================================
ÉTAPE 8 — Stratégies combinées (B&H, CNN-LSTM ± HMM ± Risk ± VaR-budget)
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Comparer 7 stratégies sur la fenêtre TEST canonique (948 jours), avec :
    - Deflated Sharpe Ratio (Bailey & López de Prado 2014) — N=7 trials
    - Diebold-Mariano vs CNN-LSTM nu (Diebold & Mariano 1995)
    - Décomposition régime-conditionnelle (Bear / Neutral / Bull)

  Stratégies évaluées :
    1. Buy & Hold                              (référence passive)
    2. CNN-LSTM nu                             (référence étape 6 — best DA)
    3. CNN-LSTM + HMM-gate                     (skip Neutral regime)
    4. CNN-LSTM + risk-gate                    (skip jours σ_GARCH high)
    5. CNN-LSTM + HMM + risk                   (intersection des deux gates)
    6. CNN-LSTM × VaR-budget                   (sizing continu B/|VaR|)
    7. CNN-LSTM × HMM-conditional budget       (B varie par régime HMM)

INPUT
  outputs/etape6/etape6_final_predictions.csv         (date, actual, predicted, regime, signal, strategy_return)
  outputs/etape7/risk_metrics_test.csv    (VaR, ES, vol GARCH, risk_regime, signaux filtrés)

OUTPUT
  outputs/etape8/strategies_returns.csv   (date + positions + returns + equity par stratégie)
  outputs/etape8/strategies_metrics.json  (Sharpe, Sortino, MDD, Calmar, DSR, DM, regime-cond)
  reports/figures/etape8/etape8_equity_curves.png
  reports/figures/etape8/etape8_drawdowns.png
  reports/figures/etape8/etape8_regime_heatmap.png
  reports/figures/etape8/etape8_sharpe_mdd_scatter.png
  reports/figures/etape8/etape8_dsr_summary.png

--------------------------------------------------------------------------------
ATTRIBUTION (prompt.md : éviter le plagiat)
--------------------------------------------------------------------------------
Les stratégies 6 et 7 (VaR-budget et HMM-conditional budget) s'inspirent de
techniques de risk-budgeting telles qu'implémentées dans :
  `_analysis_research_notebooks/masi-risk-research-notebooks-main/src/analysis/
   economic_evaluation.py` (compute_weights_from_budget_and_var L310-326,
   weights_hmm_lstm_quantile_budget L366-416)
La formule  w_t = min(cap, B_t / |VaR_t|)  est une convention standard de
risk-budgeting (Bertrand & Prigent 2003, Roncalli 2014). Notre adaptation :
  - Setup long-short (cap=+1, floor=-1) au lieu de long-only ;
  - VaR consommé depuis étape 7 (paramétrique GARCH) au lieu d'un LSTM-VaR ;
  - "Stress" identifié via régime HMM = Neutral (étape 4) — proxy défendable
    car étape 6 a montré que c'est le régime où les modèles échouent.

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5)
--------------------------------------------------------------------------------
D1  7 stratégies sélectionnées par contraste : passif (1) / actif binaire (2-5) /
    actif continu (6-7). Pas d'inclusion de filtre VaR/ES sur ŷ (étape 7 a
    montré qu'ils sont dégénérés sur le CNN-LSTM — éviter cherry-picking).

D2  Convention coûts :
      - Stratégies binaires (1-5) : flat 5 bps si position change (cohérent étape 6)
      - Stratégies continues (6-7) : 5 bps × |Δw| (turnover proportionnel,
        cohérent avec economic_evaluation.py L107)
    Justification : pour signal binaire, un flip -1 → +1 est traité comme
    un changement unique (convention héritée étape 6 pour la comparabilité).
    Pour signal continu, le coût doit suivre |Δw| sinon on sous-estime le coût
    réel des repondérations partielles.

D3  Strategy returns = pos · y_true − cost  (cohérent étapes 6 et 7).

D4  Régime HMM directionnel (étape 4 v2 causal) → distinction stress/normal :
      - HMM_regime ∈ {Bear, Bull} → "clean" : modèle attend une direction nette
      - HMM_regime = Neutral → "stress" : modèle dans la zone d'incertitude
    Choix défendable : étape 6 a quantifié -0.31 Sharpe en Neutral pour CNN-LSTM.

D5  Budget B = 0.01 (1%) pour stratégie 6 — valeur standard littérature
    risk-budgeting et celle utilisée dans le repo source.

D6  Budget rolling pour stratégie 7 :
      B_t = q30 de |VaR_param| sur 60j passés    si HMM = Neutral
      B_t = q70 de |VaR_param| sur 60j passés    sinon
    Fenêtre 60j et quantiles 30/70 alignés sur le repo source.

D7  DSR avec N=7 trials (vs N=5 étape 6) — le seuil SR0 est plus élevé, ce qui
    rend la déflation plus stricte. Honnête.

D8  DM test (Diebold-Mariano 1995) sur d_t = r_strat − r_cnn_lstm_nu pour
    chaque variante. H0 : E[d_t]=0. Statistique asymptotiquement N(0,1).

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE  (L1-L8)
--------------------------------------------------------------------------------
L1  Quantiles rolling de B_t (stratégie 7) : .shift(1).rolling(60, min_periods=60)
L2  HMM régimes consommés depuis étape 4 v2 (causaux)
L3  Idem : tous les rolling causals, fenêtre [t-60, t-1] pour B_t
L4  y_true jamais utilisé pour décider position
L5  Position t exécutée contre y_true_t = ln(P_{t+1}/P_t)
L6/L7 inhérent étapes précédentes
L8  VaR consommé depuis étape 7 (déjà GARCH TRAIN-only)
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

from scipy.stats import chi2, norm, skew, kurtosis

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTES (figées prompt.md & repo source pour comparabilité)
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "outputs" / "etape8"
PLOTS_DIR = ROOT / "reports" / "figures" / "etape8"
RES_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

COST_BPS = 5.0
COST_DEC = COST_BPS / 10_000
ROLL_BUDGET = 60       # fenêtre rolling pour B_t (repo source convention)
B_STD = 0.01           # 1% budget standard
Q_STRESS = 0.30        # quantile bas en stress (Neutral) — plus conservateur
Q_NORMAL = 0.70        # quantile haut en normal (Bear/Bull)
N_TRIALS_DSR = 7
PPY = 252              # jours de trading par an
EPS_VAR = 1e-6         # plancher pour |VaR| pour éviter division par zéro


# ============================================================================
# 1. CHARGEMENT
# ============================================================================
def load_inputs() -> pd.DataFrame:
    """Charge le fichier canonique étape 6 + métriques risque étape 7."""
    risk = pd.read_csv(ROOT / "outputs" / "etape7" / "risk_metrics_test.csv",
                       parse_dates=["date"])
    # risk a déjà date, actual_return, predicted_return, regime, regime_name,
    # signal, strategy_return, var_*, es_*, vol_garch, vol_realized_21,
    # risk_regime, signal_*, strategy_return_*
    assert len(risk) == 948, f"Attendu 948 jours TEST, reçu {len(risk)}"
    return risk


# ============================================================================
# 2. UTILITAIRES STRATEGY RETURNS
# ============================================================================
def strat_returns_binary(positions: np.ndarray, y_true: np.ndarray,
                          cost: float = COST_DEC) -> np.ndarray:
    """Convention binaire : one-way cost × |Δposition|."""
    positions = positions.astype(float)
    prev = np.concatenate([[0.0], positions[:-1]])
    c = cost * np.abs(positions - prev)
    return positions * y_true - c


def strat_returns_continuous(positions: np.ndarray, y_true: np.ndarray,
                              cost: float = COST_DEC) -> np.ndarray:
    """Convention continue (repo source) : cost × |Δw| proportionnel au turnover."""
    positions = positions.astype(float)
    prev = np.concatenate([[0.0], positions[:-1]])
    c = cost * np.abs(positions - prev)
    return positions * y_true - c


# ============================================================================
# 3. CONSTRUCTION DES 7 STRATÉGIES
# ============================================================================
def build_strategies(df: pd.DataFrame) -> dict:
    """Renvoie un dict {key: {position, strategy_return, label, mode}}."""
    y_true = df["actual_return"].values
    y_pred = df["predicted_return"].values
    signal = np.sign(y_pred).astype(int)
    regime = df["regime_name"].values
    risk_reg = df["risk_regime"].values
    var_abs = np.abs(df["var_param_5"].values)

    out = {}

    # 1. Buy & Hold (passif)
    pos = np.ones(len(df))
    out["1_buy_hold"] = {
        "label": "Buy & Hold",
        "mode": "binary",
        "position": pos,
        "strategy_return": strat_returns_binary(pos, y_true),
    }

    # 2. CNN-LSTM nu (référence étape 6)
    pos = signal.astype(float)
    out["2_cnn_lstm_nu"] = {
        "label": "CNN-LSTM nu",
        "mode": "binary",
        "position": pos,
        "strategy_return": strat_returns_binary(pos, y_true),
    }

    # 3. CNN-LSTM + HMM-gate (trade seulement si Bear ou Bull)
    pos = np.where(np.isin(regime, ["Bear", "Bull"]), signal, 0).astype(float)
    out["3_cnn_lstm_hmm_gate"] = {
        "label": "CNN-LSTM + HMM-gate",
        "mode": "binary",
        "position": pos,
        "strategy_return": strat_returns_binary(pos, y_true),
    }

    # 4. CNN-LSTM + risk-gate (skip si σ_GARCH = high) — déjà calculé étape 7
    pos = df["signal_risk_regime"].values.astype(float)
    out["4_cnn_lstm_risk_gate"] = {
        "label": "CNN-LSTM + risk-gate",
        "mode": "binary",
        "position": pos,
        "strategy_return": df["strategy_return_riskreg"].values.astype(float),
    }

    # 5. CNN-LSTM + HMM-gate + risk-gate (intersection)
    pos = np.where(
        np.isin(regime, ["Bear", "Bull"]) & (risk_reg != "high"),
        signal, 0
    ).astype(float)
    out["5_cnn_lstm_hmm_risk"] = {
        "label": "CNN-LSTM + HMM + risk-gate",
        "mode": "binary",
        "position": pos,
        "strategy_return": strat_returns_binary(pos, y_true),
    }

    # 6. CNN-LSTM × VaR-budget (sizing continu, B=0.01)
    w_raw = np.minimum(1.0, B_STD / np.maximum(var_abs, EPS_VAR))
    pos = signal.astype(float) * w_raw
    out["6_cnn_lstm_var_budget"] = {
        "label": "CNN-LSTM × VaR-budget",
        "mode": "continuous",
        "position": pos,
        "strategy_return": strat_returns_continuous(pos, y_true),
    }

    # 7. CNN-LSTM × HMM-conditional budget
    var_series = pd.Series(var_abs)
    shifted = var_series.shift(1)
    q_stress = shifted.rolling(ROLL_BUDGET, min_periods=ROLL_BUDGET).quantile(Q_STRESS)
    q_normal = shifted.rolling(ROLL_BUDGET, min_periods=ROLL_BUDGET).quantile(Q_NORMAL)
    is_stress = (regime == "Neutral")
    B_t = np.where(is_stress, q_stress.values, q_normal.values)
    # Fallback pour les 60 premiers jours (NaN rolling)
    B_t = np.where(np.isnan(B_t), B_STD, B_t)
    w_t = np.minimum(1.0, B_t / np.maximum(var_abs, EPS_VAR))
    pos = signal.astype(float) * w_t
    out["7_cnn_lstm_hmm_budget"] = {
        "label": "CNN-LSTM × HMM-conditional budget",
        "mode": "continuous",
        "position": pos,
        "strategy_return": strat_returns_continuous(pos, y_true),
    }

    return out


# ============================================================================
# 4. METRICS
# ============================================================================
def equity_from_log_returns(r: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(r))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def compute_metrics(strat_ret: np.ndarray, positions: np.ndarray,
                    regime_names: np.ndarray, mode: str) -> dict:
    r = np.asarray(strat_ret, dtype=float)
    p = np.asarray(positions, dtype=float)

    eq = equity_from_log_returns(r)
    mdd = max_drawdown(eq)
    mean_d = float(r.mean())
    std_d = float(r.std())
    ann_ret = float(np.exp(mean_d * PPY) - 1.0)
    ann_vol = float(std_d * np.sqrt(PPY))
    sr = (mean_d / std_d * np.sqrt(PPY)) if std_d > 0 else 0.0

    downside = r[r < 0]
    sortino = (mean_d / downside.std() * np.sqrt(PPY)) if len(downside) > 1 and downside.std() > 0 else 0.0

    prev = np.concatenate([[0.0], p[:-1]])
    if mode == "binary":
        n_trades = int((p != prev).sum())
    else:
        n_trades = int((np.abs(p - prev) > 1e-9).sum())
    turnover = float(np.abs(p - prev).mean())
    avg_exposure = float(np.abs(p).mean())
    pct_active = float((np.abs(p) > 1e-9).mean())

    # Régime-conditionnel
    regime_metrics = {}
    for reg in ["Bear", "Neutral", "Bull"]:
        mask = (regime_names == reg)
        r_reg = r[mask]
        sd = r_reg.std()
        sr_reg = (r_reg.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0
        regime_metrics[reg] = {
            "n": int(mask.sum()),
            "sharpe": float(sr_reg),
            "mean_return_ann": float(r_reg.mean() * PPY) if len(r_reg) > 0 else 0.0,
        }

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(sr),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(ann_ret / abs(mdd)) if mdd != 0 else 0.0,
        "final_equity": float(eq[-1]),
        "n_trades": int(n_trades),
        "turnover_mean": float(turnover),
        "avg_abs_exposure": float(avg_exposure),
        "pct_days_active": float(pct_active),
        "regime_conditional": regime_metrics,
    }


# ============================================================================
# 5. DEFLATED SHARPE RATIO (Bailey & López de Prado 2014, N=7)
# ============================================================================
def deflated_sharpe(strategy_returns: dict, n_trials: int = N_TRIALS_DSR) -> dict:
    """DSR par stratégie avec déflation pour N trials et non-normalité."""
    sr_daily = {}
    for k, r in strategy_returns.items():
        sd = np.std(r)
        sr_daily[k] = float(np.mean(r) / sd) if sd > 0 else 0.0

    sr_arr = np.array(list(sr_daily.values()))
    v = float(np.var(sr_arr))
    gamma = 0.5772156649  # Euler-Mascheroni

    # SR0 = seuil de Sharpe attendu sous le null pour le max des N trials
    sr0 = float(np.sqrt(max(v, 0.0)) *
                ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                 + gamma * norm.ppf(1 - 1 / (n_trials * np.e))))

    out = {}
    T = len(next(iter(strategy_returns.values())))
    for k, r in strategy_returns.items():
        r_arr = np.asarray(r, dtype=float)
        skew_r = float(skew(r_arr))
        kurt_r = float(kurtosis(r_arr, fisher=False))  # Pearson
        sr_hat = sr_daily[k]
        denom_sq = 1.0 - skew_r * sr_hat + (kurt_r - 1) / 4 * sr_hat ** 2
        denom = np.sqrt(max(denom_sq, 1e-12))
        psr_vs_sr0 = float(norm.cdf((sr_hat - sr0) * np.sqrt(T - 1) / denom))
        psr_vs_zero = float(norm.cdf(sr_hat * np.sqrt(T - 1) / denom))
        verdict = ("PUBLISHABLE (≥0.95)" if psr_vs_sr0 >= 0.95
                   else ("STRONG (≥0.90)" if psr_vs_sr0 >= 0.90
                         else "WEAK"))
        out[k] = {
            "sr_daily": sr_hat,
            "sr0_threshold": sr0,
            "v_trial_sharpes": v,
            "n_trials": n_trials,
            "skew": skew_r,
            "kurt_pearson": kurt_r,
            "deflated_sharpe_psr_vs_sr0": psr_vs_sr0,
            "psr_vs_zero": psr_vs_zero,
            "verdict": verdict,
        }
    return out


# ============================================================================
# 6. DIEBOLD-MARIANO (1995)
# ============================================================================
def diebold_mariano(r_a: np.ndarray, r_b: np.ndarray, h: int = 1) -> dict:
    """
    DM test sur d_t = r_a − r_b. H0 : E[d]=0.
    Pour h=1 (horizon 1 jour), variance simple ; pour h>1 Newey-West HAC.
    Statistique asymptotique ~ N(0, 1).
    """
    d = np.asarray(r_a, dtype=float) - np.asarray(r_b, dtype=float)
    n = len(d)
    d_mean = float(d.mean())
    if h == 1:
        var = float(d.var(ddof=1))
    else:
        var = float(d.var(ddof=1))
        for lag in range(1, h):
            cov = float(np.mean((d[:-lag] - d_mean) * (d[lag:] - d_mean)))
            w = 1.0 - lag / h
            var += 2.0 * w * cov
    if var <= 0 or n < 2:
        return {"dm_stat": None, "pvalue": None, "mean_diff": d_mean}
    dm = d_mean / np.sqrt(var / n)
    pval = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    return {"dm_stat": float(dm), "pvalue": pval, "mean_diff": d_mean}


# ============================================================================
# 7. PLOTS
# ============================================================================
COLORS = {
    "1_buy_hold":              "#808080",
    "2_cnn_lstm_nu":           "#2E86AB",
    "3_cnn_lstm_hmm_gate":     "#A23B72",
    "4_cnn_lstm_risk_gate":    "#1A8A1A",
    "5_cnn_lstm_hmm_risk":     "#000000",
    "6_cnn_lstm_var_budget":   "#F77F00",
    "7_cnn_lstm_hmm_budget":   "#D62828",
}


def plot_equity_curves(dates, equities: dict, labels: dict, fname: Path):
    fig, ax = plt.subplots(figsize=(11.5, 5))
    for k, eq in equities.items():
        ax.plot(dates, eq, lw=1.3, color=COLORS.get(k, None),
                label=f"{labels[k]} (eq={eq[-1]:.2f})")
    ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_title("Étape 8 — Equity curves : 7 stratégies sur TEST (948 jours)")
    ax.set_ylabel("Equity (base 1.0)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_drawdowns(dates, equities: dict, labels: dict, fname: Path):
    fig, ax = plt.subplots(figsize=(11.5, 5))
    for k, eq in equities.items():
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        ax.plot(dates, dd, lw=1.2, color=COLORS.get(k, None),
                label=f"{labels[k]} (MDD={dd.min():.2%})")
    ax.set_title("Étape 8 — Drawdowns comparés")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_regime_heatmap(metrics: dict, labels: dict, fname: Path):
    keys = list(metrics.keys())
    regs = ["Bear", "Neutral", "Bull"]
    M = np.array([[metrics[k]["regime_conditional"][r]["sharpe"]
                   for r in regs] for k in keys])
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(keys) + 1.5))
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels(regs)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([labels[k] for k in keys])
    for i in range(len(keys)):
        for j in range(len(regs)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=8.5)
    ax.set_title("Étape 8 — Sharpe régime-conditionnel × stratégie")
    fig.colorbar(im, ax=ax, label="Sharpe ann.")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_sharpe_mdd_scatter(metrics: dict, labels: dict, fname: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, m in metrics.items():
        ax.scatter(abs(m["max_drawdown"]) * 100, m["sharpe"],
                   s=140, color=COLORS.get(k, "#888"), edgecolor="black",
                   zorder=3, label=labels[k])
        ax.annotate(labels[k], (abs(m["max_drawdown"]) * 100, m["sharpe"]),
                    fontsize=8, xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("|MDD| (%)")
    ax.set_ylabel("Sharpe ann.")
    ax.set_title("Étape 8 — Frontière Sharpe × MDD (cherchez le coin haut-gauche)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_dsr_summary(metrics: dict, dsr: dict, labels: dict, fname: Path):
    keys = list(metrics.keys())
    sr = [metrics[k]["sharpe"] for k in keys]
    dsr_v = [dsr[k]["deflated_sharpe_psr_vs_sr0"] for k in keys]
    sr0_thr = dsr[keys[0]]["sr0_threshold"] * np.sqrt(PPY)  # annualisé

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    y = np.arange(len(keys))
    ax1.barh(y, sr, color=[COLORS.get(k, "#888") for k in keys])
    ax1.set_yticks(y); ax1.set_yticklabels([labels[k] for k in keys], fontsize=9)
    ax1.axvline(sr0_thr, color="red", ls="--",
                label=f"SR0 ann. seuil (≈{sr0_thr:.2f})")
    ax1.set_xlabel("Sharpe annualisé"); ax1.set_title("Sharpe par stratégie")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(y, dsr_v, color=[COLORS.get(k, "#888") for k in keys])
    ax2.set_yticks(y); ax2.set_yticklabels([labels[k] for k in keys], fontsize=9)
    ax2.axvline(0.95, color="green", ls="--", label="DSR=0.95 (publishable)")
    ax2.axvline(0.90, color="orange", ls=":", label="DSR=0.90 (strong)")
    ax2.set_xlabel("Deflated Sharpe (PSR vs SR0)")
    ax2.set_title(f"DSR par stratégie (N={N_TRIALS_DSR} trials)")
    ax2.set_xlim(0, 1); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


# ============================================================================
# 8. UTILS
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
# 9. ORCHESTRATION
# ============================================================================
def main():
    print("=" * 78)
    print("ÉTAPE 8 — Stratégies combinées (7 candidates)")
    print("=" * 78)

    df = load_inputs()
    print(f"[IN]  TEST = {len(df)} jours, "
          f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    # --- Construire 7 stratégies ---
    print("\n[1/5] Construction des 7 stratégies ...")
    strategies = build_strategies(df)
    for k, s in strategies.items():
        eq_f = float(np.exp(np.cumsum(s["strategy_return"]))[-1])
        print(f"  [{k}] {s['label']:<40s} | mode={s['mode']:<10s} | eq_finale={eq_f:.3f}")

    # --- Métriques ---
    print("\n[2/5] Calcul métriques + régime-conditionnel ...")
    metrics = {}
    strat_ret_dict = {}
    labels = {k: s["label"] for k, s in strategies.items()}
    for k, s in strategies.items():
        m = compute_metrics(s["strategy_return"], s["position"],
                            df["regime_name"].values, s["mode"])
        metrics[k] = {"label": s["label"], "mode": s["mode"], **m}
        strat_ret_dict[k] = s["strategy_return"]

    # Sanity check : CNN-LSTM nu doit retrouver les métriques courantes étape 6.
    sr_nu = metrics["2_cnn_lstm_nu"]["sharpe"]
    eq_nu = metrics["2_cnn_lstm_nu"]["final_equity"]
    step6 = json.load(open(ROOT / "outputs" / "etape6" / "backtest_metrics.json"))[
        "models"
    ]["cnn_lstm_base12"]
    expected_sr = step6["sharpe"]
    expected_eq = step6["final_equity"]
    print(f"\n[CHECK] CNN-LSTM nu : Sharpe={sr_nu:+.3f} "
          f"(étape 6={expected_sr:+.3f}) | "
          f"eq_finale={eq_nu:.3f} (étape 6={expected_eq:.3f})")
    assert abs(sr_nu - expected_sr) < 1e-6, "Régression Sharpe CNN-LSTM nu"
    assert abs(eq_nu - expected_eq) < 1e-6, "Régression eq finale CNN-LSTM nu"

    # Sanity check : Buy & Hold cumul cohérent avec sum(actual_return) - 5bps
    bh_eq = metrics["1_buy_hold"]["final_equity"]
    bh_expected = float(np.exp(df["actual_return"].sum() - COST_DEC))
    print(f"[CHECK] Buy & Hold eq finale = {bh_eq:.4f} | "
          f"attendu = exp(sum(actual) - 5bps) = {bh_expected:.4f} | "
          f"écart = {abs(bh_eq - bh_expected):.2e}")
    assert abs(bh_eq - bh_expected) < 1e-6, "Buy & Hold désaligné"

    # --- DSR ---
    print("\n[3/5] Deflated Sharpe Ratio (N=7) ...")
    dsr = deflated_sharpe(strat_ret_dict, n_trials=N_TRIALS_DSR)
    sr0_thr_daily = dsr["2_cnn_lstm_nu"]["sr0_threshold"]
    print(f"[INFO] SR0 daily threshold = {sr0_thr_daily:.4f} "
          f"(annualisé ≈ {sr0_thr_daily * np.sqrt(PPY):.3f})")
    for k in metrics:
        d = dsr[k]
        print(f"  [{k}] SR_d={d['sr_daily']:+.4f} | "
              f"DSR_vs_SR0={d['deflated_sharpe_psr_vs_sr0']:.3f} | "
              f"PSR_vs_0={d['psr_vs_zero']:.3f} | {d['verdict']}")

    # --- DM tests vs CNN-LSTM nu ---
    print("\n[4/5] Diebold-Mariano vs CNN-LSTM nu ...")
    ref = strat_ret_dict["2_cnn_lstm_nu"]
    dm = {}
    for k, r in strat_ret_dict.items():
        if k == "2_cnn_lstm_nu":
            continue
        dm_res = diebold_mariano(np.asarray(r), np.asarray(ref), h=1)
        dm[k] = dm_res
        sig = "SIG" if (dm_res["pvalue"] is not None and dm_res["pvalue"] < 0.05) else "ns"
        sign = "+" if dm_res["mean_diff"] > 0 else "−"
        print(f"  [{k}] mean_diff={dm_res['mean_diff']:+.5f} ({sign}) | "
              f"DM={dm_res['dm_stat']:+.2f} | p={dm_res['pvalue']:.3f} → {sig}")

    # --- Comparatif Sharpe / MDD / Calmar / DA ---
    print("\n[COMP] Récapitulatif :")
    print(f"{'stratégie':<40s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} "
          f"{'Calmar':>8s} {'eq_fin':>8s} {'turn':>7s} {'expo':>7s} {'trades':>7s}")
    for k in metrics:
        m = metrics[k]
        print(f"{m['label']:<40s} {m['sharpe']:+8.3f} {m['sortino']:+8.3f} "
              f"{m['max_drawdown']:+8.2%} {m['calmar']:+8.2f} {m['final_equity']:8.3f} "
              f"{m['turnover_mean']:7.3f} {m['avg_abs_exposure']:7.2f} {m['n_trades']:7d}")

    # --- Save CSV per-day ---
    print("\n[5/5] Sauvegarde sorties ...")
    out_df = df[["date", "actual_return", "predicted_return", "regime_name",
                 "risk_regime"]].copy()
    for k, s in strategies.items():
        out_df[f"pos_{k}"] = s["position"]
        out_df[f"ret_{k}"] = s["strategy_return"]
        out_df[f"eq_{k}"] = np.exp(np.cumsum(s["strategy_return"]))
    returns_csv = RES_DIR / "strategies_returns.csv"
    out_df.to_csv(returns_csv, index=False)
    print(f"[OUT] {returns_csv.name} ({len(out_df)} lignes, {len(out_df.columns)} colonnes)")

    # --- Save JSON full metrics ---
    full = {
        "n_test_days": int(len(df)),
        "test_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "cost_bps": COST_BPS,
        "parameters": {
            "B_std": B_STD,
            "rolling_budget_win": ROLL_BUDGET,
            "q_stress": Q_STRESS,
            "q_normal": Q_NORMAL,
            "n_trials_dsr": N_TRIALS_DSR,
            "ppy": PPY,
        },
        "strategies": metrics,
        "deflated_sharpe": dsr,
        "diebold_mariano_vs_cnn_lstm_nu": dm,
    }
    json_path = RES_DIR / "strategies_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(full), f, indent=2, default=str)
    print(f"[OUT] {json_path.name}")

    # --- Plots ---
    print("\n[PLOTS] Génération 5 PNG ...")
    equities = {k: np.exp(np.cumsum(s["strategy_return"])) for k, s in strategies.items()}
    dates = df["date"].values
    plot_equity_curves(dates, equities, labels, PLOTS_DIR / "etape8_equity_curves.png")
    plot_drawdowns(dates, equities, labels, PLOTS_DIR / "etape8_drawdowns.png")
    plot_regime_heatmap(metrics, labels, PLOTS_DIR / "etape8_regime_heatmap.png")
    plot_sharpe_mdd_scatter(metrics, labels, PLOTS_DIR / "etape8_sharpe_mdd_scatter.png")
    plot_dsr_summary(metrics, dsr, labels, PLOTS_DIR / "etape8_dsr_summary.png")

    print("\n" + "=" * 78)
    print("ÉTAPE 8 — TERMINÉE")
    print("=" * 78)
    return out_df, full


if __name__ == "__main__":
    main()
