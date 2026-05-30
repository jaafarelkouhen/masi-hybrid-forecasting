"""
================================================================================
ÉTAPE 7 — Couche Risque (VaR, Expected Shortfall, vol GARCH, régime de risque)
MASI Hybrid Forecasting System
================================================================================

PURPOSE
  Ajouter une couche de protection au-dessus du signal CNN-LSTM (modèle de
  production validé en étapes 5 et 6). On NE MODIFIE PAS le prédicteur : on
  ajoute des filtres de risque ex-ante qui peuvent masquer le signal en
  condition adverse (queue de distribution, vol élevée).

INPUT
  outputs/etape5/predictions_test.csv      (CNN-LSTM base12)
  outputs/etape4/regimes/masi_regimes_test.csv     (régimes HMM causaux)
  outputs/etape3/features/masi_features_train.csv  (warm-up VaR/ES)
  outputs/etape3/features/masi_features_val.csv    (warm-up VaR/ES)
  outputs/etape3/features/masi_features_test.csv   (TEST window 948 j)
  outputs/etape6/equity_curves.csv         (sanity check)

OUTPUT
  outputs/etape6/etape6_final_predictions.csv             (PRÉREQUIS : fichier
                                                        canonique unique pour
                                                        API/dashboard)
  outputs/etape7/risk_metrics_test.csv        (toutes métriques + signaux
                                                        filtrés + strategy_returns)
  outputs/etape7/risk_validation.json         (Kupiec, Christoffersen,
                                                        comparaison stratégies)
  reports/figures/etape7/etape7_vol_cone.png
  reports/figures/etape7/etape7_var_breaches.png
  reports/figures/etape7/etape7_return_distribution.png
  reports/figures/etape7/etape7_mdd_comparison.png

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5)
--------------------------------------------------------------------------------
D1  Production model = CNN-LSTM base12. Prédictions figées (étape 5). On consomme
    le CSV tel quel — pas de réentraînement.

D2  VaR 5% / ES 5% :
      - Historique : quantile empirique 5% sur fenêtre glissante 252j CAUSALE
        (closed='left' → fenêtre [t-252, t-1]). Jorion (2006).
      - Paramétrique normal : VaR = μ + σ_garch · Φ⁻¹(0.05),
                              ES  = μ - σ_garch · φ(z_α)/α
        avec σ_garch déjà fit TRAIN-only (étape 3) et causal.
        μ = rolling mean 252j causal des log_returns.

D3  Régime de risque ∈ {low, normal, high} :
      Discrétisation de σ_garch en 3 quantiles {q33, q67} FIGÉS sur TRAIN+VAL
      (anti-fuite L1). Distinct du régime HMM (étape 4 capture momentum
      directionnel ; ici on capture l'amplitude de risque).

D4  Trois variantes de signal protégé (pour étape 8 — comparaison combinatoire) :
      signal_var_filter  = sign(y_pred) · 𝟙{y_pred ≥ VaR_param_5}
      signal_es_filter   = sign(y_pred) · 𝟙{y_pred ≥ ES_param_5}
      signal_risk_regime = sign(y_pred) · 𝟙{risk_regime ≠ high}

D5  Validation backtest des VaR :
      - Kupiec POF (1995) : H0 taux de breach = 5%, LR ~ chi²(1)
      - Christoffersen (1998) : indépendance des breaches, LR_ind ~ chi²(1)

D6  Coûts de transaction : 5 bps one-way (cohérent étape 6 primary cost).

D7  Honnêteté (RULE 8 spirit) : si les filtres dégradent Sharpe sans réduire
    MDD significativement, le rapport l'écrit explicitement.

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE  (L1-L8)
--------------------------------------------------------------------------------
L1  Quantiles q33/q67 figés UNE FOIS sur TRAIN+VAL — pas refit sur TEST.
L2  HMM régimes consommés depuis étape 4 v2 (causaux, forward-only).
L3  Rolling VaR/ES : .shift(1).rolling(252, min_periods=252) → strictement causal.
L4  y_true (target_y_next) jamais utilisé pour décider le signal de t.
L5  Signal t → strategy_return t (où y_true_t = ln(P_{t+1}/P_t)), cohérent étape 6.
L6  Gap walk-forward déjà imposé étape 1 ; étape 7 ne réintroduit pas de données.
L7  Jours zero-volume déjà retirés étape 1.
L8  GARCH fit TRAIN-only (étape 3 garch_params_train.json) ; garch_vol causal.

Assertions programmatiques :
  - assert n_test == 948
  - assert quantiles figés (recalculés depuis TRAIN+VAL uniquement)
  - assert ordre temporel strict
  - assert pas de NaN sur TEST après warm-up
================================================================================
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTES (figées par prompt.md & étapes 1-6)
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
RISK_DIR = ROOT / "outputs" / "etape7"
PLOTS_DIR = ROOT / "reports" / "figures" / "etape7"
RISK_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

ROLL_WIN = 252        # fenêtre rolling = 1 an de trading
ALPHA = 0.05          # niveau VaR/ES 5%
COST_BPS = 5.0        # coût primaire étape 6 (5 bps one-way)
COST_DEC = COST_BPS / 10_000


# ============================================================================
# 1. CHARGEMENT
# ============================================================================
def load_inputs():
    """Charge prédictions CNN-LSTM, régimes, et série complète des features."""
    preds = pd.read_csv(ROOT / "outputs" / "etape5" / "predictions_test.csv")
    regs = pd.read_csv(ROOT / "outputs" / "etape4" / "regimes" / "masi_regimes_test.csv",
                       parse_dates=["date"])
    f_tr = pd.read_csv(ROOT / "outputs" / "etape3" / "features" / "masi_features_train.csv",
                       parse_dates=["date"])
    f_va = pd.read_csv(ROOT / "outputs" / "etape3" / "features" / "masi_features_val.csv",
                       parse_dates=["date"])
    f_te = pd.read_csv(ROOT / "outputs" / "etape3" / "features" / "masi_features_test.csv",
                       parse_dates=["date"])
    eq = pd.read_csv(ROOT / "outputs" / "etape6" / "equity_curves.csv",
                     parse_dates=["date"])
    return preds, regs, f_tr, f_va, f_te, eq


# ============================================================================
# 2. FICHIER CANONIQUE (prérequis étape 6 — base API/dashboard)
# ============================================================================
def build_final_predictions(preds, regs, eq) -> pd.DataFrame:
    """
    Construit etape6_final_predictions.csv au format canonique dashboard/API en
    utilisant CNN-LSTM base12 comme modèle de production.
    """
    assert len(preds) == len(regs), \
        f"Désalignement : preds={len(preds)} regs={len(regs)}"

    # cross-check : target_y_next dans regs == y_true dans preds (même source)
    diff = np.max(np.abs(preds["y_true"].values - regs["target_y_next"].values))
    assert diff < 1e-6, f"y_true ≠ target_y_next (max diff={diff:.2e})"
    print(f"[CHECK] y_true vs target_y_next : max écart = {diff:.2e} (OK <1e-6)")

    df = pd.DataFrame({
        "date": regs["date"].values,
        "actual_return":    preds["y_true"].values,
        "predicted_return": preds["y_pred"].values,
        "regime":           regs["regime"].astype(int).values,
        "regime_name":      regs["regime_name"].values,
    })

    df["signal_raw"] = np.sign(df["predicted_return"]).astype(int)
    df["signal"] = df["signal_raw"]  # compatibilité notebooks/étapes historiques

    # One-way turnover cost: a direct flip -1 -> +1 trades two exposure units.
    pos = df["signal_raw"].values.astype(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(pos - prev) * COST_DEC
    df["position"] = pos
    df["strategy_return"] = pos * df["actual_return"].values - cost
    df["equity"] = np.exp(np.cumsum(df["strategy_return"].values))
    df["strategy_name"] = "cnn_lstm_base12"
    df["mode"] = "binary"
    df["cost_bps"] = COST_BPS

    # Sanity check : somme(strategy_return) == ln(equity_finale CNN-LSTM)
    cnn_col = [c for c in eq.columns if "CNN" in c.upper()][0]
    final_eq = float(eq[cnn_col].iloc[-1])
    impl = np.log(final_eq)
    obs = float(df["strategy_return"].sum())
    spread = abs(impl - obs)
    print(f"[CHECK] CNN-LSTM eq finale = {final_eq:.6f} | "
          f"ln(eq) = {impl:+.6f} | sum(strategy_return) = {obs:+.6f} | "
          f"écart = {spread:.2e}")
    assert spread < 1e-4, f"Écart >1e-4 : {spread:.4e} — recheck cost logic"

    out = ROOT / "outputs" / "etape6" / "etape6_final_predictions.csv"
    df.to_csv(out, index=False)
    print(f"[OUT] {out.name} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


# ============================================================================
# 3. VaR / ES historiques (rolling causal)
# ============================================================================
def rolling_var_es_hist(series_full: pd.Series, win=ROLL_WIN, alpha=ALPHA):
    """
    VaR/ES historiques sur fenêtre glissante CAUSALE.
    Renvoie deux Series alignées sur series_full :
      var_t = quantile_α(series[t-win : t-1])
      es_t  = mean( series[t-win:t-1] | series[t-win:t-1] ≤ var_t )
    """
    shifted = series_full.shift(1)
    var = shifted.rolling(window=win, min_periods=win).quantile(alpha)

    def _es_left(arr):
        q = np.quantile(arr, alpha)
        below = arr[arr <= q]
        return float(below.mean()) if len(below) > 0 else np.nan

    es = shifted.rolling(window=win, min_periods=win).apply(_es_left, raw=True)
    return var, es


# ============================================================================
# 4. VaR / ES paramétriques normaux (avec σ GARCH causal)
# ============================================================================
def parametric_var_es(mu: pd.Series, sigma: pd.Series, alpha=ALPHA):
    """
    VaR/ES paramétriques normaux :
      VaR_α = μ + σ · Φ⁻¹(α)              (Φ⁻¹(0.05) ≈ -1.6449)
      ES_α  = μ - σ · φ(Φ⁻¹(α))/α          (queue gauche)
    """
    z = norm.ppf(alpha)
    var_p = mu + sigma * z
    es_p = mu - sigma * (norm.pdf(z) / alpha)
    return var_p, es_p


# ============================================================================
# 5. RÉGIME DE RISQUE (quantiles figés)
# ============================================================================
def assign_risk_regime(vol_values: np.ndarray, q33: float, q67: float) -> np.ndarray:
    """Discrétise vol_garch en {low, normal, high} avec seuils figés TRAIN+VAL."""
    return np.where(vol_values <= q33, "low",
                    np.where(vol_values <= q67, "normal", "high"))


# ============================================================================
# 6. TESTS DE VALIDATION (Kupiec POF + Christoffersen indépendance)
# ============================================================================
def kupiec_pof(breaches: np.ndarray, alpha=ALPHA) -> dict:
    """
    Kupiec Proportion of Failures (1995). H0 : taux de breach = alpha.
    LR_uc = -2 · log[ (α^x · (1-α)^(T-x)) / (p̂^x · (1-p̂)^(T-x)) ] ~ chi²(1)
    """
    T = int(len(breaches))
    x = int(breaches.sum())
    p = x / T if T > 0 else 0.0

    if x == 0 or x == T:
        return {"T": T, "x": x, "p_obs": p, "p_target": alpha,
                "lr": None, "pvalue": None, "verdict": "DEGENERATE"}

    log_null = x * np.log(alpha) + (T - x) * np.log(1 - alpha)
    log_alt = x * np.log(p) + (T - x) * np.log(1 - p)
    lr = -2.0 * (log_null - log_alt)
    pval = float(1.0 - chi2.cdf(lr, df=1))
    verdict = "OK (taux conforme à 5%)" if pval > 0.05 \
              else "REJETÉ (taux ≠ 5%)"
    return {"T": T, "x": x, "p_obs": p, "p_target": alpha,
            "lr": float(lr), "pvalue": pval, "verdict": verdict}


def christoffersen_indep(breaches: np.ndarray) -> dict:
    """
    Christoffersen (1998) test d'indépendance des breaches.
    On compte les transitions {00, 01, 10, 11}. LR_ind ~ chi²(1).
    """
    b = breaches.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        if b[i - 1] == 0 and b[i] == 0: n00 += 1
        elif b[i - 1] == 0 and b[i] == 1: n01 += 1
        elif b[i - 1] == 1 and b[i] == 0: n10 += 1
        else: n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    n = n0 + n1

    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "lr_ind": None, "pvalue": None, "verdict": "DEGENERATE"}

    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / n
    eps = 1e-12
    log_null = (n00 + n10) * np.log(max(1 - p, eps)) + \
               (n01 + n11) * np.log(max(p, eps))
    log_alt = (n00 * np.log(max(1 - p01, eps)) + n01 * np.log(max(p01, eps))
               + n10 * np.log(max(1 - p11, eps)) + n11 * np.log(max(p11, eps)))
    lr_ind = -2.0 * (log_null - log_alt)
    pval = float(1.0 - chi2.cdf(lr_ind, df=1))
    verdict = "OK (breaches indépendants)" if pval > 0.05 \
              else "REJETÉ (clusters de breaches)"
    return {"n00": int(n00), "n01": int(n01), "n10": int(n10), "n11": int(n11),
            "lr_ind": float(lr_ind), "pvalue": pval, "verdict": verdict}


# ============================================================================
# 7. METRICS Sharpe / MDD
# ============================================================================
def sharpe_ann(r: pd.Series | np.ndarray, ppy: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(ppy))


def equity_from_log_returns(r: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(r))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def strat_returns(positions: np.ndarray, y_true: np.ndarray,
                  cost: float = COST_DEC) -> np.ndarray:
    prev = np.concatenate([[0.0], positions[:-1]])
    c = np.abs(positions - prev) * cost
    return positions * y_true - c


# ============================================================================
# 8. PLOTS
# ============================================================================
def plot_vol_cone(df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["vol_garch"], color="#2E86AB",
            label="σ GARCH(1,1) (causale)", lw=1.2)
    ax.plot(df["date"], df["vol_realized_21"], color="#A23B72",
            label="vol réalisée 21j (causale)", lw=1.0, alpha=0.75)
    ax.fill_between(df["date"], 0, df["vol_garch"] * 2,
                    color="#2E86AB", alpha=0.10, label="bande ±2σ_garch")
    ax.set_title("Étape 7 — Cône de volatilité GARCH(1,1) vs réalisée (TEST 948j)")
    ax.set_ylabel("Volatilité quotidienne (log-ret)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_var_breaches(df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["actual_return"], color="#444", lw=0.6,
            label="retour réalisé")
    ax.plot(df["date"], df["var_param_5"], color="#D62828", lw=1.0,
            label="VaR paramétrique 5% (causale)")
    breaches = df["actual_return"] < df["var_param_5"]
    ax.scatter(df["date"][breaches], df["actual_return"][breaches],
               color="red", s=18, zorder=5,
               label=f"breach ({int(breaches.sum())} j / 948)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Étape 7 — Breaches du VaR paramétrique 5% sur TEST")
    ax.set_ylabel("Log-retour quotidien")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_return_distribution(df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(df["actual_return"], bins=60, color="#888",
            alpha=0.75, edgecolor="white")
    ax.axvline(df["var_param_5"].mean(), color="#D62828", lw=2, ls="--",
               label=f"VaR moy. (param.) = {df['var_param_5'].mean():+.4f}")
    ax.axvline(df["es_param_5"].mean(), color="#000", lw=2, ls=":",
               label=f"ES moy.  (param.) = {df['es_param_5'].mean():+.4f}")
    ax.axvline(df["var_hist_5"].mean(), color="#F77F00", lw=2, ls="--",
               label=f"VaR moy. (hist.)  = {df['var_hist_5'].mean():+.4f}")
    ax.set_title("Étape 7 — Distribution des retours TEST + VaR/ES moyens")
    ax.set_xlabel("Log-retour quotidien")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


def plot_mdd_comparison(dates: np.ndarray, equities: dict, fname: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {"CNN-LSTM nu": "#2E86AB", "+ filtre VaR": "#D62828",
              "+ filtre ES": "#000", "+ régime risque": "#1A8A1A"}
    for label, eq in equities.items():
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        ax.plot(dates, dd, label=f"{label} (MDD={dd.min():.2%})",
                lw=1.2, color=colors.get(label))
    ax.set_title("Étape 7 — Drawdowns : CNN-LSTM nu vs filtres risque (TEST)")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print(f"[OUT] {fname.name}")


# ============================================================================
# 9. UTILITAIRES
# ============================================================================
def _json_safe(v):
    """Convertit en types JSON-sérialisables (gère np.* et NaN)."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if (v is None or np.isnan(v)) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if v is None:
        return None
    return v


def count_trades(positions: np.ndarray) -> int:
    """Compte le nombre de jours où la position change (cohérent étape 6)."""
    prev = np.concatenate([[0.0], positions[:-1].astype(float)])
    return int(((positions.astype(float) - prev) != 0.0).sum())


# ============================================================================
# 10. ORCHESTRATION
# ============================================================================
def main():
    print("=" * 78)
    print("ÉTAPE 7 — Couche risque (VaR, ES, vol GARCH, régime de risque)")
    print("=" * 78)

    preds, regs, f_tr, f_va, f_te, eq = load_inputs()
    print(f"[IN]  predictions: {len(preds)} | regimes: {len(regs)} | "
          f"features train/val/test: {len(f_tr)}/{len(f_va)}/{len(f_te)}")

    # --- (a) Prérequis : fichier canonique étape 6 ---
    print("\n[1/6] Construction etape6_final_predictions.csv ...")
    df_pred = build_final_predictions(preds, regs, eq)

    # --- (b) Série totale pour rolling causal ---
    print("\n[2/6] Préparation série totale (train+val+test) pour rolling causal ...")
    full = pd.concat([
        f_tr[["date", "log_return", "garch_vol", "roll_vol_21"]],
        f_va[["date", "log_return", "garch_vol", "roll_vol_21"]],
        f_te[["date", "log_return", "garch_vol", "roll_vol_21"]],
    ], ignore_index=True)
    full = full.sort_values("date").reset_index(drop=True)
    assert full["date"].is_monotonic_increasing, "Ordre temporel cassé"
    n_total = len(full)
    print(f"[INFO] série totale = {n_total} jours, "
          f"{full['date'].iloc[0].date()} → {full['date'].iloc[-1].date()}")

    # --- (c) VaR/ES historiques (rolling 252j causal) ---
    print("\n[3/6] VaR/ES historiques (rolling 252j causal) ...")
    var_h, es_h = rolling_var_es_hist(full["log_return"], win=ROLL_WIN, alpha=ALPHA)
    full["var_hist_5"] = var_h
    full["es_hist_5"] = es_h

    # --- (d) VaR/ES paramétriques GARCH ---
    print("[3/6] VaR/ES paramétriques GARCH (μ rolling + σ GARCH causaux) ...")
    mu_roll = full["log_return"].shift(1).rolling(ROLL_WIN, min_periods=ROLL_WIN).mean()
    var_p, es_p = parametric_var_es(mu_roll, full["garch_vol"], alpha=ALPHA)
    full["var_param_5"] = var_p
    full["es_param_5"] = es_p

    # --- (e) Régime de risque (q33/q67 FIGÉS sur TRAIN+VAL) ---
    print("\n[4/6] Régime de risque (quantiles σ_garch TRAIN+VAL figés) ...")
    n_tr_va = len(f_tr) + len(f_va)
    vol_train_val = full["garch_vol"].iloc[:n_tr_va].dropna()
    q33 = float(vol_train_val.quantile(0.33))
    q67 = float(vol_train_val.quantile(0.67))
    print(f"[INFO] seuils figés : q33={q33:.5f} | q67={q67:.5f}")
    full["risk_regime"] = assign_risk_regime(full["garch_vol"].values, q33, q67)

    # --- (f) Slice TEST + signaux filtrés ---
    print("\n[5/6] Extraction slice TEST et signaux filtrés ...")
    test_start = f_te["date"].iloc[0]
    test_end = f_te["date"].iloc[-1]
    mask = (full["date"] >= test_start) & (full["date"] <= test_end)
    risk_test = full.loc[mask].reset_index(drop=True)
    assert len(risk_test) == len(df_pred), \
        f"slice TEST = {len(risk_test)} vs predictions = {len(df_pred)}"

    out = df_pred.merge(
        risk_test[["date", "var_hist_5", "es_hist_5", "var_param_5", "es_param_5",
                   "garch_vol", "roll_vol_21", "risk_regime"]],
        on="date", how="left"
    ).rename(columns={"garch_vol": "vol_garch", "roll_vol_21": "vol_realized_21"})

    # Vérif anti-NaN sur TEST (warm-up satisfait grâce à TRAIN+VAL)
    for col in ["var_hist_5", "es_hist_5", "var_param_5", "es_param_5"]:
        assert out[col].notna().all(), f"NaN détecté dans {col} sur TEST"

    # Signaux filtrés
    s = out["signal"].values.astype(int)
    yp = out["predicted_return"].values
    vp = out["var_param_5"].values
    ep = out["es_param_5"].values
    rr = out["risk_regime"].values

    out["signal_var_filter"] = np.where(yp >= vp, s, 0).astype(int)
    out["signal_es_filter"] = np.where(yp >= ep, s, 0).astype(int)
    out["signal_risk_regime"] = np.where(rr != "high", s, 0).astype(int)

    # Strategy returns filtrés (avec coût)
    yt = out["actual_return"].values
    out["strategy_return_var"] = strat_returns(out["signal_var_filter"].values.astype(float), yt)
    out["strategy_return_es"] = strat_returns(out["signal_es_filter"].values.astype(float), yt)
    out["strategy_return_riskreg"] = strat_returns(out["signal_risk_regime"].values.astype(float), yt)

    risk_csv = RISK_DIR / "risk_metrics_test.csv"
    out.to_csv(risk_csv, index=False)
    print(f"[OUT] {risk_csv.name} ({len(out)} lignes, {len(out.columns)} colonnes)")

    # --- (g) Validation : Kupiec + Christoffersen ---
    print("\n[6/6] Validation backtest VaR : Kupiec POF + Christoffersen ...")
    breaches_h = (out["actual_return"].values < out["var_hist_5"].values)
    breaches_p = (out["actual_return"].values < out["var_param_5"].values)

    k_h = kupiec_pof(breaches_h, alpha=ALPHA)
    k_p = kupiec_pof(breaches_p, alpha=ALPHA)
    c_h = christoffersen_indep(breaches_h)
    c_p = christoffersen_indep(breaches_p)

    print(f"[KUPIEC hist]  breaches={k_h['x']:>3}/{k_h['T']} "
          f"({k_h['p_obs']*100:5.2f}%) LR={k_h['lr']:.2f} "
          f"p={k_h['pvalue']:.3f} → {k_h['verdict']}")
    print(f"[KUPIEC para]  breaches={k_p['x']:>3}/{k_p['T']} "
          f"({k_p['p_obs']*100:5.2f}%) LR={k_p['lr']:.2f} "
          f"p={k_p['pvalue']:.3f} → {k_p['verdict']}")
    print(f"[CHRIST hist]  n01={c_h['n01']:>3} n11={c_h['n11']:>3} "
          f"LR_ind={c_h['lr_ind'] if c_h['lr_ind'] is not None else float('nan'):.2f} "
          f"p={c_h['pvalue'] if c_h['pvalue'] is not None else float('nan'):.3f} → {c_h['verdict']}")
    print(f"[CHRIST para]  n01={c_p['n01']:>3} n11={c_p['n11']:>3} "
          f"LR_ind={c_p['lr_ind'] if c_p['lr_ind'] is not None else float('nan'):.2f} "
          f"p={c_p['pvalue'] if c_p['pvalue'] is not None else float('nan'):.3f} → {c_p['verdict']}")

    # --- (h) Résumés comparatifs Sharpe / MDD ---
    sr_nu = sharpe_ann(out["strategy_return"])
    sr_var = sharpe_ann(out["strategy_return_var"])
    sr_es = sharpe_ann(out["strategy_return_es"])
    sr_rr = sharpe_ann(out["strategy_return_riskreg"])

    eq_nu = equity_from_log_returns(out["strategy_return"].values)
    eq_var = equity_from_log_returns(out["strategy_return_var"].values)
    eq_es = equity_from_log_returns(out["strategy_return_es"].values)
    eq_rr = equity_from_log_returns(out["strategy_return_riskreg"].values)

    mdd_nu = max_drawdown(eq_nu)
    mdd_var = max_drawdown(eq_var)
    mdd_es = max_drawdown(eq_es)
    mdd_rr = max_drawdown(eq_rr)

    tr_nu = count_trades(out["signal"].values.astype(float))
    tr_var = count_trades(out["signal_var_filter"].values.astype(float))
    tr_es = count_trades(out["signal_es_filter"].values.astype(float))
    tr_rr = count_trades(out["signal_risk_regime"].values.astype(float))

    print()
    print(f"[COMP] CNN-LSTM nu       : Sharpe={sr_nu:+.3f} | MDD={mdd_nu:+.2%} | "
          f"eq_finale={eq_nu[-1]:.3f} | trades={tr_nu}")
    print(f"[COMP] + filtre VaR_p    : Sharpe={sr_var:+.3f} | MDD={mdd_var:+.2%} | "
          f"eq_finale={eq_var[-1]:.3f} | trades={tr_var}")
    print(f"[COMP] + filtre ES_p     : Sharpe={sr_es:+.3f} | MDD={mdd_es:+.2%} | "
          f"eq_finale={eq_es[-1]:.3f} | trades={tr_es}")
    print(f"[COMP] + régime risque   : Sharpe={sr_rr:+.3f} | MDD={mdd_rr:+.2%} | "
          f"eq_finale={eq_rr[-1]:.3f} | trades={tr_rr}")

    # --- (i) Répartition régimes de risque sur TEST ---
    regime_counts = out["risk_regime"].value_counts().to_dict()
    print(f"\n[INFO] Régime risque TEST : {regime_counts}")

    # --- (j) Sauvegarde JSON validation ---
    validation = {
        "n_test_days": int(len(out)),
        "test_range": [str(out["date"].iloc[0].date()), str(out["date"].iloc[-1].date())],
        "alpha": ALPHA,
        "rolling_window": ROLL_WIN,
        "cost_bps": COST_BPS,
        "risk_regime_thresholds": {
            "q33": q33, "q67": q67,
            "source": "TRAIN+VAL garch_vol quantiles, frozen (anti-leakage L1)"
        },
        "kupiec": {"historical": k_h, "parametric": k_p},
        "christoffersen": {"historical": c_h, "parametric": c_p},
        "strategy_comparison": {
            "cnn_lstm_nu": {"sharpe": sr_nu, "mdd": mdd_nu,
                            "final_equity": float(eq_nu[-1]), "n_trades": tr_nu},
            "cnn_lstm_var_filter": {"sharpe": sr_var, "mdd": mdd_var,
                                     "final_equity": float(eq_var[-1]), "n_trades": tr_var},
            "cnn_lstm_es_filter": {"sharpe": sr_es, "mdd": mdd_es,
                                    "final_equity": float(eq_es[-1]), "n_trades": tr_es},
            "cnn_lstm_risk_regime": {"sharpe": sr_rr, "mdd": mdd_rr,
                                      "final_equity": float(eq_rr[-1]), "n_trades": tr_rr},
        },
        "risk_regime_counts_test": {k: int(v) for k, v in regime_counts.items()},
        "var_es_summary": {
            "var_hist_5_mean": float(out["var_hist_5"].mean()),
            "var_param_5_mean": float(out["var_param_5"].mean()),
            "es_hist_5_mean": float(out["es_hist_5"].mean()),
            "es_param_5_mean": float(out["es_param_5"].mean()),
            "worst_actual_return": float(out["actual_return"].min()),
            "best_actual_return": float(out["actual_return"].max()),
        }
    }

    json_path = RISK_DIR / "risk_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(validation), f, indent=2, default=str)
    print(f"[OUT] {json_path.name}")

    # --- (k) Plots ---
    print("\n[PLOTS] Génération 4 PNG ...")
    plot_vol_cone(out, PLOTS_DIR / "etape7_vol_cone.png")
    plot_var_breaches(out, PLOTS_DIR / "etape7_var_breaches.png")
    plot_return_distribution(out, PLOTS_DIR / "etape7_return_distribution.png")
    plot_mdd_comparison(out["date"].values,
                        {"CNN-LSTM nu": eq_nu, "+ filtre VaR": eq_var,
                         "+ filtre ES": eq_es, "+ régime risque": eq_rr},
                        PLOTS_DIR / "etape7_mdd_comparison.png")

    print("\n" + "=" * 78)
    print("ÉTAPE 7 — TERMINÉE")
    print("=" * 78)
    return out, validation


if __name__ == "__main__":
    main()
