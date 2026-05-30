"""
Risk layer — VaR (historique + paramétrique GARCH), ES, régime de risque.
Réutilisé de l'étape 7. Causal (anti-fuite L1-L8).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .config import (
    ALPHA_VAR,
    COST_DEC,
    FEATURES_TEST,
    FEATURES_TRAIN,
    FEATURES_VAL,
    PREDICTIONS_CSV,
    REGIMES_CSV,
    RISK_METRICS_CSV,
    ROLL_VAR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CALCULS VaR / ES
# ============================================================================
def rolling_var_es_hist(series: pd.Series, win: int = ROLL_VAR,
                         alpha: float = ALPHA_VAR) -> tuple[pd.Series, pd.Series]:
    """VaR/ES historiques causaux (fenêtre [t-win, t-1])."""
    shifted = series.shift(1)
    var = shifted.rolling(window=win, min_periods=win).quantile(alpha)

    def _es_left(arr):
        q = np.quantile(arr, alpha)
        below = arr[arr <= q]
        return float(below.mean()) if len(below) > 0 else np.nan

    es = shifted.rolling(window=win, min_periods=win).apply(_es_left, raw=True)
    return var, es


def parametric_var_es(mu: pd.Series, sigma: pd.Series,
                       alpha: float = ALPHA_VAR) -> tuple[pd.Series, pd.Series]:
    """VaR/ES paramétriques normaux (μ + σ·Φ⁻¹ ; μ − σ·φ(z)/α)."""
    z = norm.ppf(alpha)
    var = mu + sigma * z
    es = mu - sigma * (norm.pdf(z) / alpha)
    return var, es


def assign_risk_regime(vol_values: np.ndarray, q33: float, q67: float) -> np.ndarray:
    """Discrétise vol_garch en {low, normal, high}."""
    return np.where(vol_values <= q33, "low",
                    np.where(vol_values <= q67, "normal", "high"))


# ============================================================================
# TESTS BACKTESTING (Kupiec POF + Christoffersen indépendance)
# ============================================================================
def kupiec_pof(breaches: np.ndarray, alpha: float = ALPHA_VAR) -> dict:
    T = int(len(breaches))
    x = int(breaches.sum())
    p = x / T if T > 0 else 0.0
    if x == 0 or x == T:
        return {"T": T, "x": x, "p_obs": p, "lr": None, "pvalue": None,
                "verdict": "DEGENERATE"}
    log_null = x * np.log(alpha) + (T - x) * np.log(1 - alpha)
    log_alt = x * np.log(p) + (T - x) * np.log(1 - p)
    lr = -2.0 * (log_null - log_alt)
    pval = float(1.0 - chi2.cdf(lr, df=1))
    return {"T": T, "x": x, "p_obs": p, "p_target": alpha,
            "lr": float(lr), "pvalue": pval,
            "verdict": "OK" if pval > 0.05 else "REJETÉ"}


def christoffersen_indep(breaches: np.ndarray) -> dict:
    b = breaches.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        if b[i - 1] == 0 and b[i] == 0:
            n00 += 1
        elif b[i - 1] == 0 and b[i] == 1:
            n01 += 1
        elif b[i - 1] == 1 and b[i] == 0:
            n10 += 1
        else:
            n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    n = n0 + n1
    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "lr_ind": None, "pvalue": None, "verdict": "DEGENERATE"}
    p01, p11 = n01 / n0, n11 / n1
    p = (n01 + n11) / n
    eps = 1e-12
    log_null = (n00 + n10) * np.log(max(1 - p, eps)) + (n01 + n11) * np.log(max(p, eps))
    log_alt = (n00 * np.log(max(1 - p01, eps)) + n01 * np.log(max(p01, eps))
               + n10 * np.log(max(1 - p11, eps)) + n11 * np.log(max(p11, eps)))
    lr_ind = -2.0 * (log_null - log_alt)
    pval = float(1.0 - chi2.cdf(lr_ind, df=1))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": float(lr_ind), "pvalue": pval,
            "verdict": "OK" if pval > 0.05 else "REJETÉ"}


# ============================================================================
# COMMANDE CLI : `python -m masi_hybrid_forecasting.pipeline risk`
# ============================================================================
def run(args) -> None:
    """
    Génère la couche risque (recalcule étape 7) et écrit le CSV.
    Si --output non spécifié, écrit à RISK_METRICS_CSV (chemin canonique).
    """
    output_path = Path(args.output) if args.output else RISK_METRICS_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement features train/val/test pour rolling causal ...")
    f_tr = pd.read_csv(FEATURES_TRAIN, parse_dates=["date"])
    f_va = pd.read_csv(FEATURES_VAL, parse_dates=["date"])
    f_te = pd.read_csv(FEATURES_TEST, parse_dates=["date"])
    regs = pd.read_csv(REGIMES_CSV, parse_dates=["date"])
    preds = pd.read_csv(PREDICTIONS_CSV)

    full = pd.concat([
        f_tr[["date", "log_return", "garch_vol", "roll_vol_21"]],
        f_va[["date", "log_return", "garch_vol", "roll_vol_21"]],
        f_te[["date", "log_return", "garch_vol", "roll_vol_21"]],
    ], ignore_index=True).sort_values("date").reset_index(drop=True)

    logger.info("Calcul VaR/ES historiques + paramétriques (causaux)...")
    var_h, es_h = rolling_var_es_hist(full["log_return"], win=ROLL_VAR, alpha=ALPHA_VAR)
    mu_roll = full["log_return"].shift(1).rolling(ROLL_VAR, min_periods=ROLL_VAR).mean()
    var_p, es_p = parametric_var_es(mu_roll, full["garch_vol"], alpha=ALPHA_VAR)
    full["var_hist_5"] = var_h
    full["es_hist_5"] = es_h
    full["var_param_5"] = var_p
    full["es_param_5"] = es_p

    n_tr_va = len(f_tr) + len(f_va)
    vol_train_val = full["garch_vol"].iloc[:n_tr_va].dropna()
    q33 = float(vol_train_val.quantile(0.33))
    q67 = float(vol_train_val.quantile(0.67))
    full["risk_regime"] = assign_risk_regime(full["garch_vol"].values, q33, q67)
    logger.info(f"Seuils figés TRAIN+VAL : q33={q33:.5f}  q67={q67:.5f}")

    # Slice TEST
    mask = (full["date"] >= f_te["date"].iloc[0]) & (full["date"] <= f_te["date"].iloc[-1])
    risk_test = full.loc[mask].reset_index(drop=True)
    assert len(risk_test) == len(regs), \
        f"slice TEST {len(risk_test)} != regimes {len(regs)}"

    # Join avec regs et preds pour avoir le CSV complet
    out = pd.DataFrame({
        "date": regs["date"].values,
        "actual_return": preds["y_true"].values,
        "predicted_return": preds["y_pred"].values,
        "regime": regs["regime"].astype(int).values,
        "regime_name": regs["regime_name"].values,
    })
    out["signal"] = np.sign(out["predicted_return"]).astype(int)
    pos = out["signal"].values.astype(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    out["strategy_return"] = pos * out["actual_return"] - (np.abs(pos - prev) * COST_DEC)

    risk_cols = ["var_hist_5", "es_hist_5", "var_param_5", "es_param_5",
                 "garch_vol", "roll_vol_21", "risk_regime"]
    out = out.merge(risk_test[["date"] + risk_cols], on="date", how="left") \
              .rename(columns={"garch_vol": "vol_garch", "roll_vol_21": "vol_realized_21"})

    # Validation Kupiec/Christoffersen (sanity)
    breaches_p = (out["actual_return"].values < out["var_param_5"].values)
    k = kupiec_pof(breaches_p, alpha=ALPHA_VAR)
    c = christoffersen_indep(breaches_p)
    kupiec_p = "NA" if k["pvalue"] is None else f"{k['pvalue']:.3f}"
    christ_lr = "NA" if c["lr_ind"] is None else f"{c['lr_ind']:.2f}"
    christ_p = "NA" if c["pvalue"] is None else f"{c['pvalue']:.3f}"
    logger.info(f"Kupiec POF : {k['x']}/{k['T']} breaches "
                f"({k['p_obs']*100:.2f}%) p={kupiec_p} → {k['verdict']}")
    logger.info(f"Christoffersen : LR_ind={christ_lr} p={christ_p} → {c['verdict']}")

    out.to_csv(output_path, index=False)
    logger.info(f"Couche risque écrite : {output_path}  ({len(out)} lignes)")
