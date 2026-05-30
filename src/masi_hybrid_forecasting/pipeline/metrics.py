"""
Metrics library — Sharpe, Sortino, MDD, Calmar, DSR, JKM.
Réutilisé étapes 6-9.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

from .config import PPY


# ============================================================================
# CORE METRICS
# ============================================================================
def equity_from_log_returns(r: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(np.asarray(r, dtype=float)))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def sharpe_ann(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def sortino_ann(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(ppy))


def annualized_return(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.exp(r.mean() * ppy) - 1.0)


def annualized_vol(r: np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    return float(r.std() * np.sqrt(ppy))


def calmar(ann_ret: float, mdd: float) -> float:
    return float(ann_ret / abs(mdd)) if mdd != 0 else 0.0


def compute_full_metrics(strat_returns: np.ndarray, positions: np.ndarray,
                          mode: str = "binary",
                          regime_names: np.ndarray | None = None) -> dict:
    """
    Calcul complet : ann_return, ann_vol, Sharpe, Sortino, MDD, Calmar,
    eq finale, n_trades, turnover, exposition, (regime-conditionnel si fourni).
    """
    r = np.asarray(strat_returns, dtype=float)
    p = np.asarray(positions, dtype=float)
    eq = equity_from_log_returns(r)
    mdd = max_drawdown(eq)
    ann_ret = annualized_return(r)

    prev = np.concatenate([[0.0], p[:-1]])
    if mode == "binary":
        n_trades = int((p != prev).sum())
    elif mode == "continuous":
        n_trades = int((np.abs(p - prev) > 1e-9).sum())
    else:
        raise ValueError(f"mode inconnu : {mode!r} ; attendu 'binary' ou 'continuous'")

    out = {
        "ann_return": ann_ret,
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe_ann(r),
        "sortino": sortino_ann(r),
        "max_drawdown": mdd,
        "calmar": calmar(ann_ret, mdd),
        "final_equity": float(eq[-1]),
        "n_trades": n_trades,
        "turnover_mean": float(np.abs(p - prev).mean()),
        "avg_abs_exposure": float(np.abs(p).mean()),
        "pct_days_active": float((np.abs(p) > 1e-9).mean()),
    }

    if regime_names is not None:
        rc = {}
        for reg in ["Bear", "Neutral", "Bull"]:
            mask = (regime_names == reg)
            r_reg = r[mask]
            sd = r_reg.std()
            sr_reg = float(r_reg.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0
            rc[reg] = {
                "n": int(mask.sum()),
                "sharpe": sr_reg,
                "mean_return_ann": float(r_reg.mean() * PPY) if len(r_reg) > 0 else 0.0,
            }
        out["regime_conditional"] = rc

    return out


# ============================================================================
# DEFLATED SHARPE RATIO (Bailey & López de Prado 2014)
# ============================================================================
def deflated_sharpe_single(r: np.ndarray, sr0_threshold: float) -> dict:
    """
    DSR pour une stratégie unique étant donné sr0_threshold.
      PSR(SR0) = Φ( (SR̂ − SR0) · √(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) )
    """
    r = np.asarray(r, dtype=float)
    T = len(r)
    sd = r.std()
    sr_hat = float(r.mean() / sd) if sd > 0 else 0.0
    sk = float(skew(r))
    kt = float(kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - sk * sr_hat + (kt - 1) / 4 * sr_hat ** 2, 1e-12))
    psr = float(norm.cdf((sr_hat - sr0_threshold) * np.sqrt(T - 1) / denom))
    psr0 = float(norm.cdf(sr_hat * np.sqrt(T - 1) / denom))
    return {
        "sr_daily": sr_hat,
        "sr0_threshold": float(sr0_threshold),
        "skew": sk,
        "kurt_pearson": kt,
        "deflated_sharpe_psr_vs_sr0": psr,
        "psr_vs_zero": psr0,
    }


def deflated_sharpe_panel(returns_dict: dict, n_trials: int) -> dict:
    """
    DSR pour un panel de stratégies avec N trials.
      SR0 = √V · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]
    """
    sr_daily = {}
    for k, r in returns_dict.items():
        sd = np.std(r)
        sr_daily[k] = float(np.mean(r) / sd) if sd > 0 else 0.0
    v = float(np.var(list(sr_daily.values())))
    gamma = 0.5772156649
    sr0 = float(np.sqrt(max(v, 0.0)) *
                ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                 + gamma * norm.ppf(1 - 1 / (n_trials * np.e))))
    out = {}
    for k, r in returns_dict.items():
        out[k] = deflated_sharpe_single(np.asarray(r), sr0)
        out[k]["v_trial_sharpes"] = v
        out[k]["n_trials"] = n_trials
    return out


# ============================================================================
# JOBSON-KORKIE-MEMMEL (différence de Sharpe pairwise)
# ============================================================================
def jkm_test(r_a: np.ndarray, r_b: np.ndarray) -> dict:
    """
    H0 : SR_A = SR_B (daily). Statistique z ~ N(0, 1).
      SE² = (1/T) · [2(1−ρ) + 0.5·(SR_A² + SR_B² − 2·SR_A·SR_B·ρ²)]
    """
    a = np.asarray(r_a, dtype=float)
    b = np.asarray(r_b, dtype=float)
    T = len(a)
    if T < 30:
        return {"diff": None, "pvalue": None, "verdict": "T<30 trop court"}
    sd_a, sd_b = a.std(), b.std()
    if sd_a == 0 or sd_b == 0:
        return {"diff": None, "pvalue": None, "verdict": "std nulle"}
    sr_a, sr_b = a.mean() / sd_a, b.mean() / sd_b
    rho = float(np.corrcoef(a, b)[0, 1])
    se_sq = (1.0 / T) * (2.0 * (1.0 - rho)
                          + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2.0 * sr_a * sr_b * rho ** 2))
    if se_sq <= 0:
        return {"diff": float(sr_a - sr_b), "pvalue": None, "verdict": "SE² ≤ 0"}
    z = (sr_a - sr_b) / np.sqrt(se_sq)
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return {
        "sr_a_daily": float(sr_a),
        "sr_b_daily": float(sr_b),
        "diff": float(sr_a - sr_b),
        "z_stat": float(z),
        "pvalue": p,
        "verdict": "SR_A ≠ SR_B (p < 0.05)" if p < 0.05 else "SR_A = SR_B (p ≥ 0.05)",
    }
