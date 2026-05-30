"""
================================================================================
ÉTAPE 4 — HMM Market Regime Detection (Bull / Neutral / Bear)
MASI Hybrid Forecasting System  —  v2: comparative observation-set study
================================================================================

PURPOSE
  Detect latent market regimes with a Gaussian Hidden Markov Model, and emit a
  LEAKAGE-FREE regime feature (label + one-hot + soft probabilities) for the
  CNN-LSTM (Étape 5) "Regime-as-Feature" input.

  v2 NOTE — why a comparative study:
  The first pass (observations = [log_return, garch_vol]) produced PERSISTENT but
  DESCRIPTIVE-ONLY regimes: they separated volatility, not return direction, and
  did not predict next-day return. A HMM's regimes are only as directional as the
  series it observes. Raw daily return is ~noise; a SMOOTHED momentum signal is
  persistent and directional. So v2 fits several observation sets and selects the
  best one on TRAIN+VAL criteria, with TEST held out as the honest verdict.

INPUT
  outputs/etape3/features/masi_features_{train,val,test}.csv

OUTPUT
  outputs/etape4/regimes/masi_regimes_{train,val,test}.csv   (Étape 3 features + regime cols)
  outputs/etape4/regimes/hmm_params.json                     (winner HMM, ordered)
  outputs/etape4/regimes/hmm_obs_scaler_train_only.json      (L1 — TRAIN-only)
  outputs/etape4/regimes/regime_evaluation.json              (winner honest metrics)
  outputs/etape4/regimes/spec_comparison.json                (all specs, full table)
  reports/figures/etape4/*.png                                 (5 diagnostic plots)

--------------------------------------------------------------------------------
METHODOLOGICAL DECISIONS  (prompt.md RULE 5 — every choice justified)
--------------------------------------------------------------------------------
D1  COMPARATIVE OBSERVATION SETS. Five candidate HMM observation sets are fitted
    (3-state Gaussian HMM each). Idea under test: replace the noisy raw daily
    return by a persistent MOMENTUM signal so the regimes become directional.
       S1  [log_return, garch_vol]   - v1 baseline (volatility-driven)
       S2  [log_return]              - univariate raw return (Monteiro 2025 style)
       S3  [roll_mean_21, garch_vol] - 21-day momentum + volatility
       S4  [roll_mean_5, garch_vol]  - short momentum + volatility
       S5  [roll_mean_21, roll_vol_21] - momentum + realized vol (no GARCH)

D2  SELECTION ON TRAIN+VAL ONLY (TEST held out). A spec is eligible if regime
    persistence (mean transmat diagonal) >= 0.85. Among eligible specs, the
    winner is the one whose regime shows a CONSISTENT directional signal — the
    Bull-minus-Bear next-day return spread is positive on BOTH train AND val —
    ranked by the VAL spread. If NO spec qualifies, the v1 volatility spec (S1)
    is kept and the verdict remains DESCRIPTIVE_ONLY. TEST is never used to
    select — it is reported afterwards as the untouched out-of-sample verdict.
    This avoids selecting a spec by data-snooping the test set.

D3  Observations standardized with TRAIN-only mean/std (L1). HMM fit on TRAIN
    only (L2). 12 EM restarts per fit (non-convex). 3 states (project design;
    BIC over {2,3,4} reported for the winner).

D4  *** THE CRITICAL ANTI-LEAKAGE POINT ***
    The regime FEATURE must be CAUSAL: regime_t may use only observations <= t.
    Viterbi (model.predict) and the smoothed posterior (predict_proba) decode
    over the WHOLE sequence -> regime_t would depend on the FUTURE -> LEAKAGE.
    => The regime feature is computed with a manual CAUSAL FORWARD-FILTER
       (scaled forward algorithm): filtered_t = P(state_t | obs_{0..t}).
    Viterbi is computed only for plots/diagnostics, never exported as a feature.

D5  States re-indexed by ascending mean of the FIRST observation (the return /
    momentum axis) -> 0=Bear, 1=Neutral, 2=Bull.

D6  HONEST EVALUATION. Persistence, per-regime return & volatility, regime-
    conditional NEXT-DAY return (the predictive test), OOS coverage, and a naive
    regime-timed strategy. Verdict thresholds are explicit. Weak results are
    reported, not hidden.

--------------------------------------------------------------------------------
ANTI-LEAKAGE COMPLIANCE
--------------------------------------------------------------------------------
  L1  Observation scaler fit on TRAIN rows only.                      ENFORCED
  L2  HMM fit on TRAIN rows only; forward-applied to VAL/TEST.         ENFORCED
  L3  regime_t feature uses a CAUSAL forward-filter (obs <= t only).   ENFORCED (D4)
  L4  target_y_next inherited from Étape 1.                            INHERITED
  L6  Splits use Étape 1 dates.                                        INHERITED
  L8  garch_vol uses Étape 3 TRAIN-frozen GARCH params; per-window     PARTIAL
      GARCH + HMM refit is done in the Étape 5 walk-forward.           (by design)

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
from matplotlib.patches import Patch

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

try:                                            # UTF-8 stdout for Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEAT_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape3", "features")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape1", "splits")
REG_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape4", "regimes")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape4")

os.makedirs(REG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SEEDS = 12                     # EM random restarts per fit (D3)
N_STATES = 3                     # Bull / Neutral / Bear (project objective)
N_STATES_GRID = [2, 3, 4]        # reported for the winner spec only
TARGET = "target_y_next"
TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5
REGIME_NAMES = ["Bear", "Neutral", "Bull"]

# Candidate observation sets (D1). First column = the directional axis (D5).
SPECS: Dict[str, List[str]] = {
    "S1_ret_garchvol":   ["log_return", "garch_vol"],
    "S2_ret_only":       ["log_return"],
    "S3_mom21_garchvol": ["roll_mean_21", "garch_vol"],
    "S4_mom5_garchvol":  ["roll_mean_5", "garch_vol"],
    "S5_mom21_rvol21":   ["roll_mean_21", "roll_vol_21"],
}
PERSISTENCE_FLOOR = 0.85         # eligibility threshold (D2)
FALLBACK_SPEC = "S1_ret_garchvol"

REGIME_COLORS = {"Bear": "#d62728", "Neutral": "#bdbd62", "Bull": "#2ca02c"}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_features() -> Dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "val", "test"):
        p = os.path.join(FEAT_DIR, f"masi_features_{name}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p} — run scripts/03_feature_engineering.py first.")
        d = pd.read_csv(p)
        d["date"] = pd.to_datetime(d["date"])
        out[name] = d.set_index("date").sort_index()
        print(f"  {name:5s}: {len(out[name]):4d} rows  "
              f"{out[name].index.min().date()} -> {out[name].index.max().date()}")
    return out


def load_masi_close() -> pd.Series:
    """MASI close level — for the regime-timeline plot only (not a model input)."""
    d = pd.read_csv(os.path.join(SPLITS_DIR, "masi_clean_full.csv"))
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["masi_close"].sort_index()


# =============================================================================
# 2. HMM FITTING — multi-seed (D3)
# =============================================================================

def fit_hmm_best(X: np.ndarray, n_states: int, n_seeds: int = N_SEEDS
                 ) -> Tuple[GaussianHMM, float]:
    """Fit GaussianHMM with n_seeds random restarts; keep the best train log-lik."""
    best_model, best_ll = None, -np.inf
    for s in range(n_seeds):
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, tol=1e-4, random_state=RANDOM_SEED + s)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll:
            best_model, best_ll = m, ll
    if best_model is None:
        raise RuntimeError(f"HMM failed to fit (n_states={n_states})")
    return best_model, best_ll


# =============================================================================
# 3. STATE ORDERING — Bear < Neutral < Bull by mean of the directional axis (D5)
# =============================================================================

def reindex_model(model: GaussianHMM) -> dict:
    """Re-index states so 0=Bear..K-1=Bull by ascending mean of observation col 0."""
    perm = np.argsort(model.means_[:, 0])
    return {
        "startprob": model.startprob_[perm],
        "transmat": model.transmat_[np.ix_(perm, perm)],
        "means": model.means_[perm],
        "covars": model.covars_[perm],
        "_perm": perm,
    }


# =============================================================================
# 4. CAUSAL FORWARD-FILTER  *** THE ANTI-LEAKAGE CORE (D4, L3) ***
# =============================================================================

def emission_loglik(means: np.ndarray, covars: np.ndarray, X: np.ndarray) -> np.ndarray:
    """log b_j(o_t) for every t, j — Gaussian emission. Shape (T, K)."""
    T, K = len(X), len(means)
    ll = np.empty((T, K))
    for j in range(K):
        ll[:, j] = multivariate_normal.logpdf(X, mean=means[j], cov=covars[j],
                                              allow_singular=True)
    return ll


def causal_forward_filter(params: dict, X: np.ndarray) -> np.ndarray:
    """
    Scaled forward algorithm. Returns filtered probabilities
        filtered[t, j] = P(state_t = j | obs_{0..t})
    which depend ONLY on observations up to and including t  =>  leakage-free.
    Viterbi / smoothed posteriors would use the whole sequence and leak the future.
    """
    log_start = np.log(params["startprob"] + 1e-300)
    log_A = np.log(params["transmat"] + 1e-300)
    log_b = emission_loglik(params["means"], params["covars"], X)

    T, K = log_b.shape
    filtered = np.empty((T, K))
    log_alpha = log_start + log_b[0]
    log_alpha -= logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)
    for t in range(1, T):
        log_alpha = logsumexp(log_alpha[:, None] + log_A, axis=0) + log_b[t]
        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)
    return filtered


def viterbi_path(params: dict, X: np.ndarray) -> np.ndarray:
    """Global most-likely path (smoothing). For INTERPRETATION / PLOTS only."""
    log_start = np.log(params["startprob"] + 1e-300)
    log_A = np.log(params["transmat"] + 1e-300)
    log_b = emission_loglik(params["means"], params["covars"], X)
    T, K = log_b.shape
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_start + log_b[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_b[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# =============================================================================
# 5. METRICS & HONEST EVALUATION (D6)
# =============================================================================

def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def evaluate_regimes(params: dict, splits: Dict[str, pd.DataFrame],
                     regime: Dict[str, np.ndarray],
                     viterbi: Dict[str, np.ndarray]) -> dict:
    """Honest verdict metrics (D6)."""
    names = REGIME_NAMES
    K = len(names)
    diag = np.diag(params["transmat"])
    duration = 1.0 / (1.0 - np.clip(diag, 0, 0.999999))

    ev = {"regime_names": names, "transmat": params["transmat"].tolist(),
          "persistence_diag": diag.tolist(),
          "expected_duration_days": duration.tolist(),
          "mean_persistence": float(diag.mean()),
          "per_regime": {}, "per_split": {}, "regime_conditional_next_return": {},
          "causal_vs_viterbi_agreement": {}, "regime_strategy": {}}

    tr, r_tr = splits["train"], regime["train"]
    for i, nm in enumerate(names):
        m = r_tr == i
        rr = tr["log_return"].values[m]
        ev["per_regime"][nm] = {
            "n_days_train": int(m.sum()),
            "share_train": float(m.mean()),
            "mean_return_ann": float(rr.mean() * TRADING_DAYS) if m.sum() else None,
            "vol_ann": float(rr.std() * np.sqrt(TRADING_DAYS)) if m.sum() else None,
            "expected_duration_days": float(duration[i]),
        }

    for name, df in splits.items():
        r = regime[name]
        shares = {names[i]: float((r == i).mean()) for i in range(K)}
        ev["per_split"][name] = {
            "counts": {names[i]: int((r == i).sum()) for i in range(K)},
            "shares": shares, "min_share": float(min(shares.values()))}
        ev["causal_vs_viterbi_agreement"][name] = float((r == viterbi[name]).mean())

    for name, df in splits.items():
        r, y = regime[name], df[TARGET].values
        cond = {}
        for i, nm in enumerate(names):
            m = r == i
            cond[nm] = {"mean_next_return": float(y[m].mean()) if m.sum() else None,
                        "n": int(m.sum())}
        bull, bear = cond[names[-1]]["mean_next_return"], cond[names[0]]["mean_next_return"]
        cond["bull_minus_bear"] = (float(bull - bear)
                                   if bull is not None and bear is not None else None)
        ev["regime_conditional_next_return"][name] = cond

    for name, df in splits.items():
        r, y = regime[name], df[TARGET].values
        pos = np.zeros(len(r))
        pos[r == K - 1] = 1.0
        pos[r == 0] = -1.0
        prev = np.concatenate([[0.0], pos[:-1]])
        cost = (pos != prev).astype(float) * (TRANSACTION_COST_BPS / 1e4)
        strat = pos * y - cost
        ev["regime_strategy"][name] = {"sharpe": annualized_sharpe(strat),
                                       "max_drawdown": max_drawdown(strat),
                                       "n_trades": int((pos != prev).sum())}
    return ev


# =============================================================================
# 6. RUN ONE SPEC — fit, causal-decode, evaluate
# =============================================================================

def run_spec(name: str, obs_cols: List[str], splits: Dict[str, pd.DataFrame]) -> dict:
    """Fit a 3-state HMM on one observation set and decode regimes causally."""
    mean = splits["train"][obs_cols].mean().values
    std = splits["train"][obs_cols].std(ddof=0).values
    X = {nm: ((df[obs_cols].values - mean) / std) for nm, df in splits.items()}

    model, ll = fit_hmm_best(X["train"], N_STATES)
    params = reindex_model(model)

    X_full = np.vstack([X["train"], X["val"], X["test"]])
    filtered_full = causal_forward_filter(params, X_full)
    regime_full = filtered_full.argmax(axis=1)
    viterbi_full = viterbi_path(params, X_full)

    n_tr, n_va = len(X["train"]), len(X["val"])
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va),
              "test": (n_tr + n_va, len(X_full))}
    regime = {s: regime_full[a:b] for s, (a, b) in bounds.items()}
    viterbi = {s: viterbi_full[a:b] for s, (a, b) in bounds.items()}
    filtered = {s: filtered_full[a:b] for s, (a, b) in bounds.items()}

    ev = evaluate_regimes(params, splits, regime, viterbi)
    return {"name": name, "obs_cols": obs_cols, "obs_mean": mean, "obs_std": std,
            "model": model, "params": params, "loglik": ll,
            "bic": float(model.bic(X["train"])), "aic": float(model.aic(X["train"])),
            "X": X, "regime": regime, "viterbi": viterbi, "filtered": filtered,
            "regime_full": regime_full, "ev": ev}


def select_winner(results: Dict[str, dict]) -> Tuple[str, str, list]:
    """
    Select the winning spec on TRAIN+VAL criteria only (D2). TEST is NOT used.
    Eligible: persistence >= 0.85. Winner: consistent positive Bull-Bear next-day
    spread on train AND val, ranked by val spread. Else fall back to S1.
    """
    rows = []
    for nm, res in results.items():
        ev = res["ev"]
        persist = ev["mean_persistence"]
        tr = ev["regime_conditional_next_return"]["train"]["bull_minus_bear"]
        va = ev["regime_conditional_next_return"]["val"]["bull_minus_bear"]
        eligible = persist >= PERSISTENCE_FLOOR
        directional = eligible and tr is not None and va is not None and tr > 0 and va > 0
        rows.append({"spec": nm, "obs": "+".join(res["obs_cols"]),
                     "persistence": persist, "bic": res["bic"],
                     "train_spread": tr, "val_spread": va,
                     "eligible": eligible, "directional_train_val": directional})
    qualified = [r for r in rows if r["directional_train_val"]]
    if qualified:
        winner = max(qualified, key=lambda r: r["val_spread"])["spec"]
        reason = ("consistent positive Bull-Bear next-day spread on TRAIN and VAL "
                  "(selected on val spread; TEST held out)")
    else:
        winner = FALLBACK_SPEC
        reason = ("NO spec shows a consistent TRAIN+VAL directional signal -> keep "
                  "the volatility spec S1; verdict remains DESCRIPTIVE_ONLY")
    return winner, reason, rows


# =============================================================================
# 7. PLOTS
# =============================================================================

def plot_regime_timeline(close, regime_full, full_index, path):
    fig, ax = plt.subplots(figsize=(14, 5))
    s = pd.Series(regime_full, index=full_index)
    c = close.reindex(s.index).ffill()
    ax.plot(c.index, c.values, color="black", linewidth=0.7, zorder=3)
    for i, nm in enumerate(REGIME_NAMES):
        ax.fill_between(s.index, c.min(), c.max(), where=(s.values == i),
                        color=REGIME_COLORS[nm], alpha=0.30, step="mid")
    ax.set_title("Étape 4 — MASI close shaded by CAUSAL HMM regime (winner spec)")
    ax.set_ylabel("MASI close")
    ax.set_xlim(s.index.min(), s.index.max())
    ax.legend(handles=[Patch(facecolor=REGIME_COLORS[nm], alpha=0.5, label=nm)
                       for nm in REGIME_NAMES], loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_transition_matrix(transmat, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transmat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(REGIME_NAMES); ax.set_yticklabels(REGIME_NAMES)
    ax.set_xlabel("to regime"); ax.set_ylabel("from regime")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{transmat[i, j]:.3f}", ha="center", va="center",
                    color="white" if transmat[i, j] > 0.5 else "black", fontsize=10)
    ax.set_title("Étape 4 — HMM transition matrix (winner spec)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_regime_characteristics(ev, path):
    names = REGIME_NAMES
    colors = [REGIME_COLORS[n] for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].bar(names, [ev["per_regime"][n]["mean_return_ann"] for n in names], color=colors)
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_title("Mean return per regime (annualized, TRAIN)")
    axes[0, 1].bar(names, [ev["per_regime"][n]["vol_ann"] for n in names], color=colors)
    axes[0, 1].set_title("Volatility per regime (annualized, TRAIN)")
    axes[1, 0].bar(names, [ev["per_regime"][n]["expected_duration_days"] for n in names],
                   color=colors)
    axes[1, 0].set_title("Expected regime duration (days)")
    x = np.arange(3)
    for k, sp in enumerate(("train", "val", "test")):
        cond = ev["regime_conditional_next_return"][sp]
        axes[1, 1].bar(x + (k - 1) * 0.25, [cond[n]["mean_next_return"] for n in names],
                       0.25, label=sp)
    axes[1, 1].axhline(0, color="black", lw=0.6)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_title("Regime-conditional NEXT-DAY return (predictive test)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Étape 4 — HMM regime characteristics (winner spec)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_spec_comparison(results, rows, winner, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specs = list(results.keys())
    x = np.arange(len(specs))
    for k, sp in enumerate(("train", "val", "test")):
        vals = [results[s]["ev"]["regime_conditional_next_return"][sp]["bull_minus_bear"]
                for s in specs]
        axes[0].bar(x + (k - 1) * 0.25, vals, 0.25, label=sp)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(specs, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Bull-minus-Bear next-day return spread per spec\n"
                      "(consistent positive train+val = directional)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    persist = [results[s]["ev"]["mean_persistence"] for s in specs]
    bars = axes[1].bar(x, persist, color=["#2ca02c" if s == winner else "#7f9fbf"
                                          for s in specs])
    axes[1].axhline(PERSISTENCE_FLOOR, color="r", ls="--", lw=0.8, label="floor 0.85")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(specs, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0.7, 1.0)
    axes[1].set_title(f"Regime persistence per spec (winner = {winner})")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_model_selection(nstates_table, ev, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ks = sorted(nstates_table.keys())
    axes[0].plot(ks, [nstates_table[k]["bic"] for k in ks], "o-", label="BIC")
    axes[0].plot(ks, [nstates_table[k]["aic"] for k in ks], "s-", label="AIC")
    axes[0].set_xticks(ks); axes[0].set_xlabel("n_states")
    axes[0].set_title("HMM n_states selection — winner spec (lower = better)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    splits = ["train", "val", "test"]
    bottom = np.zeros(len(splits))
    for i, nm in enumerate(REGIME_NAMES):
        vals = [ev["per_split"][s]["shares"][nm] for s in splits]
        axes[1].bar(splits, vals, bottom=bottom, color=REGIME_COLORS[nm], label=nm)
        bottom += np.array(vals)
    axes[1].set_title("Regime coverage per split (winner spec)")
    axes[1].set_ylabel("share of days"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# 8. VERDICT
# =============================================================================

def print_verdict(ev: dict) -> dict:
    persistence = ev["mean_persistence"]
    sp_tr = ev["regime_conditional_next_return"]["train"]["bull_minus_bear"]
    sp_te = ev["regime_conditional_next_return"]["test"]["bull_minus_bear"]
    min_cov = ev["per_split"]["test"]["min_share"]
    sharpe_te = ev["regime_strategy"]["test"]["sharpe"]
    sharpe_tr = ev["regime_strategy"]["train"]["sharpe"]

    chk_persist = persistence >= PERSISTENCE_FLOOR
    chk_predict = sp_tr is not None and sp_te is not None and sp_tr > 0 and sp_te > 0
    chk_coverage = min_cov >= 0.05

    print("\n  " + "-" * 68)
    print("  HONEST VERDICT (D6)")
    print("  " + "-" * 68)
    print(f"  [{'PASS' if chk_persist else 'WEAK'}] Persistence       : mean diag = {persistence:.3f}")
    print(f"  [{'PASS' if chk_predict else 'WEAK'}] Predictive content: Bull-Bear next-day "
          f"train={sp_tr:+.6f}  test={sp_te:+.6f}")
    print(f"  [{'PASS' if chk_coverage else 'WEAK'}] OOS coverage      : rarest TEST regime "
          f"{min_cov*100:.1f}%")
    print(f"  [info ] regime strategy Sharpe  train={sharpe_tr:+.3f}  test={sharpe_te:+.3f}")

    if chk_persist and chk_predict and chk_coverage:
        overall = "USABLE"
    elif chk_persist and chk_coverage:
        overall = "DESCRIPTIVE_ONLY"
    else:
        overall = "WEAK"
    print(f"\n  OVERALL = {overall}")
    print("  " + "-" * 68)
    return {"persistence_ok": bool(chk_persist), "predictive_ok": bool(chk_predict),
            "coverage_ok": bool(chk_coverage), "overall": overall}


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("ÉTAPE 4 — HMM REGIME DETECTION  (v2 — comparative observation study)")
    print("=" * 78)

    print("\n[1] Loading Étape 3 features ...")
    splits = load_features()
    close = load_masi_close()

    # --- 2. Comparative spec study (D1) ------------------------------------
    print(f"\n[2] Fitting {len(SPECS)} HMM observation specs (12 seeds each, 3 states) ...")
    results = {}
    for nm, cols in SPECS.items():
        res = run_spec(nm, cols, splits)
        ev = res["ev"]
        results[nm] = res
        sp_tr = ev["regime_conditional_next_return"]["train"]["bull_minus_bear"]
        sp_va = ev["regime_conditional_next_return"]["val"]["bull_minus_bear"]
        sp_te = ev["regime_conditional_next_return"]["test"]["bull_minus_bear"]
        print(f"    {nm:20s} obs={'+'.join(cols):28s} "
              f"persist={ev['mean_persistence']:.3f}  BIC={res['bic']:9.1f}")
        print(f"    {'':20s} Bull-Bear next-day spread  "
              f"train={sp_tr:+.6f}  val={sp_va:+.6f}  test={sp_te:+.6f}")

    # --- 3. Select winner on TRAIN+VAL only (D2) ---------------------------
    print("\n[3] Selecting winner — TRAIN+VAL criteria, TEST held out ...")
    winner, reason, comp_rows = select_winner(results)
    print(f"    WINNER = {winner}   ({'+'.join(SPECS[winner])})")
    print(f"    reason : {reason}")
    win = results[winner]
    ev = win["ev"]

    # --- 4. n_states selection for the winner spec -------------------------
    print(f"\n[4] BIC/AIC over n_states for the winner spec ...")
    obs_cols = SPECS[winner]
    mean = splits["train"][obs_cols].mean().values
    std = splits["train"][obs_cols].std(ddof=0).values
    Xtr = (splits["train"][obs_cols].values - mean) / std
    nstates_table = {}
    for k in N_STATES_GRID:
        m, ll = fit_hmm_best(Xtr, k)
        nstates_table[k] = {"loglik": ll, "bic": float(m.bic(Xtr)),
                            "aic": float(m.aic(Xtr))}
        print(f"    n_states={k}: logL={ll:9.1f}  BIC={nstates_table[k]['bic']:9.1f}  "
              f"AIC={nstates_table[k]['aic']:9.1f}")
    bic_best = min(nstates_table, key=lambda k: nstates_table[k]["bic"])
    print(f"    BIC prefers n_states={bic_best}; project keeps {N_STATES} (interpretability).")

    # --- 5. Verdict for the winner -----------------------------------------
    print(f"\n[5] Winner spec evaluation ...")
    for nm in ("train", "val", "test"):
        print(f"    causal-vs-Viterbi agreement [{nm}]: "
              f"{ev['causal_vs_viterbi_agreement'][nm]*100:.1f}%")
    verdict = print_verdict(ev)

    # --- 6. Persist artifacts (winner only) --------------------------------
    print("\n[6] Writing artifacts (winner spec) ...")
    names = REGIME_NAMES
    onehot_cols = [f"regime_{n.lower()}" for n in names]
    prob_cols = [f"regime_prob_{n.lower()}" for n in names]
    for nm, df in splits.items():
        out = df.copy()
        r = win["regime"][nm]
        out["regime"] = r
        out["regime_name"] = [names[i] for i in r]
        oh = np.eye(N_STATES, dtype=int)[r]
        for j, c in enumerate(onehot_cols):
            out[c] = oh[:, j]
        for j, c in enumerate(prob_cols):
            out[c] = win["filtered"][nm][:, j]
        p = os.path.join(REG_DIR, f"masi_regimes_{nm}.csv")
        out.to_csv(p)
        print(f"    {p}")

    hmm_params = {
        "winner_spec": winner, "obs_columns": obs_cols, "n_states": N_STATES,
        "regime_names": names, "selection_reason": reason,
        "obs_scaler": {"mean": win["obs_mean"].tolist(), "std": win["obs_std"].tolist()},
        "startprob": win["params"]["startprob"].tolist(),
        "transmat": win["params"]["transmat"].tolist(),
        "means_standardized": win["params"]["means"].tolist(),
        "means_original_scale": (win["params"]["means"] * win["obs_std"]
                                 + win["obs_mean"]).tolist(),
        "covars_standardized": win["params"]["covars"].tolist(),
        "train_loglik": win["loglik"], "bic": win["bic"],
        "onehot_columns": onehot_cols, "prob_columns": prob_cols,
        "nstates_selection": nstates_table, "bic_preferred_n_states": bic_best,
        "leakage_note": "regime feature = CAUSAL forward-filter; Viterbi NOT a feature.",
    }
    with open(os.path.join(REG_DIR, "hmm_params.json"), "w") as f:
        json.dump(hmm_params, f, indent=2)
    with open(os.path.join(REG_DIR, "hmm_obs_scaler_train_only.json"), "w") as f:
        json.dump({"obs_columns": obs_cols, "mean": win["obs_mean"].tolist(),
                   "std": win["obs_std"].tolist()}, f, indent=2)
    ev_out = dict(ev)
    ev_out["verdict"] = verdict
    ev_out["winner_spec"] = winner
    with open(os.path.join(REG_DIR, "regime_evaluation.json"), "w") as f:
        json.dump(ev_out, f, indent=2)
    comparison = {"winner": winner, "selection_reason": reason, "specs": comp_rows,
                  "per_spec_detail": {nm: {
                      "obs_cols": results[nm]["obs_cols"],
                      "mean_persistence": results[nm]["ev"]["mean_persistence"],
                      "bic": results[nm]["bic"],
                      "bull_minus_bear": {s: results[nm]["ev"][
                          "regime_conditional_next_return"][s]["bull_minus_bear"]
                          for s in ("train", "val", "test")},
                      "strategy_sharpe": {s: results[nm]["ev"]["regime_strategy"][s]["sharpe"]
                                          for s in ("train", "val", "test")},
                  } for nm in SPECS}}
    with open(os.path.join(REG_DIR, "spec_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"    {os.path.join(REG_DIR, 'spec_comparison.json')}")

    # --- 7. Plots ----------------------------------------------------------
    print("\n[7] Generating diagnostic plots ...")
    full_index = (splits["train"].index.append(splits["val"].index)
                  .append(splits["test"].index))
    plot_regime_timeline(close, win["regime_full"], full_index,
                         os.path.join(PLOTS_DIR, "etape4_regime_timeline.png"))
    plot_transition_matrix(win["params"]["transmat"],
                           os.path.join(PLOTS_DIR, "etape4_transition_matrix.png"))
    plot_regime_characteristics(ev, os.path.join(PLOTS_DIR, "etape4_regime_characteristics.png"))
    plot_spec_comparison(results, comp_rows, winner,
                         os.path.join(PLOTS_DIR, "etape4_spec_comparison.png"))
    plot_model_selection(nstates_table, ev,
                         os.path.join(PLOTS_DIR, "etape4_model_selection.png"))
    print(f"    5 plots -> {PLOTS_DIR}")

    print("\n" + "=" * 78)
    print(f"ÉTAPE 4 COMPLETE — winner = {winner}, verdict = {verdict['overall']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
