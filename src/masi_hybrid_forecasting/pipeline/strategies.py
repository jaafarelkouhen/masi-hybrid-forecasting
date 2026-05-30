"""
Strategies library — les 7 stratégies évaluées étapes 8-9.

Pour chaque stratégie : `build_<name>(df)` renvoie un vecteur de positions
day-by-day. `strategy_returns(positions, y_true, mode)` calcule les retours
avec les coûts cohérents avec étape 6/8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AVAILABLE_STRATEGIES,
    B_STD,
    COST_DEC,
    EPS_VAR,
    Q_NORMAL,
    Q_STRESS,
    ROLL_BUDGET,
)


# ============================================================================
# RETOURS STRATÉGIE (binary vs continuous)
# ============================================================================
def strategy_returns(positions: np.ndarray, y_true: np.ndarray,
                     mode: str = "binary", cost_dec: float = COST_DEC) -> np.ndarray:
    """
    Calcule strategy_return = positions * y_true − cost.
      - mode="binary" : one-way cost × |Δposition| (flip -1→+1 costs 2 units)
      - mode="continuous" : cost × |Δpos| (turnover proportionnel)
    """
    positions = np.asarray(positions, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    prev = np.concatenate([[0.0], positions[:-1]])
    if mode == "binary" or mode == "continuous":
        c = cost_dec * np.abs(positions - prev)
    else:
        raise ValueError(f"mode inconnu : {mode!r} ; attendu 'binary' ou 'continuous'")
    return positions * y_true - c


def signal_from_prediction(y_pred: np.ndarray) -> np.ndarray:
    """signal = sign(y_pred) ∈ {-1, 0, +1}."""
    return np.sign(y_pred).astype(int)


# ============================================================================
# CONSTRUCTORS DES 7 STRATÉGIES
# ============================================================================
def build_buy_hold(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    return np.ones(len(df)), "binary"


def build_cnn_lstm_nu(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    return signal_from_prediction(df["predicted_return"].values).astype(float), "binary"


def build_hmm_gate(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Trade seulement si HMM regime ∈ {Bear, Bull}."""
    signal = signal_from_prediction(df["predicted_return"].values)
    pos = np.where(np.isin(df["regime_name"].values, ["Bear", "Bull"]), signal, 0).astype(float)
    return pos, "binary"


def build_risk_gate(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Trade seulement si risk_regime ≠ high (σ_GARCH q67-)."""
    _require(df, ["risk_regime"], "risk_gate")
    signal = signal_from_prediction(df["predicted_return"].values)
    pos = np.where(df["risk_regime"].values != "high", signal, 0).astype(float)
    return pos, "binary"


def build_hmm_risk_gate(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Trade si HMM ∈ {Bear, Bull} AND risk_regime ≠ high."""
    _require(df, ["risk_regime"], "hmm_risk_gate")
    signal = signal_from_prediction(df["predicted_return"].values)
    pos = np.where(
        np.isin(df["regime_name"].values, ["Bear", "Bull"]) &
        (df["risk_regime"].values != "high"),
        signal, 0
    ).astype(float)
    return pos, "binary"


def build_var_budget(df: pd.DataFrame, B: float = B_STD) -> tuple[np.ndarray, str]:
    """Position continue w = sign(ŷ) · min(1, B/|VaR|). Inspiré repo source."""
    _require(df, ["var_param_5"], "var_budget")
    signal = signal_from_prediction(df["predicted_return"].values).astype(float)
    var_abs = np.abs(df["var_param_5"].values)
    w_raw = np.minimum(1.0, B / np.maximum(var_abs, EPS_VAR))
    return signal * w_raw, "continuous"


def build_hmm_cond_budget(df: pd.DataFrame,
                           rolling_win: int = ROLL_BUDGET,
                           q_stress: float = Q_STRESS,
                           q_normal: float = Q_NORMAL) -> tuple[np.ndarray, str]:
    """
    Position continue w = sign(ŷ) · min(1, B_t/|VaR|) où B_t varie selon le
    régime HMM (Neutral=stress → q30 ; sinon → q70). Inspiré repo source.
    """
    _require(df, ["var_param_5"], "hmm_cond_budget")
    signal = signal_from_prediction(df["predicted_return"].values).astype(float)
    var_abs = np.abs(df["var_param_5"].values)
    var_series = pd.Series(var_abs)
    shifted = var_series.shift(1)
    q_s = shifted.rolling(rolling_win, min_periods=rolling_win).quantile(q_stress)
    q_n = shifted.rolling(rolling_win, min_periods=rolling_win).quantile(q_normal)
    is_stress = (df["regime_name"].values == "Neutral")
    B_t = np.where(is_stress, q_s.values, q_n.values)
    B_t = np.where(np.isnan(B_t), B_STD, B_t)
    w_t = np.minimum(1.0, B_t / np.maximum(var_abs, EPS_VAR))
    return signal * w_t, "continuous"


# ============================================================================
# DISPATCH
# ============================================================================
_BUILDERS = {
    "buy_hold": build_buy_hold,
    "cnn_lstm_nu": build_cnn_lstm_nu,
    "hmm_gate": build_hmm_gate,
    "risk_gate": build_risk_gate,
    "hmm_risk_gate": build_hmm_risk_gate,
    "var_budget": build_var_budget,
    "hmm_cond_budget": build_hmm_cond_budget,
}


def build(name: str, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    Construit la stratégie `name` à partir du dataframe `df`.
    Renvoie (positions, mode) où mode ∈ {"binary", "continuous"}.

    Raise:
      ValueError si name inconnu
      KeyError si df manque les colonnes requises
    """
    if name not in _BUILDERS:
        raise ValueError(f"Stratégie inconnue : {name!r}. "
                         f"Choisir parmi {AVAILABLE_STRATEGIES}")
    return _BUILDERS[name](df)


# ============================================================================
# HELPERS
# ============================================================================
def _require(df: pd.DataFrame, cols: list[str], strategy: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colonnes manquantes pour stratégie '{strategy}' : {missing}. "
            f"Lance `python -m masi_hybrid_forecasting.pipeline risk` d'abord "
            f"pour générer les VaR/régime."
        )
