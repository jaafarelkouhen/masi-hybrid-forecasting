"""
backtest command — applique une stratégie aux prédictions + couche risque,
calcule toutes les métriques mémoire (Sharpe, Sortino, MDD, Calmar, DSR,
regime-conditional), affiche et écrit en JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import metrics as M
from . import strategies as strat
from .config import (
    AVAILABLE_STRATEGIES,
    RISK_METRICS_CSV,
)
from .predict import load_predictions_with_dates

logger = logging.getLogger(__name__)


# Stratégies qui nécessitent la couche risque (VaR, risk_regime)
NEEDS_RISK = {"risk_gate", "hmm_risk_gate", "var_budget", "hmm_cond_budget"}


def _load_dataframe(strategy: str) -> pd.DataFrame:
    """Charge prédictions + (si nécessaire) la couche risque."""
    df = load_predictions_with_dates()
    if strategy in NEEDS_RISK:
        if not RISK_METRICS_CSV.exists():
            raise FileNotFoundError(
                f"Couche risque introuvable : {RISK_METRICS_CSV}\n"
                f"Lance `python -m masi_hybrid_forecasting.pipeline risk` d'abord."
            )
        risk_df = pd.read_csv(RISK_METRICS_CSV, parse_dates=["date"])
        risk_cols = ["var_param_5", "risk_regime"]
        df = df.merge(risk_df[["date"] + risk_cols], on="date", how="left")
    return df


def run(args) -> None:
    strategy = args.strategy
    cost_bps = float(args.cost_bps)
    cost_dec = cost_bps / 10_000

    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy!r}. "
                         f"Choisir parmi {AVAILABLE_STRATEGIES}")

    logger.info(f"Backtest stratégie={strategy} @ {cost_bps} bps")

    df = _load_dataframe(strategy)
    logger.info(f"  TEST = {len(df)} jours, "
                f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    # Build positions
    positions, mode = strat.build(strategy, df)
    rets = strat.strategy_returns(positions, df["actual_return"].values,
                                    mode=mode, cost_dec=cost_dec)

    # Compute full metrics
    m = M.compute_full_metrics(rets, positions, mode=mode,
                                regime_names=df["regime_name"].values)

    # DSR (single, vs SR0 du panel 7 stratégies — étape 8 figé)
    # SR0 ≈ 0.021 (étape 8 V=2.3e-4, N=7). Pour single backtest on l'utilise comme référence.
    dsr = M.deflated_sharpe_single(rets, sr0_threshold=0.021)

    out = {
        "strategy": strategy,
        "mode": mode,
        "cost_bps": cost_bps,
        "n_days": int(len(df)),
        "test_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "metrics": m,
        "deflated_sharpe": dsr,
    }

    # Print human-friendly
    print(f"\n{'='*78}")
    print(f"BACKTEST {strategy.upper()}  @ {cost_bps} bps  ({len(df)} jours)")
    print(f"{'='*78}")
    print(f"  Sharpe ann.     : {m['sharpe']:+8.3f}")
    print(f"  Sortino ann.    : {m['sortino']:+8.3f}")
    print(f"  Max Drawdown    : {m['max_drawdown']:+8.2%}")
    print(f"  Calmar          : {m['calmar']:+8.2f}")
    print(f"  Equity finale   : {m['final_equity']:8.3f}")
    print(f"  Trades          : {m['n_trades']:8d}")
    print(f"  % jours actifs  : {m['pct_days_active']*100:7.1f}%")
    print(f"  Expo. moyenne   : {m['avg_abs_exposure']:8.3f}")
    print(f"  DSR (vs SR0)    : {dsr['deflated_sharpe_psr_vs_sr0']:7.3f}")
    print(f"  PSR (vs 0)      : {dsr['psr_vs_zero']:7.3f}")
    if "regime_conditional" in m:
        print("\n  Sharpe régime-conditionnel :")
        for reg, rd in m["regime_conditional"].items():
            print(f"    {reg:<8s} (n={rd['n']:>3}) : {rd['sharpe']:+.2f}")
    print("=" * 78)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"✓ Métriques écrites : {output_path}")
