"""
export command — produit le fichier canonique unique pour API/dashboard.

Colonnes : date, actual_return, predicted_return, signal, regime, regime_name,
            risk_regime, strategy_return (de la stratégie sélectionnée),
            equity (cumul exp(sum(strategy_return))).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as strat
from .config import (
    CANONICAL_CSV,
    COST_DEC,
    PRODUCTION_STRATEGY,
    RISK_METRICS_CSV,
)
from .predict import load_predictions_with_dates

logger = logging.getLogger(__name__)


def run(args) -> None:
    strategy = args.strategy if hasattr(args, "strategy") else PRODUCTION_STRATEGY
    output_path = Path(args.output) if args.output else CANONICAL_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Export fichier canonique  stratégie={strategy}  → {output_path}")

    df = load_predictions_with_dates()
    # Charger risque (toujours utile pour l'API même si stratégie ne l'utilise pas)
    if RISK_METRICS_CSV.exists():
        risk_df = pd.read_csv(RISK_METRICS_CSV, parse_dates=["date"])
        df = df.merge(risk_df[["date", "var_param_5", "es_param_5",
                                 "vol_garch", "risk_regime"]],
                       on="date", how="left")
    else:
        logger.warning(f"Couche risque absente ({RISK_METRICS_CSV.name}). "
                       f"Colonnes VaR/risk_regime manquantes dans l'export.")

    # Build positions + returns pour la stratégie
    positions, mode = strat.build(strategy, df)
    rets = strat.strategy_returns(positions, df["actual_return"].values,
                                    mode=mode, cost_dec=COST_DEC)

    out = pd.DataFrame({
        "date": df["date"].values,
        "actual_return": df["actual_return"].values,
        "predicted_return": df["predicted_return"].values,
        "signal_raw": np.sign(df["predicted_return"]).astype(int),
        "regime": df["regime"].values,
        "regime_name": df["regime_name"].values,
        "position": positions,
        "strategy_return": rets,
        "equity": np.exp(np.cumsum(rets)),
    })
    if "risk_regime" in df.columns:
        out["risk_regime"] = df["risk_regime"].values
        out["var_param_5"] = df["var_param_5"].values
        out["es_param_5"] = df["es_param_5"].values
        out["vol_garch"] = df["vol_garch"].values

    out["strategy_name"] = strategy
    out["mode"] = mode
    out["cost_bps"] = (COST_DEC * 10_000)

    out.to_csv(output_path, index=False)
    final_eq = float(out["equity"].iloc[-1])
    logger.info(f"✓ Export écrit : {output_path}")
    logger.info(f"  {len(out)} lignes, {len(out.columns)} colonnes")
    logger.info(f"  Stratégie '{strategy}' : equity finale = {final_eq:.4f}")
