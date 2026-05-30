"""
predict command — charge les prédictions TEST CNN-LSTM (walk-forward étape 5)
et les expose proprement avec dates (jointure avec les régimes étape 4).

Si le fichier predictions_test.csv n'existe pas → invite à lancer `train`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import PREDICTIONS_CSV, REGIMES_CSV

logger = logging.getLogger(__name__)


def load_predictions_with_dates() -> pd.DataFrame:
    """Charge predictions_test.csv + REGIMES (pour les dates). Renvoie df aligné."""
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Prédictions introuvables : {PREDICTIONS_CSV}\n"
            f"Lance `python -m masi_hybrid_forecasting.pipeline train` d'abord."
        )
    if not REGIMES_CSV.exists():
        raise FileNotFoundError(
            f"Fichier régimes introuvable : {REGIMES_CSV}\n"
            f"Lance d'abord l'étape 4 (scripts/04_hmm_regimes.py)."
        )

    preds = pd.read_csv(PREDICTIONS_CSV)
    regs = pd.read_csv(REGIMES_CSV, parse_dates=["date"])

    if len(preds) != len(regs):
        raise ValueError(
            f"Désalignement : predictions={len(preds)} vs régimes={len(regs)}. "
            f"Régénère l'un ou l'autre."
        )

    df = pd.DataFrame({
        "date": regs["date"].values,
        "actual_return": preds["y_true"].values,
        "predicted_return": preds["y_pred"].values,
        "regime": regs["regime"].astype(int).values,
        "regime_name": regs["regime_name"].values,
    })
    return df


def run(args) -> None:
    df = load_predictions_with_dates()
    logger.info(f"Prédictions chargées : {len(df)} jours TEST")
    logger.info(f"  Période : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    logger.info(f"  Couverture régimes : "
                f"{df['regime_name'].value_counts().to_dict()}")

    output = Path(args.output) if getattr(args, "output", None) else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info(f"✓ Prédictions écrites : {output}")
    else:
        # Affichage compact
        print("\n=== HEAD ===")
        print(df.head().to_string(index=False))
        print("\n=== TAIL ===")
        print(df.tail().to_string(index=False))
        print("\nStats prédictions :")
        print(f"  mean(y_pred) = {df['predicted_return'].mean():+.6f}")
        print(f"  std(y_pred)  = {df['predicted_return'].std():.6f}")
        print(f"  signe +/-/0  : {int((df['predicted_return']>0).sum())}/"
              f"{int((df['predicted_return']<0).sum())}/"
              f"{int((df['predicted_return']==0).sum())}")
