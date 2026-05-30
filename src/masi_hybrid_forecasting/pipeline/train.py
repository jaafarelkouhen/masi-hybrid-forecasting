"""
train command — wrapper qui (re)lance le walk-forward CNN-LSTM de l'étape 5.

Note méthodo : le pipeline production n'entraîne PAS un modèle unique. Il
exécute le walk-forward 5 folds défini en étape 5 (`scripts/05_cnn_lstm.py`).
Chaque fold (re)fit le modèle sur fenêtre glissante TRAIN+VAL avant prédiction
TEST. Cette commande relance cette procédure complète.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from .config import ETAPE5_SCRIPT, PREDICTIONS_CSV

logger = logging.getLogger(__name__)


def run(args) -> None:
    if not ETAPE5_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script étape 5 introuvable : {ETAPE5_SCRIPT}\n"
            f"Vérifie que tu lances la CLI depuis la racine du projet."
        )

    logger.info("Lancement du walk-forward CNN-LSTM (étape 5) ...")
    logger.info(f"  Script : {ETAPE5_SCRIPT}")
    logger.info(f"  Sortie attendue : {PREDICTIONS_CSV}")
    logger.warning("Runtime ≈ 10-15 min (5 folds, ~5000 epochs total)")

    result = subprocess.run(
        [sys.executable, str(ETAPE5_SCRIPT)],
        cwd=str(ETAPE5_SCRIPT.parent.parent),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Le script étape 5 a échoué (exit code {result.returncode}). "
            f"Voir la sortie ci-dessus."
        )

    if not PREDICTIONS_CSV.exists():
        raise RuntimeError(
            f"Le walk-forward a fini sans erreur mais le fichier "
            f"{PREDICTIONS_CSV} n'existe pas. Vérifie scripts/05_cnn_lstm.py."
        )

    logger.info(f"✓ Entraînement terminé. Prédictions écrites : {PREDICTIONS_CSV}")
