"""
CLI entry point : masi-pipeline <command> [options]

5 sous-commandes :
  train     entraîne CNN-LSTM walk-forward (étape 5)
  predict   charge / produit les prédictions TEST + dates
  backtest  applique une stratégie + métriques mémoire (Sharpe, DSR, ...)
  risk      génère la couche risque (VaR, ES, vol, régime)
  export    exporte le fichier canonique pour API/dashboard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import AVAILABLE_STRATEGIES, COST_BPS, PRODUCTION_STRATEGY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="masi-pipeline",
        description=(
            "MASI Hybrid Forecasting Pipeline — étapes 0-10. "
            "Stratégie production = CNN-LSTM base12 + HMM-gate (DSR 0.997, "
            "robuste sur 5 axes étape 9)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  masi-pipeline predict
  masi-pipeline risk
  masi-pipeline backtest --strategy hmm_gate --cost-bps 5
  masi-pipeline backtest --strategy cnn_lstm_nu --cost-bps 20
  masi-pipeline export --strategy hmm_gate --output canonical.csv
  masi-pipeline train          # /!\\ relance walk-forward étape 5 (10-15 min)
        """,
    )
    parser.add_argument("--version", action="version", version=f"masi-pipeline {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="logs DEBUG au lieu de INFO")

    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # train
    p_train = sub.add_parser("train",
                              help="entraîne CNN-LSTM walk-forward (relance étape 5)")
    p_train.set_defaults(_module="train")

    # predict
    p_pred = sub.add_parser("predict",
                             help="charge/affiche les prédictions TEST")
    p_pred.add_argument("--output", type=Path, default=None,
                         help="écrit le CSV joint (date + y_true + y_pred + régime)")
    p_pred.set_defaults(_module="predict")

    # backtest
    p_bt = sub.add_parser("backtest",
                           help="backtest stratégie + métriques mémoire")
    p_bt.add_argument("--strategy", choices=AVAILABLE_STRATEGIES,
                       default=PRODUCTION_STRATEGY,
                       help=f"stratégie (defaut: {PRODUCTION_STRATEGY} = production étape 9)")
    p_bt.add_argument("--cost-bps", type=float, default=COST_BPS,
                       help=f"coût one-way en bps (defaut: {COST_BPS})")
    p_bt.add_argument("--output", type=Path, default=None,
                       help="écrit JSON métriques complètes")
    p_bt.set_defaults(_module="backtest")

    # risk
    p_risk = sub.add_parser("risk",
                             help="génère la couche risque (VaR/ES/vol/régime)")
    p_risk.add_argument("--output", type=Path, default=None,
                         help="path CSV de sortie (defaut: étape 7 canonique)")
    p_risk.set_defaults(_module="risk")

    # export
    p_exp = sub.add_parser("export",
                            help="exporte le fichier canonique pour API/dashboard")
    p_exp.add_argument("--strategy", choices=AVAILABLE_STRATEGIES,
                        default=PRODUCTION_STRATEGY,
                        help=f"stratégie (defaut: {PRODUCTION_STRATEGY})")
    p_exp.add_argument("--output", type=Path, default=None,
                        help="path CSV final (defaut: etape6_final_predictions.csv)")
    p_exp.set_defaults(_module="export")

    # forecast
    p_fc = sub.add_parser("forecast",
                           help="projection hors-échantillon ARIMA + Monte-Carlo + drift régime")
    p_fc.add_argument("--year", type=int, default=2026,
                       help="année cible (defaut: 2026)")
    p_fc.add_argument("--months", type=str, default="5,6",
                       help="liste des mois cibles, séparés par virgule (defaut: 5,6)")
    p_fc.add_argument("--n-mc", dest="n_mc", type=int, default=5000,
                       help="nombre de trajectoires Monte-Carlo (defaut: 5000)")
    p_fc.add_argument("--output-dir", dest="output_dir", type=Path, default=None,
                       help="répertoire de sortie (defaut: outputs/etape10/)")
    p_fc.set_defaults(_module="forecast")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s : %(message)s",
        datefmt="%H:%M:%S",
    )

    # Dispatch
    try:
        if args._module == "train":
            from . import train as mod
        elif args._module == "predict":
            from . import predict as mod
        elif args._module == "backtest":
            from . import backtest as mod
        elif args._module == "risk":
            from . import risk as mod
        elif args._module == "export":
            from . import export as mod
        elif args._module == "forecast":
            from . import forecast as mod
        else:
            parser.print_help()
            return 1
        mod.run(args)
        return 0
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[ERREUR] {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"\n[ERREUR RUNTIME] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
