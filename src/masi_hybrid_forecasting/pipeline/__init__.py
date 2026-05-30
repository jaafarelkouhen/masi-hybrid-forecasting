"""
masi_hybrid_forecasting.pipeline — Hybrid forecasting pipeline for the MASI index.

Production stack (étapes 0-10) :
  - CNN-LSTM base12 (étape 5) for next-day log-return prediction
  - HMM-gate (étape 8/9) for regime-conditional signal activation
  - Risk layer (étape 7) : VaR, ES, GARCH vol, risk regime — défense alternative
  - 5 CLI commands : train, predict, backtest, risk, export

Usage:
  python -m masi_hybrid_forecasting.pipeline <command> [options]
  masi-pipeline <command> [options]

See: README.md
"""

__version__ = "1.0.0"
__author__ = "MASI Research Pipeline — Mémoire 2026"
__all__ = ["cli", "config", "strategies", "risk", "metrics",
           "train", "predict", "backtest", "export", "forecast"]
