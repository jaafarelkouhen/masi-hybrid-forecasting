"""
Default paths and trading constants for the MASI pipeline.

Toutes les valeurs sont alignées avec les étapes 0-9 (prompt.md). Pour
override, passer les arguments CLI explicites.
"""

from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================
# Layout (nouveau, après migration vers masi-hybrid-forecasting/) :
#   PROJECT_ROOT = .../masi-hybrid-forecasting/
#   ├── src/masi_hybrid_forecasting/pipeline/   <- PIPELINE_ROOT (ce fichier)
#   ├── data/  outputs/  scripts/  reports/figures/  ...
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent.parent   # masi-hybrid-forecasting/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Inputs (générés par les étapes précédentes)
PREDICTIONS_CSV = OUTPUTS_DIR / "etape5" / "predictions_test.csv"
REGIMES_CSV = OUTPUTS_DIR / "etape4" / "regimes" / "masi_regimes_test.csv"
FEATURES_TRAIN = OUTPUTS_DIR / "etape3" / "features" / "masi_features_train.csv"
FEATURES_VAL = OUTPUTS_DIR / "etape3" / "features" / "masi_features_val.csv"
FEATURES_TEST = OUTPUTS_DIR / "etape3" / "features" / "masi_features_test.csv"
RISK_METRICS_CSV = OUTPUTS_DIR / "etape7" / "risk_metrics_test.csv"
EQUITY_CURVES_CSV = OUTPUTS_DIR / "etape6" / "equity_curves.csv"

# Outputs canoniques
CANONICAL_CSV = OUTPUTS_DIR / "etape6" / "etape6_final_predictions.csv"

# Scripts (pour le wrapper train)
ETAPE5_SCRIPT = SCRIPTS_DIR / "05_cnn_lstm.py"
ETAPE7_SCRIPT = SCRIPTS_DIR / "07_risk_layer.py"

# ============================================================================
# CONSTANTES DE TRADING (figées par prompt.md & étapes 0-9)
# ============================================================================
COST_BPS = 5.0
COST_DEC = COST_BPS / 10_000
PPY = 252                      # jours de trading par an

# Risque
ALPHA_VAR = 0.05               # niveau VaR/ES 5%
ROLL_VAR = 252                 # fenêtre rolling VaR/ES historique
ROLL_BUDGET = 60               # fenêtre rolling budget (stratégie 7)
B_STD = 0.01                   # budget standard 1% (stratégies 6-7)
Q_STRESS = 0.30                # quantile bas en stress (stratégie 7)
Q_NORMAL = 0.70                # quantile haut en normal (stratégie 7)
EPS_VAR = 1e-6

# ============================================================================
# MODÈLE & STRATÉGIE PRODUCTION (étapes 5-9)
# ============================================================================
PRODUCTION_MODEL = "cnn_lstm_base12"
PRODUCTION_STRATEGY = "hmm_gate"   # confirmé étape 8 (DSR 0.997) + étape 9 (robustesse)

# 7 stratégies disponibles
AVAILABLE_STRATEGIES = [
    "buy_hold",            # 1. réf. passive
    "cnn_lstm_nu",         # 2. réf. étape 6
    "hmm_gate",            # 3. PRODUCTION — étape 8/9
    "risk_gate",           # 4. défense étape 7
    "hmm_risk_gate",       # 5. combinaison (dégradée)
    "var_budget",          # 6. risk-budgeting continu
    "hmm_cond_budget",     # 7. risk-budgeting régime-conditionnel
]
