# masi_hybrid_forecasting.pipeline — CLI propre (étape 10)

Package Python du **MASI Hybrid Forecasting System** (mémoire 2026). Fournit
une CLI à 5 sous-commandes (`train`, `predict`, `backtest`, `risk`, `export`)
qui orchestrent les étapes 0-9 du projet.

## Production stack (confirmée étape 9)

- **Modèle** : CNN-LSTM `base12` (étape 5, walk-forward 5 folds, ~5000 paramètres)
- **Stratégie** : HMM-gate (étape 8) — trade seulement si HMM régime ∈ {Bear, Bull}
- **Risque** : couche optionnelle VaR/ES/régime (étape 7) — défense alternative
- **Validation** : DSR = 0.997 sous le protocole historique · robuste à 5/10/20 bps · stable P1 ≈ P2
- **Coûts courants** : coût one-way proportionnel au turnover `|Δposition|`
  (un flip `-1 → +1` coûte deux unités).

## Installation

Le package s'installe via pip depuis la racine `masi-hybrid-forecasting/` :

```powershell
cd masi-hybrid-forecasting
pip install -e ".[notebooks,dev]"
```

(deps core + jupyter + pytest/ruff). Voir [`../../../pyproject.toml`](../../../pyproject.toml).

## Usage

```powershell
$env:PYTHONIOENCODING = "utf-8"     # une fois par session PowerShell (Windows)
python -m masi_hybrid_forecasting.pipeline <command> [options]
masi-pipeline <command> [options]  # après pip install -e ., si Scripts est dans PATH
```

Si tu n'as pas installé le package (`pip install -e .`), utiliser à la place :

```powershell
$env:PYTHONPATH = "src"
python -m masi_hybrid_forecasting.pipeline <command> [options]
```

### `predict` — charge / affiche les prédictions TEST

Charge `outputs/etape5/predictions_test.csv` + dates étape 4, affiche stats.

```powershell
python -m masi_hybrid_forecasting.pipeline predict
python -m masi_hybrid_forecasting.pipeline predict --output mes_predictions.csv
```

### `risk` — génère la couche risque

Recalcule VaR (hist + paramétrique GARCH), ES, régime de risque (seuils
TRAIN+VAL figés) et écrit `outputs/etape7/risk_metrics_test.csv`. Inclut tests
Kupiec POF + Christoffersen indépendance.

```powershell
python -m masi_hybrid_forecasting.pipeline risk
python -m masi_hybrid_forecasting.pipeline risk --output mon_risque.csv
```

### `backtest` — applique stratégie + métriques

Applique une des 7 stratégies aux prédictions, calcule Sharpe, Sortino, MDD,
Calmar, DSR, regime-conditionnel.

```powershell
# Stratégie production par défaut
python -m masi_hybrid_forecasting.pipeline backtest

# Comparer plusieurs stratégies à différents coûts
python -m masi_hybrid_forecasting.pipeline backtest --strategy cnn_lstm_nu --cost-bps 5
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate --cost-bps 5
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate --cost-bps 20
python -m masi_hybrid_forecasting.pipeline backtest --strategy buy_hold

# Export métriques en JSON
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate --output bt_hmm.json
```

Stratégies disponibles :

| Nom CLI | Description | Mode |
|---|---|---|
| `buy_hold` | Buy & Hold (réf. passive) | binary |
| `cnn_lstm_nu` | sign(ŷ) sans filtre (réf. étape 6) | binary |
| `hmm_gate` | **PRODUCTION** : trade si HMM ∈ {Bear, Bull} | binary |
| `risk_gate` | trade si risk_regime ≠ high (défense étape 7) | binary |
| `hmm_risk_gate` | intersection HMM + risk (dégradée — étape 9 p=0.043) | binary |
| `var_budget` | sizing continu w = min(1, B/\|VaR\|) (inspiré repo source) | continuous |
| `hmm_cond_budget` | budget régime-conditionnel | continuous |

### `export` — fichier canonique pour API / dashboard

Construit `etape6_final_predictions.csv` (ou path custom) avec toutes les
colonnes nécessaires pour brancher un frontend (date, actual, predicted, signal,
regime, position, strategy_return, equity, + risque si dispo).

```powershell
python -m masi_hybrid_forecasting.pipeline export
python -m masi_hybrid_forecasting.pipeline export --strategy hmm_gate --output canonical.csv
```

### `train` — re-entraîne CNN-LSTM walk-forward (⚠ 10-15 min)

Relance `scripts/05_cnn_lstm.py` (5 folds, ~5000 époques total). À ne lancer que
si tu modifies les features (étape 3) ou veux changer l'architecture.

```powershell
python -m masi_hybrid_forecasting.pipeline train
```

## Pipeline end-to-end (de zéro à dashboard)

```powershell
$env:PYTHONIOENCODING = "utf-8"

# 1. (Optionnel) Re-entraîner si features changées
python -m masi_hybrid_forecasting.pipeline train

# 2. Vérifier prédictions
python -m masi_hybrid_forecasting.pipeline predict

# 3. Calculer la couche risque
python -m masi_hybrid_forecasting.pipeline risk

# 4. Backtester la stratégie production
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate --output bt.json

# 5. Exporter le fichier canonique pour API/dashboard
python -m masi_hybrid_forecasting.pipeline export --strategy hmm_gate
```

Sortie finale : `outputs/etape6/etape6_final_predictions.csv` — single source
of truth pour brancher un dashboard (Streamlit, FastAPI, etc.).

## Architecture du package

```
masi_hybrid_forecasting/pipeline/
├── __init__.py          version + métadonnées
├── __main__.py          entry point python -m
├── cli.py               argparse 5 sous-commandes
├── config.py            chemins + constantes
├── strategies.py        les 7 stratégies (étape 8)
├── metrics.py           Sharpe, Sortino, MDD, DSR, JKM
├── risk.py              VaR/ES/régime + tests Kupiec/Christoffersen
├── train.py             wrapper subprocess vers scripts/05_cnn_lstm.py
├── predict.py           charge prédictions + dates
├── backtest.py          applique stratégie + métriques + JSON
├── export.py            fichier canonique pour API/dashboard
└── README.md            ce fichier
```

## Anti-fuite (L1–L8)

Le pipeline préserve toutes les garanties anti-fuite des étapes 0-9 :

- L1 — scaler/quantiles fit TRAIN(+VAL) uniquement, jamais TEST
- L2 — HMM régimes causaux (étape 4 v2)
- L3 — rolling features `.shift(1).rolling(...)` strictement causal
- L4 — `y_true` jamais utilisé pour décider signal
- L5 — signal `t` exécuté contre `y_true_t = ln(P_{t+1}/P_t)`
- L6 — gap walk-forward inhérent étape 1
- L7 — jours zero-volume retirés étape 1
- L8 — GARCH fit TRAIN-only (étape 3) ; full per-fold refit = future work

Toute commande qui violerait ces règles lèverait une `AssertionError`.

## Limitations connues

1. **`train` relance le walk-forward complet** — pas de fit incrémental.
2. **`predict` consomme les prédictions disque** — pas de scoring temps réel sur
   nouvelles données (limite d'un walk-forward académique).
3. **Stratégies `var_budget` fragiles aux coûts** : Sharpe négatif à 20 bps
   (étape 9). Réservées au cas où le coût réel est < 10 bps.
4. **Échelle données** : conçu pour ~948 jours TEST canoniques étape 1.
   D'autres splits demanderaient régénération étape 3/4/5.

## Crédits & attributions

- Stratégies `var_budget` et `hmm_cond_budget` inspirées de
  `masi-risk-research-notebooks/src/analysis/economic_evaluation.py`
  (fonctions `compute_weights_from_budget_and_var`, `weights_hmm_lstm_quantile_budget`).
- DSR : Bailey & López de Prado (2014).
- Kupiec POF : Kupiec (1995). Christoffersen : Christoffersen (1998).
- Jobson-Korkie : Jobson & Korkie (1981). Memmel : Memmel (2003).
- Risk-budgeting : Bertrand & Prigent (2003), Roncalli (2014).

## Voir aussi

- `prompt.md` (racine projet) — gouvernance étapes 0-6
- `outputs/etape0/report.md` à `outputs/etape10/report.md` — rapports par étape
- `outputs/etape6/etape6_final_predictions.csv` — fichier canonique produit par
  `export`
