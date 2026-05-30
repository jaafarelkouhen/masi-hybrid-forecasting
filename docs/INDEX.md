# INDEX — MASI Hybrid Forecasting System

Index courant du projet après migration vers `masi-hybrid-forecasting/`.
Le code reproductible vit dans `scripts/`, les artefacts locaux dans `outputs/`,
les figures dans `reports/figures/`, et la CLI propre dans
`src/masi_hybrid_forecasting/pipeline/`.

## Pipeline — Étapes 0 à 10

| Étape | Statut | Script | Artefacts principaux | Rapport |
|---|---|---|---|---|
| 0 — Data audit | OK | `scripts/00_data_audit.py` | `data/interim/masi_merged.csv`, `outputs/etape0/` | `outputs/etape0/report.md` |
| 1 — Preprocessing | OK | `scripts/01_preprocessing.py` | `outputs/etape1/splits/` | `outputs/etape1/report.md` |
| 2 — Baselines | OK | `scripts/02_baselines.py` | `outputs/etape2/metrics.json`, predictions | `outputs/etape2/report.md` |
| 3 — Feature engineering | OK | `scripts/03_feature_engineering.py` | `outputs/etape3/features/` | `docs/anti_leakage.md` |
| 4 — HMM regimes | OK | `scripts/04_hmm_regimes.py` | `outputs/etape4/regimes/` | `outputs/etape4/report.md` |
| 5 — CNN-LSTM | OK | `scripts/05_cnn_lstm.py` | `outputs/etape5/predictions_test.csv` | `outputs/etape5/report.md` |
| 6 — Model backtest | OK | `scripts/06_backtesting.py` | `outputs/etape6/backtest_metrics.json` | `outputs/etape6/report.md` |
| 7 — Risk layer | OK | `scripts/07_risk_layer.py` | `outputs/etape7/risk_metrics_test.csv` | `outputs/etape7/report.md` |
| 8 — Strategy layer | OK | `scripts/08_strategies.py` | `outputs/etape8/strategies_metrics.json` | `outputs/etape8/report.md` |
| 9 — Robustness | OK | `scripts/09_robustness.py` | `outputs/etape9/robustness_metrics.json` | `outputs/etape9/report.md` |
| 10 — CLI package | OK | `src/masi_hybrid_forecasting/pipeline/` | `masi-pipeline` / `python -m masi_hybrid_forecasting.pipeline` | `reports/final_results.md` |

## Commandes Rapides

```powershell
pip install -e ".[notebooks,dev]"

python -m masi_hybrid_forecasting.pipeline predict
python -m masi_hybrid_forecasting.pipeline risk
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate --cost-bps 5
python -m masi_hybrid_forecasting.pipeline export --strategy hmm_gate

python -m pytest
```

Après installation editable, la CLI console est aussi disponible si le dossier
`Scripts` de l'installation Python est dans le `PATH` :

```powershell
masi-pipeline backtest --strategy hmm_gate --cost-bps 5
```

## Résultat Final À Retenir

La recommandation historique du projet est **CNN-LSTM `base12` + HMM-gate** :
le CNN-LSTM fournit le signal directionnel, et le HMM-gate coupe l'exposition
dans le régime Neutral. Le résultat est économiquement intéressant et robuste
dans le protocole du mémoire, mais il reste à confirmer sur un holdout ultérieur
ou en paper-trading hors période 2022-2026.

## Notes

- Les chemins de l'ancien lab sont conservés seulement dans `docs/migration_plan.md`,
  `docs/README_legacy.md` et `docs/project_spec.md` comme trace historique.
- Les artefacts `outputs/`, `data/raw/` et `reports/figures/` sont ignorés par
  git; il faut les régénérer ou les fournir séparément pour un clone propre.
- La règle L8 est partielle : le GARCH et le HMM sont fit sur TRAIN et appliqués
  causalement. Un refit complet par fold reste une extension future.
