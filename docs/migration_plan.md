# Plan de migration — ancien projet → `masi-hybrid-forecasting/`

**Principe** : on COPIE d'abord (`Copy-Item -Recurse`), on vérifie, on supprime les
originaux à la fin. Toutes les commandes sont à lancer depuis la racine
`C:\Users\jelko\OneDrive\Desktop\2-TimeSeriesProject\`.

**Notation** : `OLD\` = racine actuelle, `NEW\` = `masi-hybrid-forecasting\`.

---

## Phase 1 — Données (risque : nul)

| Source | Destination |
|---|---|
| `OLD\Data\masi_raw.csv` | `NEW\data\raw\masi_raw.csv` |
| `OLD\Data\master_dataset.csv` | `NEW\data\raw\master_dataset.csv` |
| `OLD\Data\master_dataset.xlsx` | `NEW\data\raw\master_dataset.xlsx` |
| `OLD\Data\masi_merged.csv` | `NEW\data\interim\masi_merged.csv` |
| `OLD\Data\masi_processed.csv` | `NEW\data\processed\masi_processed.csv` |
| `OLD\Data\masi_with_regimes.csv` | `NEW\data\processed\regimes\masi_with_regimes.csv` |

```powershell
Copy-Item "Data\masi_raw.csv"             "masi-hybrid-forecasting\data\raw\"
Copy-Item "Data\master_dataset.csv"       "masi-hybrid-forecasting\data\raw\"
Copy-Item "Data\master_dataset.xlsx"      "masi-hybrid-forecasting\data\raw\"
Copy-Item "Data\masi_merged.csv"          "masi-hybrid-forecasting\data\interim\"
Copy-Item "Data\masi_processed.csv"       "masi-hybrid-forecasting\data\processed\"
Copy-Item "Data\masi_with_regimes.csv"    "masi-hybrid-forecasting\data\processed\regimes\"
```

Les splits TRAIN/VAL/TEST sont dans `Output_Labs\etape1_splits\` — traités phase 6.

---

## Phase 2 — Bibliographie (risque : nul, juste des PDFs)

```powershell
Copy-Item "Anti-Leakage & Validation"           "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "Architecture HMM + Deep Learning"    "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "Deep Learning Portfolio & Général"   "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "Marché Marocain (MASI) — Articles directs" "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "Marchés Émergents & Frontière"      "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "Proxy Volatilité sans VIX"          "masi-hybrid-forecasting\docs\references\" -Recurse
```

Les deux dossiers d'analyse de repos similaires :

```powershell
Copy-Item "_analysis_research_notebooks"  "masi-hybrid-forecasting\docs\references\" -Recurse
Copy-Item "_analysis_similar_dashboard"   "masi-hybrid-forecasting\docs\references\" -Recurse
```

---

## Phase 3 — Documentation racine

| Source | Destination | Note |
|---|---|---|
| `OLD\prompt.md` | `NEW\docs\project_spec.md` | **REMPLACE** le placeholder (c'est LA spec) |
| `OLD\INDEX.md` | `NEW\docs\INDEX.md` | Index global du projet |
| `OLD\README.md` | `NEW\docs\README_legacy.md` | À fusionner manuellement avec le `NEW\README.md` minimal |

```powershell
Copy-Item "prompt.md"   "masi-hybrid-forecasting\docs\project_spec.md" -Force
Copy-Item "INDEX.md"    "masi-hybrid-forecasting\docs\INDEX.md"
Copy-Item "README.md"   "masi-hybrid-forecasting\docs\README_legacy.md"
```

`Output_Labs\etape0_literature_synthesis.md` complète `docs\literature_review.md` :

```powershell
Copy-Item "Output_Labs\etape0_literature_synthesis.md" "masi-hybrid-forecasting\docs\literature_review.md" -Force
```

`Output_Labs\etape3_leakage_audit.md` alimente `docs\anti_leakage.md` :

```powershell
Copy-Item "Output_Labs\etape3_leakage_audit.md" "masi-hybrid-forecasting\docs\anti_leakage.md" -Force
```

---

## Phase 4 — Notebooks (renommer `etapeN_*` → `0N_*`)

| Source | Destination |
|---|---|
| `Output_Labs\etape1_preprocessing.ipynb` | `NEW\notebooks\01_preprocessing.ipynb` |
| `Output_Labs\etape2_baselines.ipynb` | `NEW\notebooks\02_baselines.ipynb` |
| `Output_Labs\etape3_features.ipynb` | `NEW\notebooks\03_feature_engineering.ipynb` |
| `Output_Labs\etape4_hmm.ipynb` | `NEW\notebooks\04_hmm_regimes.ipynb` |
| `Output_Labs\etape5_cnn_lstm.ipynb` | `NEW\notebooks\05_cnn_lstm.ipynb` |
| `Output_Labs\etape6_backtest.ipynb` | `NEW\notebooks\06_backtesting.ipynb` |
| `Output_Labs\etape7_risk.ipynb` | `NEW\notebooks\07_risk_layer.ipynb` (extension) |
| `Output_Labs\etape8_strategies.ipynb` | `NEW\notebooks\08_strategies.ipynb` (extension) |
| `Output_Labs\etape9_robustness.ipynb` | `NEW\notebooks\09_robustness.ipynb` (extension) |

> Pas d'étape 0 en notebook (étape 0 = audit `.py` only). Le slot `00_data_audit.ipynb`
> proposé reste vide ou tu peux y dériver un notebook à partir de `etape0_audit.py`.

```powershell
Copy-Item "Output_Labs\etape1_preprocessing.ipynb" "masi-hybrid-forecasting\notebooks\01_preprocessing.ipynb"
Copy-Item "Output_Labs\etape2_baselines.ipynb"     "masi-hybrid-forecasting\notebooks\02_baselines.ipynb"
Copy-Item "Output_Labs\etape3_features.ipynb"      "masi-hybrid-forecasting\notebooks\03_feature_engineering.ipynb"
Copy-Item "Output_Labs\etape4_hmm.ipynb"           "masi-hybrid-forecasting\notebooks\04_hmm_regimes.ipynb"
Copy-Item "Output_Labs\etape5_cnn_lstm.ipynb"      "masi-hybrid-forecasting\notebooks\05_cnn_lstm.ipynb"
Copy-Item "Output_Labs\etape6_backtest.ipynb"      "masi-hybrid-forecasting\notebooks\06_backtesting.ipynb"
Copy-Item "Output_Labs\etape7_risk.ipynb"          "masi-hybrid-forecasting\notebooks\07_risk_layer.ipynb"
Copy-Item "Output_Labs\etape8_strategies.ipynb"    "masi-hybrid-forecasting\notebooks\08_strategies.ipynb"
Copy-Item "Output_Labs\etape9_robustness.ipynb"    "masi-hybrid-forecasting\notebooks\09_robustness.ipynb"
```

**⚠️ Attention** : les notebooks contiennent des chemins relatifs vers `Data\` et
`Output_Labs\etapeN_*\`. Ils NE TOURNERONT PAS depuis `NEW\notebooks\` tant
que tu n'as pas adapté les chemins. Voir Phase 10 (vérification).

---

## Phase 5 — Scripts `.py` étape par étape

Les `.py` sont des entry-points (pas du code librairie réutilisable), donc
`scripts/` est le bon endroit.

| Source | Destination |
|---|---|
| `Output_Labs\etape0_audit.py` | `NEW\scripts\00_data_audit.py` |
| `Output_Labs\etape1_preprocessing.py` | `NEW\scripts\01_preprocessing.py` |
| `Output_Labs\etape2_baselines.py` | `NEW\scripts\02_baselines.py` |
| `Output_Labs\etape3_features.py` | `NEW\scripts\03_feature_engineering.py` |
| `Output_Labs\etape4_hmm.py` | `NEW\scripts\04_hmm_regimes.py` |
| `Output_Labs\etape5_cnn_lstm.py` | `NEW\scripts\05_cnn_lstm.py` |
| `Output_Labs\etape6_backtest.py` | `NEW\scripts\06_backtesting.py` |
| `Output_Labs\etape7_risk.py` | `NEW\scripts\07_risk_layer.py` |
| `Output_Labs\etape8_strategies.py` | `NEW\scripts\08_strategies.py` |
| `Output_Labs\etape9_robustness.py` | `NEW\scripts\09_robustness.py` |

```powershell
Copy-Item "Output_Labs\etape0_audit.py"          "masi-hybrid-forecasting\scripts\00_data_audit.py"
Copy-Item "Output_Labs\etape1_preprocessing.py"  "masi-hybrid-forecasting\scripts\01_preprocessing.py"
Copy-Item "Output_Labs\etape2_baselines.py"      "masi-hybrid-forecasting\scripts\02_baselines.py"
Copy-Item "Output_Labs\etape3_features.py"       "masi-hybrid-forecasting\scripts\03_feature_engineering.py"
Copy-Item "Output_Labs\etape4_hmm.py"            "masi-hybrid-forecasting\scripts\04_hmm_regimes.py"
Copy-Item "Output_Labs\etape5_cnn_lstm.py"       "masi-hybrid-forecasting\scripts\05_cnn_lstm.py"
Copy-Item "Output_Labs\etape6_backtest.py"       "masi-hybrid-forecasting\scripts\06_backtesting.py"
Copy-Item "Output_Labs\etape7_risk.py"           "masi-hybrid-forecasting\scripts\07_risk_layer.py"
Copy-Item "Output_Labs\etape8_strategies.py"     "masi-hybrid-forecasting\scripts\08_strategies.py"
Copy-Item "Output_Labs\etape9_robustness.py"     "masi-hybrid-forecasting\scripts\09_robustness.py"
```

**⚠️ Chemins en dur** : ces scripts lisent depuis `Data\` et écrivent dans
`Output_Labs\etapeN_*\`. Tu DEVRAS les adapter pour pointer vers `..\data\` et
`..\outputs\etapeN\`.

---

## Phase 6 — Artefacts (resultats / splits / metadata)

| Source | Destination |
|---|---|
| `Output_Labs\etape1_splits\` | `NEW\outputs\etape1\splits\` |
| `Output_Labs\etape2_results\` | `NEW\outputs\etape2\` |
| `Output_Labs\etape3_features\` | `NEW\outputs\etape3\features\` |
| `Output_Labs\etape4_regimes\` | `NEW\outputs\etape4\regimes\` |
| `Output_Labs\etape5_results\` | `NEW\outputs\etape5\` |
| `Output_Labs\etape6_results\` | `NEW\outputs\etape6\` |
| `Output_Labs\etape6_final_predictions.csv` | `NEW\outputs\etape6\etape6_final_predictions.csv` (fichier canonique) |
| `Output_Labs\etape7_risk\` | `NEW\outputs\etape7\` |
| `Output_Labs\etape8_results\` | `NEW\outputs\etape8\` |
| `Output_Labs\etape9_results\` | `NEW\outputs\etape9\` |

```powershell
Copy-Item "Output_Labs\etape1_splits"               "masi-hybrid-forecasting\outputs\etape1\splits"   -Recurse
Copy-Item "Output_Labs\etape2_results"              "masi-hybrid-forecasting\outputs\etape2"          -Recurse
Copy-Item "Output_Labs\etape3_features"             "masi-hybrid-forecasting\outputs\etape3\features" -Recurse
Copy-Item "Output_Labs\etape4_regimes"              "masi-hybrid-forecasting\outputs\etape4\regimes"  -Recurse
Copy-Item "Output_Labs\etape5_results"              "masi-hybrid-forecasting\outputs\etape5"          -Recurse
Copy-Item "Output_Labs\etape6_results"              "masi-hybrid-forecasting\outputs\etape6"          -Recurse
Copy-Item "Output_Labs\etape6_final_predictions.csv" "masi-hybrid-forecasting\outputs\etape6\"
Copy-Item "Output_Labs\etape7_risk"                 "masi-hybrid-forecasting\outputs\etape7"          -Recurse
Copy-Item "Output_Labs\etape8_results"              "masi-hybrid-forecasting\outputs\etape8"          -Recurse
Copy-Item "Output_Labs\etape9_results"              "masi-hybrid-forecasting\outputs\etape9"          -Recurse
```

---

## Phase 7 — Reports markdown et READMEs étapes

Chaque étape a un `etapeN_README.md` (descriptif court) + un `etapeN_*_report.md`
(résultats détaillés). Je propose : tout dans `outputs\etapeN\` à côté des artefacts.

```powershell
# READMEs courts
Copy-Item "Output_Labs\etape0-README.md"          "masi-hybrid-forecasting\outputs\etape0\README.md"
Copy-Item "Output_Labs\etape1_README.md"          "masi-hybrid-forecasting\outputs\etape1\README.md"
Copy-Item "Output_Labs\etape2_README.md"          "masi-hybrid-forecasting\outputs\etape2\README.md"
Copy-Item "Output_Labs\etape3_README.md"          "masi-hybrid-forecasting\outputs\etape3\README.md"
Copy-Item "Output_Labs\etape4_README.md"          "masi-hybrid-forecasting\outputs\etape4\README.md"
Copy-Item "Output_Labs\etape5_README.md"          "masi-hybrid-forecasting\outputs\etape5\README.md"
Copy-Item "Output_Labs\etape6_README.md"          "masi-hybrid-forecasting\outputs\etape6\README.md"

# Reports détaillés
Copy-Item "Output_Labs\etape0_audit_report.md"     "masi-hybrid-forecasting\outputs\etape0\report.md"
Copy-Item "Output_Labs\etape1_report.md"           "masi-hybrid-forecasting\outputs\etape1\report.md"
Copy-Item "Output_Labs\etape2_report.md"           "masi-hybrid-forecasting\outputs\etape2\report.md"
Copy-Item "Output_Labs\etape4_regime_report.md"    "masi-hybrid-forecasting\outputs\etape4\report.md"
Copy-Item "Output_Labs\etape5_walkforward_report.md" "masi-hybrid-forecasting\outputs\etape5\report.md"
Copy-Item "Output_Labs\etape6_final_report.md"     "masi-hybrid-forecasting\outputs\etape6\report.md"
Copy-Item "Output_Labs\etape7_risk_report.md"      "masi-hybrid-forecasting\outputs\etape7\report.md"
Copy-Item "Output_Labs\etape8_strategies_report.md" "masi-hybrid-forecasting\outputs\etape8\report.md"
Copy-Item "Output_Labs\etape9_robustness_report.md" "masi-hybrid-forecasting\outputs\etape9\report.md"
Copy-Item "Output_Labs\etape10_pipeline_report.md" "masi-hybrid-forecasting\outputs\etape10\report.md"
```

Les dossiers `outputs\etape0\`, `etape10\` n'existent pas encore — créer avant :

```powershell
New-Item -ItemType Directory -Force "masi-hybrid-forecasting\outputs\etape0", "masi-hybrid-forecasting\outputs\etape10" | Out-Null
```

Le rapport final consolidé alimente `reports\final_results.md` :

```powershell
Copy-Item "Output_Labs\etape10_pipeline_report.md" "masi-hybrid-forecasting\reports\final_results.md" -Force
```

---

## Phase 8 — Figures (plots PNG)

Chaque `etapeN_plots\` → `reports\figures\etapeN\`.

```powershell
Copy-Item "Output_Labs\etape0-udit_plots" "masi-hybrid-forecasting\reports\figures\etape0" -Recurse
Copy-Item "Output_Labs\etape1_plots"      "masi-hybrid-forecasting\reports\figures\etape1" -Recurse
Copy-Item "Output_Labs\etape2_plots"      "masi-hybrid-forecasting\reports\figures\etape2" -Recurse
Copy-Item "Output_Labs\etape3_plots"      "masi-hybrid-forecasting\reports\figures\etape3" -Recurse
Copy-Item "Output_Labs\etape4_plots"      "masi-hybrid-forecasting\reports\figures\etape4" -Recurse
Copy-Item "Output_Labs\etape5_plots"      "masi-hybrid-forecasting\reports\figures\etape5" -Recurse
Copy-Item "Output_Labs\etape6_plots"      "masi-hybrid-forecasting\reports\figures\etape6" -Recurse
Copy-Item "Output_Labs\etape7_plots"      "masi-hybrid-forecasting\reports\figures\etape7" -Recurse
Copy-Item "Output_Labs\etape8_plots"      "masi-hybrid-forecasting\reports\figures\etape8" -Recurse
Copy-Item "Output_Labs\etape9_plots"      "masi-hybrid-forecasting\reports\figures\etape9" -Recurse
```

L'analyse PDF du projet similaire :

```powershell
Copy-Item "Output_Labs\analyse_projet_similaire_masi_risk_dashboard.pdf" "masi-hybrid-forecasting\docs\references\"
```

---

## Phase 9 — Pipeline CLI `masi_pipeline` (⚠️ RISQUE ÉLEVÉ)

Le package `Output_Labs\masi_pipeline\` est un module Python avec imports
internes ET un `config.py` qui contient des chemins absolus vers `Output_Labs\`.
**Le déplacer cassera le pipeline tant que ces chemins ne seront pas adaptés.**

**Stratégie recommandée** : conserver la structure interne du package, juste
le déplacer en bloc dans `src\masi_hybrid_forecasting\pipeline\` :

```powershell
Copy-Item "Output_Labs\masi_pipeline" "masi-hybrid-forecasting\src\masi_hybrid_forecasting\pipeline" -Recurse -Exclude "__pycache__"
```

Puis :
1. **Éditer `src\masi_hybrid_forecasting\pipeline\config.py`** : remplacer les
   chemins `Output_Labs\etapeN_*` par `outputs\etapeN\` relatifs à la racine
   `NEW\`.
2. **Adapter le lancement** : `python -m masi_hybrid_forecasting.pipeline <command>`
   au lieu de `python -m Output_Labs.masi_pipeline <command>`.
3. **Vérifier les imports relatifs** dans `cli.py`, `__main__.py`, etc.

Le `__init__.py` du package racine `src\masi_hybrid_forecasting\__init__.py`
existe déjà — laisser tel quel, le sous-paquet `pipeline\` aura le sien.

**Alternative plus sûre** : ne PAS déplacer le pipeline maintenant. Garder
`Output_Labs\masi_pipeline\` en place jusqu'à ce que tu veuilles refactoriser.

---

## Phase 10 — Outils Node (générateur résumé exécutif)

Le générateur `.docx` est un outil de build, pas du code de pipeline.

```powershell
Copy-Item "generate_resume_executif.js"      "masi-hybrid-forecasting\tools\"
Copy-Item "package.json"                     "masi-hybrid-forecasting\tools\"
Copy-Item "package-lock.json"                "masi-hybrid-forecasting\tools\"
Copy-Item "MASI_Resume_Executif.docx"        "masi-hybrid-forecasting\reports\executive_summary\"
```

`node_modules\` : **ne pas copier**. Réinstaller via `cd masi-hybrid-forecasting\tools && npm install`.

---

## Phase 11 — Vérification avant suppression des originaux

Avant tout `Remove-Item`, vérifier :

1. **Comptage de fichiers** :
   ```powershell
   (Get-ChildItem "masi-hybrid-forecasting" -Recurse -File).Count
   ```
2. **Aucun chemin codé en dur n'est cassé** — relancer un script test :
   ```powershell
   cd masi-hybrid-forecasting
   python scripts\00_data_audit.py
   ```
3. **Notebooks ouvrables** : ouvrir `notebooks\01_preprocessing.ipynb` dans Jupyter
   et tenter le premier `pd.read_csv(...)`.

Une fois TOUT vérifié, supprimer les originaux :

```powershell
Remove-Item "Data"                 -Recurse -Force
Remove-Item "Output_Labs"          -Recurse -Force
Remove-Item "node_modules"         -Recurse -Force
Remove-Item "Anti-Leakage & Validation",
            "Architecture HMM + Deep Learning",
            "Deep Learning Portfolio & Général",
            "Marché Marocain (MASI) — Articles directs",
            "Marchés Émergents & Frontière",
            "Proxy Volatilité sans VIX",
            "_analysis_research_notebooks",
            "_analysis_similar_dashboard" -Recurse -Force
Remove-Item "INDEX.md", "MASI_Resume_Executif.docx", "README.md",
            "generate_resume_executif.js", "package.json", "package-lock.json",
            "prompt.md"
```

> **Ne lance ces suppressions QUE quand tu es 100% certain que la nouvelle structure
> tourne**. Les copies dans `NEW\` sont le seul backup.

---

## Récap — fichiers à NE PAS migrer

- `node_modules\` : régénéré par `npm install`
- `Output_Labs\__pycache__\` : régénéré par Python
- `Output_Labs\ETAPE0\` : dossier vide (étape 0 = audit `.py` only, pas de sous-dossier de résultat)
- Fichier `.claude\` : config locale de Claude Code, à laisser à la racine du workspace

---

## Ordre recommandé d'exécution

1. **Phase 1 → 8** : copies (toutes safe, aucun chemin cassé)
2. **Phase 9** : pipeline — copie + édition `config.py` + test `python -m masi_hybrid_forecasting.pipeline export`
3. **Phase 10** : Node tools
4. **Adapter les chemins** dans `scripts\*.py` et `notebooks\*.ipynb` (un par un)
5. **Phase 11** : vérifications puis nettoyage

Tu peux exécuter les phases 1, 2, 3, 7, 8 en parallèle ; les phases 4–6 dépendent
du fait que `outputs\etapeN\` existe (Phase 6 crée la structure).
