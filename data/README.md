# `data/`

Data lake for the engine, following the common raw → interim → processed
convention. **Heavy raw files are gitignored**; regenerate them with the pipeline
or supply them separately for a clean clone.

```
data/
├── raw/         # source files (masi_raw.csv, master_dataset.csv/.xlsx) — gitignored
├── external/    # third-party inputs
├── interim/     # intermediate merges (masi_merged.csv)
└── processed/   # model-ready data
    ├── masi_processed.csv
    ├── features/          # engineered features
    └── regimes/           # masi_with_regimes.csv (HMM output)
```

## Flow

`raw/` → audited & merged → `interim/` → cleaned, split, feature-engineered →
`processed/`. The HMM step adds the regime labels in `processed/regimes/`.

> ⚠️ Anti-leakage: all statistics used to build `processed/` (scalers, feature
> stats, HMM/GARCH fits) are computed on the **training window only** and applied
> causally. See [`../docs/anti_leakage.md`](../docs/anti_leakage.md).
