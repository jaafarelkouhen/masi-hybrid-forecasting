# MASI Hybrid Forecasting — Documentation

Hybrid forecasting system for the **MASI** (Casablanca Stock Exchange) index :
**HMM** for regime detection + **CNN-LSTM** for next-day log-return prediction,
under a strict walk-forward anti-leakage methodology.

```{image} ../reports/figures/etape6/etape6_equity_curves.png
:alt: Equity curves — production strategy vs baselines
:align: center
```

> **Headline result** — CNN-LSTM `base12` + HMM-gate, TEST 2022-06-28 → 2026-04-17
> (948 days) : Sharpe ≈ **+1.71** (historical convention) / **+1.55**
> (turnover-aware), Max Drawdown ≈ **−6 %**, DSR ≈ **0.997**, robust to
> 5/10/20 bps costs.

---

## Quick links

- [Pipeline index (étapes 0 → 10)](pipeline_index.md)
- [Methodology](methodology.md)
- [Anti-leakage rules (L1–L8)](anti_leakage.md)
- [Data pipeline](data_pipeline.md)
- [Visual gallery — all 44 figures](gallery.md)
- [Literature review](literature_review.md)

---

## Site navigation

```{toctree}
:maxdepth: 2
:caption: Overview

pipeline_index
methodology
data_pipeline
gallery
```

```{toctree}
:maxdepth: 2
:caption: Methodology & rigor

anti_leakage
literature_review
```

---

## Quick start

```bash
pip install -e ".[notebooks,dev]"

python -m masi_hybrid_forecasting.pipeline predict
python -m masi_hybrid_forecasting.pipeline risk
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate
python -m masi_hybrid_forecasting.pipeline export  --strategy hmm_gate
```

Full CLI reference : see the project [README](https://github.com/jaafarelkouhen/masi-hybrid-forecasting/blob/main/README.md)
on GitHub or `src/masi_hybrid_forecasting/pipeline/README.md` in the repo.

---

## Build this site locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

The hosted build on Read the Docs is driven by `.readthedocs.yaml` at the repo
root and rebuilds automatically on every push to `main`.

---

## Disclaimer

This is a research project on a frontier market. Nothing here is investment
advice. The headline numbers are defensible **under the study protocol**, not
proof of live alpha — a later holdout or paper-trading period is still
required.
