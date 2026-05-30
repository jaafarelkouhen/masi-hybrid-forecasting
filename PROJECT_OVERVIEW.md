# MASI Hybrid Forecasting — System Overview

This is the master guide to the whole system. It explains **the approach**, the
**two repositories** that make it up, and **how everything fits together**. Read
this first; the per-folder `README.md` files go one level deeper.

---

## 1. What this project does

It forecasts the **next-day log-return of the MASI index** (Casablanca Stock
Exchange) and turns that forecast into a **risk-aware trading signal**, then
exposes the results through a **web dashboard with an AI assistant**.

The core idea is a **hybrid model**:

- a **Hidden Markov Model (HMM)** detects the current *market regime*
  (bull / neutral / bear-like states),
- a compact **CNN-LSTM** predicts the next-day return,
- a **strategy layer** combines the two: the CNN-LSTM gives the directional
  signal, and the HMM acts as a *gate* that cuts exposure in the Neutral regime.

Everything is built under a **strict walk-forward, anti-leakage protocol** — the
single most important methodological commitment of the project.

---

## 2. The two repositories

| Repo | Role | Stack |
|---|---|---|
| **`masi-hybrid-forecasting`** | The research engine: data, models, backtests, the reproducible pipeline, and the canonical output files. | Python (numpy, pandas, scikit-learn, hmmlearn, PyTorch) |
| **`dashbord-masi-hybrid-forecasting-01`** | The product layer: a FastAPI backend + Next.js frontend that *reads* the engine's outputs, plus a RAG assistant that answers questions about the project. | FastAPI, Next.js/React/Tailwind, ChromaDB, sentence-transformers, optional Ollama LLM |

**Key relationship:** the dashboard does **not** re-implement any modelling. It
reads the canonical CSV/JSON artifacts produced by the engine's CLI. By default
it looks for the engine at `../masi-hybrid-forecasting` (next to the dashboard
folder); override with the `MASI_PROJECT_ROOT` environment variable.

```
2-TimeSeriesProject/
├── masi-hybrid-forecasting/              ← engine (this repo)
└── dashbord-masi-hybrid-forecasting-01/  ← dashboard, reads ../masi-hybrid-forecasting
```

---

## 3. End-to-end data flow

```text
[ raw MASI data ]
      │
      ▼   (engine: scripts 00→09 / pipeline CLI)
preprocessing & temporal splits
      │
      ▼
leakage-free feature engineering
      │
      ▼
HMM regime detection ───────┐
      │                     │
      ▼                     │
CNN-LSTM next-day return    │
      │                     │
      ▼                     ▼
strategy layer (HMM-gate, risk-gate, VaR budget)
      │
      ▼
risk layer (VaR / ES / GARCH vol / risk regime)
      │
      ▼
canonical export  →  outputs/etape*/  +  reports/
      │
      ▼   (dashboard backend reads these files)
FastAPI services  →  REST API  →  Next.js dashboard (Forecast / Risk / Backtest)
      │
      ▼
RAG assistant indexes docs + reports + notebooks → answers questions with sources
```

---

## 4. Methodology in one paragraph

Data is split chronologically — **TRAIN 2007–2020 / VAL 2020–2022 / TEST
2022–2026** (70/10/20) with **8-day gaps** between splits. Every transform
(scalers, HMM fit, feature statistics, GARCH) is **fit on the training window
only and applied causally**, so no future information leaks into the past. Deep
learning is only evaluated *after* simple baselines (naive, mean, ARIMA, random
forest). Performance is reported with realistic transaction costs and a full
metric set (directional accuracy, Sharpe, Sortino, max drawdown, Calmar,
Deflated Sharpe Ratio, regime-conditional metrics). The headline result —
**CNN-LSTM `base12` + HMM-gate**, DSR ≈ 0.997, robust to 5/10/20 bps costs — is
treated as **defensible under the study protocol, not proof of live alpha**. A
later holdout or paper-trading period is still required. Full rules:
[`docs/methodology.md`](docs/methodology.md) and
[`docs/anti_leakage.md`](docs/anti_leakage.md).

---

## 5. How to run the whole thing

**Step 1 — the engine** (produces the artifacts the dashboard needs):

```bash
cd masi-hybrid-forecasting
pip install -e ".[notebooks,dev]"
python -m masi_hybrid_forecasting.pipeline predict
python -m masi_hybrid_forecasting.pipeline risk
python -m masi_hybrid_forecasting.pipeline backtest --strategy hmm_gate
python -m masi_hybrid_forecasting.pipeline export  --strategy hmm_gate
```

**Step 2 — the dashboard**:

```bash
cd ../dashbord-masi-hybrid-forecasting-01
pip install -r requirements.txt
cp .env.example .env          # MASI_PROJECT_ROOT defaults to ../masi-hybrid-forecasting
python -m rag_project.scripts.build_index   # build the assistant's index
python -m uvicorn app.main:app --port 8000  # backend
# in another terminal:
cd frontend && npm install && npm run dev    # frontend at http://localhost:3000
```

The RAG assistant works **with no LLM and no network** by default
(`LLM_BACKEND=fallback`). Set `LLM_BACKEND=ollama` (+ `ollama pull qwen2.5:3b`)
for generative answers.

---

## 6. Where to read next

- Engine internals → [`README.md`](README.md), [`docs/INDEX.md`](docs/INDEX.md)
- Per-step code → [`scripts/README.md`](scripts/README.md), [`notebooks/README.md`](notebooks/README.md)
- The package & CLI → [`src/README.md`](src/README.md)
- Dashboard backend → `../dashbord-masi-hybrid-forecasting-01/app/README.md`
- RAG assistant → `../dashbord-masi-hybrid-forecasting-01/rag_project/README.md`

> ⚠️ This is a research project on a frontier market. Nothing here is investment
> advice. The assistant is explicitly built to **refuse buy/sell recommendations**.
