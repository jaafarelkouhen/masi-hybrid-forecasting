# MASI Hybrid Forecasting & Trading Research System
## Literature-Driven Strategy Synthesis (19 Papers, 6 Priority Folders)

**Project:** Hybrid HMM + CNN-LSTM forecasting for the Moroccan All Shares Index (MASI)
**Date:** 2026-05-19
**Author:** Quantitative Research Lab (project owner: J. Jelko)
**Status:** Étape 0 (v2) + Étape 1 (v2) complete on MERGED multi-factor dataset

---

## TABLE OF CONTENTS

0. [Data Strategy & Provenance (UPGRADE 2026-05-19)](#0-data-strategy--provenance)
1. [Executive Summary — The Recommended Strategy Stack](#1-executive-summary)
2. [Anti-Leakage Framework (P1)](#2-anti-leakage-framework-p1)
3. [MASI-Specific Findings (P2)](#3-masi-specific-findings-p2)
4. [HMM + Deep Learning Architecture (P3)](#4-hmm--deep-learning-architecture-p3)
5. [Frontier Market Constraints (P4)](#5-frontier-market-constraints-p4)
6. [Volatility Proxy Without VIX (P5)](#6-volatility-proxy-without-vix-p5)
7. [Deep Learning Portfolio Strategies (P6)](#7-deep-learning-portfolio-strategies-p6)
8. [Cross-Paper Consensus & Conflicts](#8-cross-paper-consensus--conflicts)
9. [Final Trading Strategy — Étape-by-Étape Roadmap](#9-final-trading-strategy)
10. [Statistical Reality Check](#10-statistical-reality-check)

---

## 0. DATA STRATEGY & PROVENANCE

### Data sources

The project uses TWO complementary datasets, merged into a single multi-factor file
(`Data/masi_merged.csv`) by `Output_Labs/ETAPE0/etape0_audit.py`:

| Source | Path | Provenance | Coverage | Role |
|--------|------|------------|----------|------|
| **`masi_raw.csv`** | `Data/masi_raw.csv` | Investing.com / Casablanca Stock Exchange exports (MASI index OHLC) | 2016-04-01 → 2026-04-20 (2,735 rows) | OHLC overlay + recent extension |
| **`master_dataset.csv`** | `Data/master_dataset.csv` | External research repository **`masi-risk-research-notebooks-main`** (provided by project owner). Contains MASI close + 4 most liquid Moroccan stocks (ATW, IAM, LHM, MNG) + macro factors (Brent, Gold, EUR/MAD, Geopolitical Risk Index, BAM policy rate). Original sources: Casablanca Stock Exchange, FRED, BAM, Caldara & Iacoviello GPR series. | 2007-01-31 → 2026-03-19 (4,765 rows × 85 columns) | **Multi-factor base** with extended history |

### Credits & acknowledgments

- `master_dataset.csv` provenance: the public repository `masi-risk-research-notebooks-main`
  was authored by an external researcher who pre-aggregated MASI close prices, the most
  liquid Casablanca-listed stocks, and macroeconomic factors known in the literature to
  drive MASI dynamics (cf. Belcaid & El Ghini 2021, Kharbouch & Ouaskou 2023,
  Alimoussa & Assalih 2023).
- The pre-computed derivative columns (lags, rolling statistics, log-returns) in
  `master_dataset.csv` were **independently audited for leakage** by
  `Output_Labs/ETAPE0/leakage_quickcheck.py`. All 10 tested constructions passed —
  features use strict past-only `shift(+k)` and non-centered rolling windows.
  **Nonetheless, we drop these derivative columns and recompute them with strict
  `shift(1)` in Étape 3** to enforce a single, audited source of truth.

### Why this merge (literature justification)

| Claim | Evidence | Citation |
|-------|----------|----------|
| Multi-factor models outperform univariate on MASI | GARCH-MIDAS with macro indicators (interbank rates, CPI, FX) explains long-run MASI volatility 1998-2018 | Belcaid & El Ghini 2021 |
| Monetary policy directly affects MASI | VECM (2002-2020): money supply, interbank rates, inflation drive MASI volatility long-run | Kharbouch & Ouaskou 2023 |
| CPI + GDP + REER macro impact on MASI | VAR (2002-2022): CPI positive, GDP/REER negative on MASI | Alimoussa & Assalih 2023 |
| Brent oil + commodity sensitivity in frontier markets | SSA frontier markets driven by commodity exposure | Korkpoe & Howard 2019 |
| GPR (geopolitical risk) is a global asset price driver | Caldara & Iacoviello (2022, AER) — GPR Granger-causes EM equity volatility | Caldara & Iacoviello 2022 |
| EUR/MAD regime-dependent on MASI | Frontier markets show regime-switching FX↔equity comovement | Korley & Giouvris 2021 |
| Individual liquid stocks enable cross-sectional volatility proxy (CSV) as VIX substitute | CSV from 4-10 constituents tracks VIX well in Asian markets without derivatives | Md Fadzil, O'Hara, Ng 2017 |

### Merge result (Étape 0 v2 actual figures)

| Metric | Value |
|--------|-------|
| Merged dataset | `Data/masi_merged.csv` |
| Final rows | **4,786** (vs 2,735 with masi_raw alone — +75%) |
| Final columns (raw) | 13 (date, MASI OHLC, 4 stocks, 5 macro) |
| Pre-computed derivatives dropped | 74 columns (will be re-built in Étape 3) |
| Date range | 2007-01-31 → 2026-04-20 (~19.3 years) |
| **TRAIN** | 3,348 obs (2007-02-01 → 2020-07-09) — includes **2008 crisis + COVID 2020** |
| **VAL** | 478 obs (2020-07-17 → 2022-06-20) |
| **TEST** | 948 obs (2022-06-28 → 2026-04-17) |
| Gap train→val | 8 calendar days (L6 ≥ 5 ✅) |
| Gap val→test | 8 calendar days (L6 ≥ 5 ✅) |
| Log-return ADF p | 0.0000 (stationary ✅) |
| Log-return KPSS p | 0.1000 (stationary ✅) |
| Annualized vol | 12.37% |
| Skewness | −0.81 (left tail) |
| Excess kurtosis | 12.33 (fat tails) |
| ARCH (LB on r²) | p < 0.0001 at all lags ✅ |
| GARCH(1,1) | ω=0.0622, α=0.2548, β=0.6464, persistence=0.9012 |

### Why this upgrade matters for the project

1. **Sample size**: 4,786 obs vs 2,735 → **75% more data** for Étape 5 LSTM training
2. **Regime diversity**: TRAIN now contains **2008 + COVID 2020** (two distinct Bear regimes)
   — critical for HMM regime identification (Sivakumar 2024)
3. **Multi-factor inputs**: 10 independent series justified by 4 Moroccan macro studies
4. **Reproducibility**: leakage audit + dropped pre-computed features ensures Étape 3
   builds all derivative features from a single clean source

### Anti-leakage verification ledger

| Audit step | Outcome | Script |
|------------|---------|--------|
| 10-feature L3/L4 quick-check | ✅ All past-only, no leakage in master_dataset | `Output_Labs/ETAPE0/leakage_quickcheck.py` |
| L6 temporal split assertion | ✅ Passed — 8 calendar day gap each side | `Output_Labs/ETAPE0/etape0_audit.py` |
| L1 scaler train-only | ✅ StandardScaler fit on 3,348 train obs only | `Output_Labs/etape1_preprocessing.py` |
| L4 target after features | ✅ `target_y_next` computed after cleaning | `Output_Labs/etape1_preprocessing.py` |
| L7 ffill ≤ 2 days MASI | ✅ Enforced (separate from macro ffill ≤ 30) | `Output_Labs/etape1_preprocessing.py` |

### Outputs of this upgrade

```
Data/
  ├── masi_raw.csv               (original OHLC, 2016-2026)
  ├── master_dataset.csv         (multi-factor, 2007-2026)
  └── masi_merged.csv            (NEW — merged, 2007-2026, 4,786 rows) ⭐

Output_Labs/
  ├── ETAPE0/                    (NEW — v2 audit outputs)
  │   ├── etape0_audit.py
  │   ├── etape0_audit_report.md
  │   ├── etape0_literature_synthesis.md
  │   ├── leakage_quickcheck.py
  │   └── audit_plots/           (4 PNGs)
  ├── etape1_preprocessing.py    (UPDATED — multi-factor v2)
  ├── etape1_splits/             (4 CSVs + scaler_stats JSON)
  └── etape1_plots/              (4 PNGs)
```

---

## 1. EXECUTIVE SUMMARY — The Recommended Strategy Stack

Based on the synthesis of **19 peer-reviewed papers**, the **highest-probability strategy** for MASI is:

| Component | Choice | Justification | Source |
|-----------|--------|---------------|--------|
| **Target** | y_t = ln(P_{t+1}/P_t) (next-day log-return) | Stationary, standard, interpretable | All papers |
| **Volatility proxy** | **GARCH(1,1) primary + rolling 21d fallback** | ARCH confirmed in MASI; Realized GARCH requires intraday data (UNAVAILABLE for MASI) | Hansen 2021, Korkpoe 2019, our audit |
| **Regime model** | **3-state Gaussian HMM** trained on TRAIN only | 3 states (Bull/Neutral/Bear) outperforms 2-state in frontier markets; Sivakumar 2024 confirms | Sivakumar 2024, Rikken 2022, Korkpoe 2019 |
| **HMM-DL integration** | **HMM hidden states + state means as additional LSTM features** ("Regime-as-Feature") | Sivakumar 2024: MSE 0.86 (with HMM features) vs 1.99 (without) — 57% reduction | Sivakumar 2024, Zheng 2023 |
| **DL architecture** | **Compact LSTM (1 layer, 32-64 units)** with optional CNN front-end | Oukhouya 2023 found SVR/MLP > LSTM on 541-obs MASI (small data); use **L=10 or L=20**, NOT L=30 | Oukhouya 2023, Sivakumar 2024 |
| **Scaling** | **z-score (StandardScaler)** fit on TRAIN only | More robust to outliers than min-max (Nguyen 2025); fat tails confirmed in MASI | Nguyen 2025, our audit |
| **Validation** | **Walk-forward with sequential 70/10/20 + 5-day gap + post-split sequence generation** | 10-fold CV produces up to 20.5% RMSE inflation due to leakage; 3-way split safest | Albelali 2025, Deep 2025 |
| **Sequence generation** | **AFTER splitting, never before** | The single most violated rule in the literature | Albelali 2025 |
| **Baselines (MANDATORY first)** | RW → Historical Mean → ARIMA → Random Forest | DL is FORBIDDEN until all 4 are evaluated | Deep 2025, prompt.md |
| **Transaction costs** | **5–10 bps for MASI** | Standard for emerging/frontier (Shu 2024: 5bps; Deep 2025: 5bps) | Shu 2024, Deep 2025 |
| **Signal execution** | Signal at t → execute OPEN of t+1 | Anti-lookahead bias | Deep 2025 |
| **Trading rule** | Long-only if P(Bull regime) × predicted_return > threshold | MASI is long-only equity index | Touzani 2021 |
| **Out-of-sample folds** | 5–8 walk-forward folds (NOT 34 like Deep 2025) | Adapted to MASI's ~2,500 obs (Deep 2025 had 100 US stocks × 10 years = abundant) | Adapted |

### Realistic Expected Performance (based on literature)

- **Annualized return:** 5–20% (Touzani 2021: 27.13%; Deep 2025: 0.55%; reality probably in between)
- **Sharpe ratio:** 0.3–0.8 (Deep 2025: 0.33; Monteiro 2025: 0.77)
- **Maximum drawdown:** −5% to −15% (Kemper 2025: 50% volatility reduction over benchmark; Deep 2025: −2.76%)
- **Statistical significance:** **likely NOT significant at conventional levels** (Deep 2025 p=0.34 with 34 folds; we'll have fewer) — this is **expected and honest**, not a failure

---

## 2. ANTI-LEAKAGE FRAMEWORK (P1)

### Paper 1.1 — Albelali & Ahmed 2025 (KFUPM), "Hidden Leaks in Time Series Forecasting"

**Core finding:** Sequence generation BEFORE splitting causes RMSE inflation up to **20.5%** under 10-fold CV.

**Validation strategy ranking (most leaky → least leaky):**
1. **10-fold CV** — up to **20.5% RMSE Gain** (leaky); avoid for time series
2. **3-way split (70/10/20)** — moderate, supports early stopping
3. **2-way split (80/20)** — **most robust** (under 5% RMSE Gain in all configs)

**Architectural factors that EXACERBATE leakage:**
- Smaller windows (window=3 → more leakage than window=10)
- Larger lag steps (lag=3 → 20.51% RMSE Gain under 10-fold)

**Mandatory rule:** **POST-SPLIT sequence generation** — split data first, then build sliding windows independently within each partition. Buffer zones between folds if k-fold required.

### Paper 1.2 — Deep, Deep & Lamptey 2025 (Texas Tech), "Interpretable Hypothesis-Driven Trading"

**Walk-forward protocol (gold standard):**
- Window W = 252 days (train), H = 63 days (test), step Δ = 63 days
- 34 independent test periods over 10 years
- ε-greedy: ε_train=0.7 (explore), ε_test=0.1 (exploit) — strict separation
- Transaction cost: **$1 fixed + 5 bps slippage**, orders execute at t+1 OPEN
- Position constraints: max 5 concurrent, 20% per position, 50% per sector

**Honesty findings:**
- 100 US stocks, 2015–2024: 0.55% annualized return, Sharpe 0.33, MDD −2.76%
- **p-value = 0.34 (NOT statistically significant)** — reported transparently
- Statistical power = 12% → would need 540 folds for 80% power
- **Regime dependence:** +0.60% quarterly in high-vol (2020–2024) vs −0.16% in low-vol (2015–2019)
- Bear market 2022 was hardest (−0.70%, 0% win rate)

**Statistical tests used:** t-test, bootstrap (10,000 resamples), Monte Carlo permutation (10,000 shuffles), binomial, Shapiro-Wilk.

### Anti-Leakage Rules — Operationalized for MASI (L1–L8)

| Rule | Description | Mitigation |
|------|-------------|------------|
| **L1** | StandardScaler/normalizer fit only on TRAIN | `scaler.fit(train); scaler.transform(val,test)` |
| **L2** | HMM trained on TRAIN only; forward-predict on val/test | `hmm.fit(train); hmm.predict(val); hmm.predict(test)` |
| **L3** | Rolling features use `.shift(1)` (closed='left') | Never centered windows |
| **L4** | Target y_t computed AFTER features | Features at t, y_t = ln(P_{t+1}/P_t) |
| **L5** | Signal at t → execute at OPEN of t+1 | Use Open[t+1] not Close[t] for entry |
| **L6** | 5-day gap between train end and val start; same val→test | Hard assertion |
| **L7** | MASI holidays: forward-fill max 2 consecutive days | If gap > 5 days → segment boundary |
| **L8** | GARCH re-estimated WITHIN each walk-forward window | No global GARCH parameters |

---

## 3. MASI-SPECIFIC FINDINGS (P2)

### Paper 2.1 — Touzani & Douzi 2021, "LSTM/GRU trading strategy for Moroccan market" (J. Big Data)

**Critical MASI facts:**
- MASI has **76 stocks total** — narrow universe
- **Trading discontinuity**: many stocks not traded for days/weeks/months (low liquidity)
- COVID period 2020 included in test
- **27.13% annualized strategy return** vs 0.43% MASI buy-hold (sector pharma: 19.94%, distributor: 15.24%)

**Two innovations from this paper:**
1. **Transfer learning**: train LSTM on US + French markets; use MASI only for validation+test (compensates for data scarcity)
2. **Target = moving average of next h days**, not raw next-day return (smooths noise, easier to predict)

**Brokerage fees subtracted from returns** — realistic backtesting.

### Paper 2.2 — Oukhouya & El Himdi 2023, "SVR, XGBoost, LSTM, MLP for MSI 20" (Comput. Sci. Math. Forum)

**Surprising result on small Moroccan data (541 obs):**

Performance ranking: **SVR > MLP > LSTM > XGBoost**

- SVR optimal: kernel=linear, C=1000, ε=0.001 → MAE 3.092, RMSE 3.993, MAPE 0.368%
- **XGBoost performed WORST** — insufficient training data
- LSTM: 2 hidden layers, neurons (250,200), 100 epochs, patience 4

**Implication for MASI:** With limited data, **simpler models can beat deep learning**. We must validate this empirically and accept SVR/RF over CNN-LSTM if they win.

### Paper 2.3 — Oukhouya, Kadiri, El Himdi, Guerbaz 2024, "Hybrid LSTM-XGBoost for International Stocks" (SOIC)

**6 international indices including MASI** (March 2018–March 2023, 1,259 days):
- MASI-tuned LSTM: batch=4 (tiny!), neurons (300,300), epochs=100, patience=4
- MASI-tuned XGBoost: 300 trees, depth=20, lr=0.02
- **LSTM-XGBoost residual hybrid** (XGBoost trained on LSTM residuals) outperforms standalone

**5-fold blocked time series CV** + **skforecast** library for backtesting + bootstrap prediction intervals (Q=0.1, Q=0.9 → 80% PI).

### Paper 2.4 — Talhartit, Ait Jillali, El Kabbouri 2025, "Theoretical Approach for Moroccan Market" (IJMSBR)

**Key MASI references this paper synthesizes:**

| Author | Method | Finding |
|--------|--------|---------|
| Belcaid & El Ghini 2021 | GARCH-MIDAS (1998-2018) | Macro indicators (interbank rates, inflation) explain long-term MASI volatility |
| Kharbouch & Ouaskou 2023 | VECM (2002-2020) | Monetary policy + interbank rates + inflation drive long-run MASI volatility |
| Elbousty & Oubdi 2020 | Stylized facts | MASI exhibits **volatility clustering, long memory, asymmetry** |
| Bourezk et al. 2020 | Bi-LSTM + Random Forest hybrid | Notable accuracy on MASI |
| Razouk et al. 2023 | Twitter sentiment | Islamic finance perception in Morocco |
| Alimoussa & Assalih 2023 | VAR (2002-2022) | CPI positively impacts MASI; GDP/REER negatively |
| El Yamani 2023 | VAR (2008-2021) | Market cap + liquidity → economic growth (reverse causality) |

**Implication:** Adding macro features (interbank rates, inflation, FX) is theoretically motivated for MASI. **Out of scope for our first pass** but a viable Étape 6+ extension.

---

## 4. HMM + DEEP LEARNING ARCHITECTURE (P3)

### Paper 3.1 — Sivakumar 2024 (arXiv), "HMM-LSTM Fusion Model for Economic Forecasting"

**THE blueprint paper for our project.**

**Architecture:**
- HMM (3-state Gaussian, full covariance) trained first
- Extract: (a) hidden state sequence, (b) state means (μ_state)
- Feed both as **additional features to LSTM**
- LSTM: 2 layers (tanh + sigmoid) + Dense (linear) + output

**Quantitative impact of HMM features (CPI inflation, 647 monthly obs):**

| Input | MSE | MAE | R² |
|-------|-----|-----|-----|
| Original only | 1.992 | 0.964 | 0.519 |
| + Hidden states | 3.548 | 1.290 | 0.144 |
| + Means | 0.818 | 0.694 | 0.803 |
| + States + Means | **0.861** | **0.705** | **0.792** |

**KEY INSIGHT:** The **means matter more than the hidden states**. Always include both.

**Validation:** 5-fold forward-chaining CV (gives COVID and 2008 their own folds). **Integrated Gradients** for feature importance.

**Future direction acknowledged:** "Markov Switching Neural Networks" — embed regime layer INSIDE the network. Too complex for our Étape 5 first pass.

### Paper 3.2 — Kemper 2025 (SSRN), "Hybrid HMM-LSTM for Semiconductor Equities"

**Risk-management overlay (not alpha-generation):**
- HMM + LSTM regime predictions fused via **entropy-weighted Bayesian model averaging**
- Real-time risk engine adjusts position sizing based on (volatility, drawdown, model confidence)
- Applied to SMH/NVDA/AMD/TSM, 2019–2024
- Result: **50% volatility reduction, 15–17 pp drawdown improvement** vs passive

**Implication for MASI:** This is the right framing — **regime detection's primary value is RISK MANAGEMENT, not return prediction**. Adjust position size in Bear regime, full exposure in Bull, neutral in Neutral.

### Paper 3.3 — Monteiro 2025 (QFE), "AI-Powered Energy Algorithmic Trading"

**Dual-model alpha:**
- 5-state Gaussian HMM (Viterbi for optimal state sequence)
- Feedforward NN: 4 hidden layers ReLU (10, 10, 10, 5)
- **Buy ONLY when HMM state prediction AND NN signal AGREE**
- Best state selected by: argmax_j E[r | state=j]
- 5 epochs only — aggressive overfit prevention
- COVID period (2019–2022): **83% return, Sharpe 0.77**

**Key technique we'll borrow:** "Signal agreement" — combine HMM regime probability AND CNN-LSTM return prediction; only execute if both agree.

### Paper 3.4 — Shu, Yu, Mulvey 2024 (Princeton, arXiv), "Dynamic Asset Allocation with Asset-Specific Regime Forecasts"

**Asset-SPECIFIC regimes** (not broad economic regimes):
- Step 1: Statistical Jump Model (unsupervised) labels each historical period
- Step 2: Gradient-boosted decision tree (supervised) FORECASTS regime
- Step 3: Regime forecast → Markowitz mean-variance optimization

**Transaction cost: 5 bps** (matches MASI!). 12 asset universe, 1991–2023.

**Two key ideas for MASI:**
1. Regimes specific to MASI itself, not global macro
2. Forecast NEXT regime, not just classify current — use forecast in optimization

### Paper 3.5 — Tiwari et al. 2025 (Ingénierie des Systèmes d'Information), "CNN-LSTM + TFT for AAPL"

**Most complex architecture seen:**
- Hybrid: H_t = CNN(X_t) + LSTM(X_t), output z_t fed into Temporal Fusion Transformer (TFT)
- 180-day lookback, 128 hidden, 8 attention heads, 2 LSTM layers, 0.2 dropout
- Adam lr=5e-4, weight decay 1e-4, 80 epochs, batch 256
- AAPL 2012-2025, robust scaling by ticker (median/IQR)
- Long/short return 80.7% vs buy-hold 38%
- **~12% lower RMSE vs baseline LSTM**

**TOO COMPLEX FOR MASI** (1,057-2,500 obs). However:
- **Robust scaling (median/IQR)** instead of z-score worth testing (better with outliers)
- TFT's **quantile forecast** for uncertainty would be valuable if data permitted
- Documented for Étape 6+ extension only

---

## 5. FRONTIER MARKET CONSTRAINTS (P4)

### Paper 4.1 — Korley & Giouvris 2021 (J. Risk Financial Mgmt), "Regime-Switching for Frontier SSA Markets"

**CRITICAL FINDING for MASI** (frontier market like Sub-Saharan Africa):

In frontier markets, the relationship between regime and returns is **OPPOSITE to emerging markets**:

| Market type | High-vol regime | Low-vol regime |
|-------------|-----------------|----------------|
| Developed/Emerging | High vol + LOW returns | Low vol + high returns |
| **Frontier (incl. MASI?)** | **High vol + HIGH returns** | Low vol + low returns |

**Implication:** Don't assume "high volatility = bear regime". For MASI we must:
- Re-test this empirically with 3-state HMM
- Allow Bull state to coexist with elevated volatility
- The "regime" label should be data-driven, not pre-assumed

Markov-switching VAR; weekly data Côte d'Ivoire, Ghana, Kenya, Mauritius, Nigeria 2000–2018.

### Paper 4.2 — Korkpoe & Howard 2019 (Emerging Markets J.), "Volatility Model Choice for SSA Frontier"

**Bayesian MCMC model selection across 12 models** (Single & 2-state × {GARCH, EGARCH, GJR-GARCH} × {normal, skewed-t, student-t}):

| Country | Best model |
|---------|-----------|
| Ghana | **2-state GJR-GARCH(1,1) with skewed student-t** |
| Nigeria | 2-state GJR-GARCH(1,1) with student-t |
| Kenya | 2-state EGARCH(1,1) with skewed student-t |
| Botswana | 2-state EGARCH(1,1) with skewed student-t |

**Three key takeaways for MASI:**
1. **2-state regime-switching BEATS single-state** in every case (DIC criterion)
2. **Asymmetric variants (GJR, EGARCH) beat plain GARCH** — leverage effect matters
3. **Student-t innovations beat Gaussian** — fat tails (our audit found excess kurtosis 7.98)

**Étape 4 upgrade:** Test GJR-GARCH or EGARCH with student-t (not just GARCH(1,1)) once Étape 0 GARCH is validated.

### Paper 4.3 — Robeco 2024, "5-Year Expected Returns: Triple Power Play" (industry)

Macro outlook, not methodology. Notes that emerging markets have AI/automation tailwinds; not directly applicable.

---

## 6. VOLATILITY PROXY WITHOUT VIX (P5)

### Paper 5.1 — Hansen, Huang, Tong, Wang 2021 (arXiv econ.EM), "Realized GARCH, CBOE VIX, VRP"

**Realized GARCH model** (joint return + realized variance):
- r_t = r + λ√h_t − ½h_t + √h_t z_t
- log h_{t+1} = ω + β log h_t + τ(z_t) + γσ u_t
- log x_t = κ + φ log h_t + δ(z_t) + σ u_t  (x_t = realized variance from HF data)
- Dual-shock (z_t return, u_t volatility) — better than standard GARCH for VIX modeling

**HARD CONSTRAINT FOR MASI:** Realized GARCH **REQUIRES HIGH-FREQUENCY (intraday) DATA** to compute realized variance. **MASI provides daily data only.** → **Realized GARCH is REJECTED for MASI.**

**Conclusion:** Standard GARCH(1,1) is the appropriate fallback. Optionally upgrade to GJR-GARCH or EGARCH (asymmetric) per Korkpoe 2019.

### Paper 5.2 — Md Fadzil, O'Hara, Ng 2017 (Cogent Econ. Fin.), "Cross-Sectional Volatility Index (CSV) as VIX Proxy in Asian Markets"

**The most directly relevant paper for MASI's "no-VIX" problem.**

**CSV definition (equally weighted):**

$$ \text{CSV}_t^{EW} = \frac{1}{N_t} \sum_{i=1}^{N_t} (r_{it} - \bar{r}_t^{EW})^2 $$

Where r_it = return of stock i at time t, N_t = number of constituents.

**Properties:**
- As N_t → ∞, CSV converges to σ²_ε (idiosyncratic variance)
- **Model-free**, computable at any frequency
- Strong correlation with VIX in developed markets (validated on Japan)
- **Requires individual constituent data**, not just the index

**Implication for MASI:** If individual stock data for MASI's 76 constituents is collected, **CSV becomes a viable VIX proxy**. For now (index-level only) → use GARCH(1,1). **CSV listed as future extension (Étape 7+).**

---

## 7. DEEP LEARNING PORTFOLIO STRATEGIES (P6)

### Paper 6.1 — Nguyen 2025 (PLOS One), "DL Portfolio Optimization for VN-100 Vietnam"

**Direct emerging-market analog** (Vietnam ~= Morocco in liquidity profile):
- VN-100, 2017–2024, 59 stocks after filtering
- 30-day lookback, **z-score** standardization (preferred over min-max due to outlier robustness)
- Chronological split: Train 2017-2021, Val 2022, Test 2023-2024
- Three portfolio frameworks tested:
  - **MVF** (Mean-Variance with Forecasting) — return-seeking
  - **RPP** (Risk Parity Portfolio) — moderate risk
  - **MDP** (Maximum Drawdown Portfolio) — conservative

**Result:** **LSTM > 1D-CNN**; LSTM+MVF best risk-adjusted; LSTM+MDP best total return.

**For MASI single-index forecast** (not multi-asset portfolio): not directly applicable but the scaling and split methodology are reference-quality.

### Paper 6.2 — Fozap 2025 (J. Risk Financial Mgmt), "Hybrid LSTM-CNN with Technical Indicators for S&P 500"

**Architecture:** LSTM (temporal) + CNN (spatial pattern in indicators)
- Result vs Random Forest: RF wins RMSE/R² but lacks sequential learning
- LSTM-CNN: RMSE 0.1012, MAE 0.0800, MAPE 10.22%, R² 0.4199 (S&P 500, 2010-2024)

**Technical indicators added as features (recommended for MASI Étape 3):**
- **SMA**(10, 50)
- **EMA**(10, 50)
- **Bollinger Bands** (20-day SMA ± k·σ)
- **MACD**
- **RSI**

Forward-fill imputation for missing values. Feature selection via correlation + domain expertise.

### Paper 6.3 — Kevin & Yugopuspito 2025 (arXiv), "Hybrid LSTM + PPO for Dynamic Portfolio"

**LSTM forecaster + PPO (Proximal Policy Optimization) RL agent:**
- 4 asset classes (US equities, Indonesian equities, US Treasuries, top-10 cryptos), 2018-2024
- 30-week lookback, 64 hidden units, 0.2 dropout, lr=1e-3, batch 64, 40 epochs
- **Z-score fit on TRAIN only** (L1 compliant)
- 70/30 train/test, weekly resampled
- Transaction cost: 0.1% per turnover
- Hybrid LSTM+PPO > LSTM-only > PPO-only > equal-weight

**Implication for MASI:** PPO is **too complex** for our data size. Use **deterministic allocation** based on LSTM forecast threshold instead. Save RL for Étape 7+ extension.

---

## 8. CROSS-PAPER CONSENSUS & CONFLICTS

### Strong Consensus (≥4 papers agree)

| Claim | Supporting papers |
|-------|------------------|
| Sequence generation must be POST-split | Albelali 2025, Deep 2025, Nguyen 2025, Kevin 2025 |
| Walk-forward beats k-fold for time series | Albelali 2025, Deep 2025, Sivakumar 2024, Shu 2024 |
| 2- or 3-state regime-switching beats single-state | Korkpoe 2019, Korley 2021, Sivakumar 2024, Kemper 2025 |
| HMM regimes improve LSTM accuracy | Sivakumar 2024, Kemper 2025, Monteiro 2025, Shu 2024 |
| ARCH/volatility clustering universal in equity returns | Hansen 2021, Korkpoe 2019, Talhartit 2025, our audit |
| Transaction costs realistic = 5–10 bps for emerging | Deep 2025, Shu 2024, Kevin 2025 (0.1%) |
| z-score preferred over min-max for outlier-heavy data | Nguyen 2025, Kevin 2025 (vs Sivakumar 2024, Oukhouya 2023, Fozap 2025 min-max) |

### Moderate Consensus (2–3 papers)

- **LSTM > GRU** in stock prediction (Touzani 2021 finds them similar; Chung 2014, Jozefowicz 2015 cited as conflicting)
- **DL > ARIMA on large data** — Siami-Namini 2018 (84-87% error reduction), Fozap 2025
- **DL ≤ ARIMA/RF on small data** — Oukhouya 2023 (SVR best on 541 obs MASI), Albelali 2025
- **3-state HMM > 2-state HMM** (Rikken 2022 cited by Sivakumar 2024) but 2-state simpler to label (Korkpoe 2019)

### Genuine Conflicts (must resolve empirically)

| Conflict | Side A | Side B | Our resolution |
|----------|--------|--------|----------------|
| LSTM vs SVR/MLP on MASI | LSTM works (Touzani 2021) | SVR > LSTM (Oukhouya 2023) | **Test both; choose by walk-forward Sharpe** |
| Min-max vs z-score scaling | Min-max (Sivakumar 2024, Oukhouya 2023) | Z-score (Nguyen 2025, Kevin 2025) | **Z-score** (fat tails confirmed; outlier robustness) |
| High-vol regime returns | High vol = LOW return (developed/emerging) | High vol = HIGH return in frontier (Korley 2021) | **Let HMM learn it from MASI data** |
| Window size | L=180 (Tiwari 2025), L=30 (Nguyen 2025) | L=10-20 for small data (prompt.md, Oukhouya 2023) | **L ∈ {10,15,20}** — test all |
| Direct return vs MA target | Direct y_t = log return (most papers) | MA of next h days (Touzani 2021) | **Direct log return primary**; MA as robustness check |

---

## 9. FINAL TRADING STRATEGY — Étape-by-Étape Roadmap

### Étape 0 — Data Audit ✅ COMPLETE
- 1,057 obs loaded; ADF/KPSS/ARCH confirmed
- GARCH(1,1) parameters: ω=0.115, α=0.448, β=0.482, persistence=0.93
- **Action required:** Merge `masi_raw.csv` to extend coverage to ~2,500 obs

### Étape 1 — Preprocessing (NEXT)
1. **Merge all MASI CSV files** (2016-2026 target)
2. Forward-fill ≤2 days; segment gaps >5 days (L7)
3. Compute y_t = ln(P_{t+1}/P_t) **AFTER** all features (L4)
4. Detect non-trading days via consecutive zero log-returns (volume column empty)
5. Temporal split 70/10/20 + 5-day gap (L6)
6. **z-score** fit on TRAIN only (L1)
7. Output: `Output_Labs/etape1_preprocessing.py`, `etape1_report.md`

### Étape 2 — Mandatory Baselines (DL FORBIDDEN until done)
Order (per prompt.md):
1. **Random Walk** (naive drift)
2. **Historical Mean**
3. **ARIMA** (auto-lag via AIC/BIC)
4. **Random Forest** (100 trees, walk-forward)
5. *(optional)* XGBoost, LightGBM

Metric set: RMSE, MAE, **Directional Accuracy**, **Sharpe (annualized)**, **MDD**.
- Apply transaction costs: **5 bps** (one-way) per trade
- Confidence intervals via bootstrap (10,000 resamples, Deep 2025)

### Étape 3 — Feature Engineering
**Features (F ≤ 15):**
1. log_return_t (target lag)
2. log_return_5d_MA, log_return_21d_MA (shift(1))
3. **GARCH(1,1) conditional volatility** σ_t (re-estimated per window, L8)
4. realized_vol_21d (rolling)
5. SMA_10, SMA_50 (shift(1))
6. EMA_10, EMA_50 (shift(1))
7. Bollinger %B (position within bands)
8. RSI_14 (shift(1))
9. MACD_signal (shift(1))
10. ATR_14 (if OHLC complete, shift(1))
11-13. *(reserved for HMM features in Étape 4)*

All rolling features → `closed='left'` or `.shift(1)` (L3).
All features stationarity-tested before inclusion (ADF/KPSS).

Output: `etape3_features.py`, `etape3_leakage_audit.md`

### Étape 4 — HMM Regime Detection
1. **Test 2-state vs 3-state** via BIC/AIC + log-likelihood
2. Gaussian HMM with full covariance (Sivakumar 2024)
3. **Train on TRAIN only** (L2); forward-predict on val/test
4. Extract: hidden_state_t (one-hot), state_mean_t (continuous)
5. Append BOTH to feature set (Sivakumar 2024: both matter)
6. *(stretch)* Test GJR-GARCH or EGARCH innovations (Korkpoe 2019)
7. Interpret states by mean return + volatility — DON'T assume "high-vol = bear" for MASI (Korley 2021)

Output: `etape4_hmm.py`, `etape4_regime_report.md`

### Étape 5 — CNN-LSTM (only if baselines insufficient)
**Architecture (max — per prompt.md constraints):**
- Conv1D: 1 layer, 32 filters, kernel=3
- LSTM: 1 layer, 32 units
- Dropout 0.2-0.3
- Dense 16 units
- Output 1 neuron (regression)

**Hyperparameters:**
- L ∈ {10, 15, 20} — test all (Albelali 2025: smaller windows = more leakage risk; balance with data size)
- Optimizer: Adam, lr=1e-3 (or 5e-4 for fine-tune phase)
- Loss: **Huber** (robust to fat tails, NOT pure MSE)
- Early stopping: patience=5-10
- Batch size: 16-32 (small data)
- Epochs: 50-100 max
- **Z-score on TRAIN only**
- **Sequences generated POST-split** (L4 reinforcement)

**Walk-forward folds:** 5-8 (NOT 34) — adapted to MASI size
**Signal rule (Monteiro-inspired):** Execute long ONLY IF (HMM_p(Bull) > 0.5) AND (CNN-LSTM predicted return > threshold)

Output: `etape5_cnn_lstm.py`, `etape5_walkforward_report.md`

### Étape 6 — Backtest
Per Deep 2025 + Shu 2024:
- Transaction costs: 5 bps slippage + nominal commission
- Signal at t → execute at OPEN of t+1 (L5)
- Position limit: 100% long-only OR 0% cash (binary regime gate)
- Exit: opposite signal OR stop-loss (4-5% per Deep 2025)
- Metrics: Annualized return, Sharpe, Sortino, MDD, Calmar, t-test, bootstrap CI, permutation test
- **Benchmark vs MASI buy-and-hold** AND vs each baseline

Statistical reporting (mandatory):
- p-value for mean return ≠ 0
- 95% bootstrap CI for return
- Statistical power (expect <30% given small sample)
- Regime-conditional performance (Bull vs Bear fold returns)

Output: `etape6_backtest.py`, `etape6_final_report.md`

### Étape 7+ (FUTURE EXTENSIONS — out of current scope)
- Collect MASI constituent stock data → build **CSV (Cross-Sectional Volatility Index)** as VIX proxy
- Add macro features: interbank rate, inflation, MAD/EUR FX
- **Transfer learning** (Touzani 2021): pre-train on CAC40 or other related index
- Bayesian model averaging (Kemper 2025) of HMM + LSTM probabilities
- RL allocation (Kevin 2025) — only if dataset grows significantly

---

## 10. STATISTICAL REALITY CHECK

### Honest Performance Expectations for MASI

**Per Deep 2025 (100 US stocks, 10 years, 34 folds):**
- Annualized return: 0.55%
- Sharpe: 0.33
- p-value: **0.34** (NOT statistically significant)
- Statistical power: 12% (need 540 folds for 80% power)

**Per Touzani 2021 (MASI, 4× 3-year periods):**
- Strategy return: 27.13% annualized
- BUT: small sample, includes COVID extreme period, transfer-learning approach

**Honest middle ground for MASI:**

| Scenario | Annualized Return | Sharpe | MDD | p-value |
|----------|-------------------|--------|-----|---------|
| Optimistic (best fold) | 10-20% | 0.6-0.9 | -10% to -15% | 0.05-0.15 |
| Realistic (avg fold) | 3-8% | 0.2-0.5 | -10% to -20% | 0.20-0.50 |
| Pessimistic | -2 to +3% | -0.1 to +0.2 | -25% | > 0.50 |

### What "Success" Means for This Project

**Success ≠ beating MASI by 20% annually.**

Success is:
1. ✅ Building a **leakage-free** pipeline that produces results we can trust
2. ✅ Demonstrating that **HMM regime features improve** the LSTM marginally (even 5% RMSE improvement is publishable)
3. ✅ **Risk-adjusted** improvement (lower drawdown OR higher Sharpe) vs buy-and-hold
4. ✅ Identifying **regime-conditional** performance (model works in regime X, fails in regime Y)
5. ✅ Reporting results with **honest confidence intervals** and **non-significant p-values when applicable**

This matches Deep 2025's philosophy and is **the only honest standard** for academic-quality quantitative research on a frontier market with ~2,500 daily observations.

---

## APPENDIX A — Paper Reading Log (19 papers covered)

| # | Folder | Paper | Status |
|---|--------|-------|--------|
| 1 | P1 | Albelali & Ahmed 2025 — Hidden Leaks in TS Forecasting | ✅ Full read |
| 2 | P1 | Deep, Deep & Lamptey 2025 — Interpretable Walk-Forward | ✅ Full read |
| 3 | P2 | Touzani & Douzi 2021 — LSTM/GRU for Moroccan market | ✅ Full read |
| 4 | P2 | Oukhouya & El Himdi 2023 — SVR/XGBoost/LSTM for MSI-20 | ✅ Full read |
| 5 | P2 | Oukhouya et al. 2024 — LSTM-XGBoost international | ✅ Full read |
| 6 | P2 | Talhartit et al. 2025 — Theoretical macro+sentiment+DL | ✅ Full read |
| 7 | P2 | Oukhouya & El Himdi 2024 (book chapter, abstract) | ⚠️ Partial (abstract only) |
| 8 | P3 | Sivakumar 2024 — HMM-LSTM Fusion CPI | ✅ Full read |
| 9 | P3 | Kemper 2025 — Bayesian HMM-LSTM semiconductor | ✅ Partial read |
| 10 | P3 | Monteiro 2025 — HMM + NN energy trading | ✅ Partial read |
| 11 | P3 | Shu, Yu, Mulvey 2024 — Asset-specific regime forecasts | ✅ Partial read |
| 12 | P3 | Tiwari et al. 2025 — CNN-LSTM+TFT for AAPL | ✅ Partial read |
| 13 | P4 | Korley & Giouvris 2021 — MS-VAR for Sub-Saharan frontier | ✅ Partial read |
| 14 | P4 | Korkpoe & Howard 2019 — MS-GARCH Bayesian for frontier | ✅ Full read |
| 15 | P4 | Robeco 2024 — 5-Year Expected Returns | ⚠️ Industry outlook (non-technical) |
| 16 | P5 | Hansen et al. 2021 — Realized GARCH, VIX, VRP | ✅ Partial read |
| 17 | P5 | Md Fadzil, O'Hara, Ng 2017 — CSV as VIX proxy | ✅ Partial read |
| 18 | P6 | Nguyen 2025 — DL Portfolio for VN-100 | ✅ Partial read |
| 19 | P6 | Fozap 2025 — Hybrid LSTM-CNN with TA indicators | ✅ Partial read |
| 20 | P6 | Kevin & Yugopuspito 2025 — LSTM + PPO portfolio | ✅ Partial read |

Duplicate `isi_30.11_22.pdf` appeared in both P2 and P3 folders — counted once.

---

## APPENDIX B — Methods Explicitly REJECTED for MASI

| Method | Reason for rejection | Source |
|--------|---------------------|--------|
| **Realized GARCH** | Requires intraday data (MASI = daily only) | Hansen 2021 |
| **10-fold CV (non-temporal)** | Up to 20.5% RMSE inflation due to leakage | Albelali 2025 |
| **Random train/test split** | Violates temporal causality | All P1 papers |
| **Pre-split sequence generation** | The #1 leakage source in TS forecasting | Albelali 2025 |
| **TFT (Temporal Fusion Transformer)** | Too many parameters for ~2,500 obs | Constraint C3 |
| **PPO/RL allocation** | Too data-hungry; unstable on small samples | Kevin 2025 acknowledged |
| **Pure MSE loss for fat-tailed returns** | Overweights extreme observations | Implicit in Korkpoe 2019, Hansen 2021 |
| **Min-Max scaling** | Sensitive to outliers (excess kurtosis = 7.98 in MASI) | Nguyen 2025 |
| **L=30 default input window** | Constraint C3 (limited data); larger windows = stable BUT smaller windows = test for robustness | Albelali 2025 + prompt.md |
| **CSV proxy (current scope)** | Requires individual MASI constituent stock data — not currently collected | Md Fadzil 2017 |
| **Markov Switching Neural Networks** | Implementation complexity exceeds Étape 5 scope | Sivakumar 2024 (future work) |
| **N-BEATS / Transformer-based forecasters** | Too data-hungry | Nguyen 2025 acknowledged |

---

*End of README — synthesized from 19 papers in 6 priority folders (Étape 0 literature complete, 2026-05-19).*
*Next action: USER validation ("validé") → proceed to Étape 1 — Data Collection & Preprocessing.*
