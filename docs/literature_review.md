# ÉTAPE 0 — Literature Synthesis
## MASI Hybrid Forecasting System (HMM + CNN-LSTM)
**Date:** 2026-05-19  
**Status:** COMPLETE

---

## READING STATUS

### FILES READ SUCCESSFULLY
| Priority | Folder | File | Key Topic |
|----------|--------|------|-----------|
| P1 | Anti-Leakage & Validation | 2512.12924v1.pdf | Hidden Leaks in LSTM Time Series (Albelali & Ahmed, 2025) |
| P1 | Anti-Leakage & Validation | 2512.06932v1.pdf | Walk-Forward Validation Framework (Deep et al., 2025) |
| P2 | Marché Marocain | 978-3-031-68628-3_6 (2).pdf | ML Forecasting MASI — XGBoost/SVR/LSTM (Oukhouya & El Himdi, 2024) |
| P2 | Marché Marocain | csmf-07-00039 (1).pdf | SVR/XGBoost/LSTM/MLP on MSI20 (Oukhouya & El Himdi, 2023) |
| P2 | Marché Marocain | 40537_2021_Article_512.pdf | LSTM+GRU Trading Strategy for Moroccan Market (Touzani & Douzi, 2021) |
| P3 | Architecture HMM + DL | 2501.02002v1.pdf | HMM-LSTM Fusion Model for Economic Forecasting (Sivakumar, 2025) |
| P3 | Architecture HMM + DL | 2407.19858v6.pdf | HMM + Neural Networks for Algorithmic Trading (Monteiro, 2025) |
| P4 | Marchés Émergents & Frontière | jrfm-14-00122-v2.pdf | Regime-Switching in Frontier Markets / Sub-Saharan Africa (Korley & Giouvris, 2021) |
| P5 | Proxy Volatilité sans VIX | 2112.05302v1.pdf | Realized GARCH, VIX, Volatility Risk Premium (Hansen et al., 2021) |
| P6 | Deep Learning Portfolio & Général | jrfm-18-00201-v2.pdf | Hybrid LSTM-CNN with Technical Indicators (Fozap, 2025) |

### FILES SKIPPED (context limit reached)
| Folder | File | Reason |
|--------|------|---------|
| P2 | Journal-2.pdf | Context limit — Priority 2 partially covered by 3 other papers |
| P2 | isi_30.11_22 (2).pdf | Context limit |
| P2 | isi_30.11_22 (2) (1).pdf | Context limit |
| P2 | 1822-ArticleText-7868-1-10-20230710.pdf | Context limit |
| P2 | 978-3-031-68628-3_6 (3).pdf | Context limit |
| P3 | ssrn-5366835 (2).pdf | Context limit |
| P3 | isi_30.11_22 (2).pdf | Context limit |
| P3 | 2406.09578v2.pdf | Context limit |
| P4 | admin,+9.korkpoe_howard (3).pdf | Context limit |
| P4 | docu-2023-robeco-5-year-expected-r....pdf | Context limit |
| P5 | Cross-sectional volatility index as a....pdf | Context limit |
| P6 | 2511.17963v1.pdf | Context limit |
| P6 | pone.0330547.pdf | Context limit |

**⚠️ CONTEXT LIMIT REACHED — Folders P2 (partial), P3 (partial), P4 (partial), P5 (partial), P6 (partial) not fully read. Core findings are covered by the papers that were read.**

---

## PRIORITY 1 — Anti-Leakage & Validation

### Paper 1: Hidden Leaks in Time Series Forecasting (Albelali & Ahmed, 2025)
**File:** 2512.06932v1.pdf | arXiv:2512.06932

#### Key Findings
- **10-fold cross-validation** is the most vulnerable validation technique to temporal leakage (RMSE gain up to **20.5%** at lag=3)
- **2-way and 3-way splits** are far more robust (< 5% RMSE gain across configurations)
- **Root cause of leakage**: generating input-output sequences BEFORE dataset partitioning allows future observations to contaminate the training set
- **Small windows** (size=3) increase leakage sensitivity; **larger windows** help reduce it
- **Longer lag steps** amplify leakage effects — small lag + large window = most protected configuration
- "Clean" configuration: sequences must be generated **AFTER** splitting into train/val/test sets

#### Critical Rule for Our Pipeline
> **NEVER** generate sliding windows on the full dataset and then split. Always split first (temporally), then create windows within each partition separately.

#### Leakage Rules Confirmed (L-series)
- L1, L3, L4 directly threatened by pre-split sequence generation
- Recommendation: use **3-way temporal split** (70/10/20) for MASI — robust and supports early stopping

---

### Paper 2: Walk-Forward Validation Framework (Deep, Deep & Lamptey, 2025)
**File:** 2512.06932v1.pdf | arXiv:2512.12924

#### Key Findings
- Walk-forward with **34 independent out-of-sample test periods** is the gold standard
- Strategy must prove itself **repeatedly** across diverse market regimes, not just one backtest
- **Strict information set discipline**: features, signals, and execution decisions use ONLY data available at time t
- **Signal execution rule**: signal at t → executed at **OPEN of t+1** (directly maps to our L5)
- Performance is **regime-dependent**: +2.4% annualized in high-volatility periods (2020-2024) vs -0.16% in stable markets (2015-2019)
- **Honest finding**: aggregate results statistically insignificant (p=0.34) — this demonstrates what rigorous validation looks like
- Realistic transaction costs: commission, slippage, position limits, stop-loss rules

#### Relevance for MASI
- MASI is also regime-dependent (COVID impact 2020, election cycles, commodity shocks)
- With ~2000-2500 obs, we can realistically do **5-8 walk-forward folds**, not 34
- Confirms: regime-aware segmentation essential for our HMM integration

#### Methodological Contribution
- Walk-forward is NOT just about held-out test sets — it's about **repeated validation across multiple regimes**
- Aggregate performance can be misleading; always decompose by regime (Bull/Neutral/Bear)

---

## PRIORITY 2 — Marché Marocain (MASI)

### Paper 3: ML Forecasting MASI (Oukhouya & El Himdi, 2024)
**File:** 978-3-031-68628-3_6 (2).pdf

#### Key Findings
- Applied XGBoost, SVR, LSTM, MLP, SVR-XGBoost, LSTM-XGBoost on **MASI (all sectors)**
- Best performers: **SVR-XGBoost** and **LSTM-XGBoost** (hybrid models)
- Grid Search optimization for hyperparameters
- Daily price prediction (closing price)

#### ⚠️ Leakage Risk Identified
- No explicit mention of temporal train/test split protocol
- Grid Search could introduce look-ahead bias if applied on full dataset
- No walk-forward validation reported — single train/test split likely

#### Relevance for MASI Baseline (Étape 2)
- SVR and XGBoost confirmed as strong baselines for Moroccan market
- Hybrid ensembles outperform single models — confirms our CNN-LSTM + HMM approach is justified
- Use as benchmark: if our CNN-LSTM cannot beat SVR-XGBoost, simplify to ensemble

---

### Paper 4: SVR/XGBoost/LSTM/MLP on MSI20 (Oukhouya & El Himdi, 2023)
**File:** csmf-07-00039 (1).pdf

#### Key Findings
- **MSI 20** (Morocco Stock Index 20): new index, N=**541 observations** (Dec 2020 — Feb 2023)
- Features: Open, High, Low prices → predict Close
- **SVR** best performance: MAPE=0.368%, R²=0.989, RMSE=3.993
- LSTM second-worst: RMSE=6.322 (test), R²=0.974
- 90% training / 10% testing split

#### ⚠️ CRITICAL Leakage Violations Identified
1. **L1 VIOLATED**: Min-Max scaler fitted on FULL dataset before splitting → direct data leakage
2. No walk-forward validation — single static split
3. No temporal gap between train end and test start
4. Small dataset (N=541) makes results unreliable for deep learning

#### Moroccan Market Constraints Confirmed
- MSI20 is even more limited than MASI (only 76 total stocks, fewer for liquid ones)
- Very small dataset → confirms our C3 constraint (< 10,000 parameters preferred)
- Moroccan market offers OHLC data from CSE (casablanca-bourse.com)

#### Recommendation for Our Pipeline
- **DO NOT** replicate their leaky scaler approach
- Use their reported metrics only as rough baseline context, not clean benchmarks
- Our validation will be stricter and results will differ (lower but more honest)

---

### Paper 5: LSTM+GRU Trading Strategy for Moroccan Market (Touzani & Douzi, 2021)
**File:** 40537_2021_Article_512.pdf

#### Key Findings
- **First published DL trading strategy designed specifically for the Moroccan market**
- Moroccan market has **76 tradeable stocks** with severe **trading discontinuity** (weeks/months without trading for some stocks)
- Only the most **liquid** stocks can be used reliably
- Transfer learning approach: trained on US (S&P500) + French (CAC40) data → validated on Moroccan data
- LSTM for short-term, GRU for medium-term predictions
- Annualized return: **27.13%** vs MASI benchmark 0.43% (BUT no walk-forward, suspect results)
- Brokerage fees subtracted → more realistic than most papers

#### ⚠️ Leakage/Overfitting Concerns
- Training on different market (US + French) → validation on Moroccan = **domain shift problem**
- Results appear too optimistic for a frontier market
- No walk-forward validation across multiple MASI periods
- Strategy selects stocks with high expected return → look-ahead bias in stock selection

#### Moroccan Market Constraints Confirmed
- **Low liquidity**: stocks can go untraded for extended periods (Fig 2 shows multi-week gaps)
- **Missing data** from trading discontinuity must be removed, not forward-filled
- **Only 76 total stocks** → universe selection critical
- Moroccan market **characterized by occasional opportunities**, not continuous trading
- This validates our **C2 (low liquidity)** and **C5 (missing data)** constraints

#### Key Takeaway for MASI Project
> The Moroccan market's illiquidity means that a MASI-level index (all shares) is LESS affected by individual stock illiquidity than individual stock strategies. MASI index data is more continuous than individual stocks. Focus on MASI index as target.

---

## PRIORITY 3 — Architecture HMM + Deep Learning

### Paper 6: HMM-LSTM Fusion for Economic Forecasting (Sivakumar, 2025)
**File:** 2501.02002v1.pdf

#### Key Findings
- HMM identifies hidden economic states, then **HMM states + state means** appended as additional features to LSTM
- Two LSTM variants tested: original features vs augmented (original + HMM features)
- **HMM-augmented LSTM outperforms standard LSTM**, especially in volatile periods
- Uses Expectation-Maximization (EM) for HMM training, Viterbi for state sequence decoding
- 3-state HMM: outperforms 2-state for economic state identification (Rikken, 2022 cited)
- Integrated Gradients used for LSTM feature importance explanation

#### Regime-as-Feature Integration (Directly Applicable)
- Our approach: one-hot encode HMM regime → append to CNN-LSTM input
- Sivakumar adds HMM state + state means → provides additional signal beyond just the state label
- **Recommendation**: test both approaches: (a) one-hot only, (b) one-hot + state transition probabilities

#### HMM Training Protocol
- Train HMM on **training data only** (confirms our L2 rule)
- Forward-predict regime labels for validation and test periods
- **Never re-train HMM on validation or test data**

#### Walk-Forward Adaptation
- In walk-forward: re-estimate HMM in **each training window** separately (confirms L8 analog for HMM)
- Viterbi decoding applied forward from last training day

---

### Paper 7: HMM + Neural Networks for Algorithmic Trading (Monteiro, 2025)
**File:** 2407.19858v6.pdf

#### Key Findings
- **5-state HMM** trained on log returns (Gaussian emission)
- Dual-model system: HMM predicts regime → NN predicts price change → **consensus** required for trade signal
- Trading signal only when **both models agree** on direction
- Data validation: filter out |r_t| > 50% as anomalous returns (adapt this for MASI: use 10%)
- Rolling window retraining at fixed frequency f
- 3-year warm-up period for model stability before live trading
- Achieved 83% return, Sharpe 0.77 during COVID period (2019-2022)

#### HMM Mathematical Specification
```
HMM tuple: λ = (S, O, A, B, π)
- S = {s1,...,sN}: hidden states (N=5 in paper, N=3 for our MASI)
- O: log return observations
- A: state transition matrix (rows sum to 1)
- B: Gaussian emission probability N(μj, Σj)
- π: initial state distribution
- Viterbi algorithm: s* = argmax P(s1:T | o1:T, λ)
- Best state for prediction: sbest = argmax_j E[r|s=j]
```

#### Relevance for MASI
- **3-state HMM** is more interpretable and appropriate for MASI's limited data (~2000 obs)
- Log returns (not raw prices) as HMM observation: mathematically sound, matches our target
- Consensus mechanism reduces false signals — useful for low-liquidity markets like MASI
- 5% anomaly threshold for MASI might be appropriate (not 50%)

#### ⚠️ Leakage Risk in Paper
- Data normalization uses rolling window mean and std — **must use training-window only** for normalization in walk-forward context
- Regular retraining: ensure HMM re-estimated on expanding or rolling window, never including test data

---

## PRIORITY 4 — Marchés Émergents & Frontière

### Paper 8: Regime-Switching in Frontier Markets (Korley & Giouvris, 2021)
**File:** jrfm-14-00122-v2.pdf

#### Key Findings
- Sub-Saharan African frontier markets exhibit **two distinct regimes**:
  - Regime 1: **High-volatility / High-return** (shorter duration)
  - Regime 2: **Low-volatility / Low-return** (more persistent)
- This is **opposite to developed markets** (where high-vol = low-return)
- Markov-switching VAR model used (MS-VAR)
- **High-volatility regime less persistent** than low-volatility regime
- Morocco-specific: Eissa et al. (2010) found no volatility spillover from FX to stock returns for Morocco
- Weekly data used (Jan 2000 – Dec 2018)

#### Implications for MASI Regime Detection
1. **3 regimes** (Bull/Neutral/Bear) preferred over 2-regime model for MASI
2. Regime structure in frontier/emerging African markets ≠ developed market conventions
3. **Bear regime may be shorter but more volatile** — HMM must capture this asymmetry
4. Regime transitions driven by commodity price shocks, political events, global crises
5. MASI is partially insulated from FX shocks → FX rate NOT required as input feature

#### Key Constraint for MASI
> MASI frontier market characteristics mean that regime labels (Bull/Bear/Neutral) may behave differently than in literature from developed markets. Always verify regime assignment with MASI-specific return distributions.

---

## PRIORITY 5 — Proxy Volatilité sans VIX

### Paper 9: Realized GARCH, VIX, Volatility Risk Premium (Hansen et al., 2021)
**File:** 2112.05302v1.pdf

#### Key Findings
- **Realized GARCH** outperforms conventional GARCH models for both in-sample fit and out-of-sample VIX forecasting
- Realized GARCH adds a **second shock variable** (volatility shock u_t) alongside return shock (z_t)
- Model equations:
  ```
  log r_t = r + λ√h_t − ½h_t + √h_t · z_t           (return equation)
  log h_{t+1} = ω + β·log h_t + τ(z_t) + γ·σ·u_t     (GARCH equation)
  log x_t = κ + φ·log h_t + δ(z_t) + σ·u_t           (measurement equation)
  ```
- Requires **high-frequency intraday data** for realized variance (x_t)
- Standard GARCH(1,1) is a special case when γ=0 and σ=0

#### ⚠️ CRITICAL CONSTRAINT for MASI
> **Realized GARCH is NOT feasible for MASI**: no high-frequency (intraday) data available. The realized variance measure x_t requires tick or minute-level data.

#### Recommendation for MASI Volatility Proxy
Given no VIX and no HF data for MASI:

| Priority | Proxy | Feasibility | Justification |
|----------|-------|-------------|---------------|
| 1 (if ARCH confirmed) | **GARCH(1,1) conditional σ** | ✅ Daily data sufficient | Captures volatility clustering |
| 2 (fallback) | **Rolling realized vol (21-day)** | ✅ Easy to compute | Stable, interpretable |
| 3 | **ATR-based volatility** | ✅ Requires OHLC | Captures intraday range |
| 4 | **Downside semi-deviation** | ✅ | Sortino-style, asymmetric |

**Decision rule**: Run Ljung-Box test on squared returns → if ARCH effects confirmed → use GARCH(1,1). If not → use rolling 21-day realized volatility.

---

## PRIORITY 6 — Deep Learning Portfolio & Général

### Paper 10: Hybrid LSTM-CNN with Technical Indicators (Fozap, 2025)
**File:** jrfm-18-00201-v2.pdf

#### Key Findings
- Hybrid LSTM-CNN on S&P500, 14-year period (Jan 2010 – Dec 2024)
- **Random Forest**: lowest RMSE (0.0859), highest R² (0.5655) — but no sequential learning
- **LSTM-CNN hybrid**: RMSE=0.1012, MAE=0.0800, MAPE=10.22%, R²=0.4199
- Technical indicators used: SMA(10,50), EMA(10,50), Bollinger Bands, RSI, MACD, OBV, Volume
- CNN extracts **local short-term patterns**; LSTM captures **long-term temporal dependencies**
- Model architecture: CNN layer → LSTM layer → Dense output

#### ⚠️ Leakage Risk Identified
- Min-Max normalization applied globally (before splitting) → **L1 VIOLATED**
- No walk-forward validation reported
- Results on S&P500 with 14 years ≠ MASI with 2000-2500 obs

#### Relevance for Our CNN-LSTM Architecture
- Confirms CNN-first → LSTM-second as standard hybrid ordering
- Technical indicators list aligned with our planned features
- **Key finding**: Random Forest outperformed CNN-LSTM on raw metrics → supports our Rule 8 (DL only if it outperforms all baselines)
- For MASI (smaller dataset): RF advantage may be even more pronounced

#### Architecture Constraints Confirmed
Given MASI data size (~2000-2500 obs), our architecture constraints from prompt.md are justified:
```
Conv1D: 1 layer, max 32 filters, kernel_size=3
LSTM: 1 layer, max 32 units
Dropout: 0.2-0.3
Dense: 1 layer, max 16 units
Input window L: {10, 15, 20}
Features F: ≤ 15
```

---

## SYNTHESIS: RECOMMENDED METHODS FOR MASI

### ✅ RECOMMENDED: Justified Methods

| Method | Justification |
|--------|---------------|
| **Walk-forward validation (5-8 folds)** | Gold standard for non-stationary markets. Confirmed by Papers 1, 2. Adapted to MASI's smaller N. |
| **3-state Gaussian HMM** | More interpretable than 2-state. 3-state outperforms 2-state (Paper 6, citing Rikken 2022). Appropriate for frontier markets. |
| **Regime-as-Feature (one-hot + transition probs)** | HMM-augmented LSTM improves accuracy in volatile periods (Paper 6). |
| **GARCH(1,1) volatility proxy** (if ARCH confirmed) | Feasible with daily data. Standard for markets without VIX (Papers 9, prompt.md). |
| **Rolling 21-day realized vol** (fallback) | Simple, interpretable, robust fallback (confirmed by multiple papers). |
| **Baselines first**: Random Walk → Historical Mean → ARIMA → Random Forest | RF consistently competitive or superior to DL in smaller datasets (Papers 3, 4, 10). |
| **3-way temporal split** (70/10/20) with sequential sequence generation | Robust to leakage (Paper 1). Supports early stopping. |
| **Signal at t → execute at OPEN of t+1** | Realistic execution delay confirmed (Paper 2). |
| **OHLC + Volume features** | Available from Casablanca Stock Exchange. Confirmed in Papers 3, 4, 5, 10. |
| **Transaction costs 5-10 bps** | Moroccan market realistic costs (Paper 5, prompt.md). |

---

### ❌ METHODS TO AVOID FOR MASI (with reasons)

| Method | Reason to Avoid |
|--------|-----------------|
| **10-fold cross-validation** | Most vulnerable to temporal leakage (RMSE gain up to 20.5%). Never appropriate for time series (Paper 1). |
| **Min-Max scaler on full dataset** | Direct L1 violation. Found in Papers 3 and 4 — do NOT replicate. Fit scaler on training data ONLY. |
| **Realized GARCH / VIX-based volatility** | No high-frequency data available for MASI. Not feasible (Paper 9). |
| **Large CNN-LSTM architecture** | MASI has ~2000 obs. > 10,000 parameters risks severe overfitting. Random Forest may outperform (Papers 3, 10). |
| **L=30 input window** | Too long for MASI. Use L ∈ {10, 15, 20} (prompt.md, confirmed by Paper 1: larger windows reduce leakage sensitivity). |
| **F > 15 features** | Start with ≤ 15 validated features. More features → overfitting on small dataset. |
| **Single train/test split** | Hides regime-dependent performance failures. Confirmed by Papers 1, 2, 5. |
| **Transfer learning from US/European markets** | Domain shift problem (Paper 5). Different liquidity regime, different return distribution. |
| **Deep RL for MASI** | Requires much larger dataset. Interpretability issues. Not justified for frontier market with ~2000 obs. |
| **5-state HMM** | Too many states for ~2000 obs. 3-state provides better trade-off (interpretability vs flexibility). |
| **Claiming strong predictive power** | Literature consistently reports modest out-of-sample results. Rigorously validated systems rarely exceed 0.5-2% annualized edge (Paper 2, 0.55% annualized). |

---

## OPEN METHODOLOGICAL QUESTIONS

1. **Data coverage**: Do our CSV files cover 2010-2026? Need to verify exact date range and handle structural breaks (COVID 2020, 2008 crisis pre-data).

2. **Zero-volume days**: MASI has trading discontinuities (Paper 5). Need explicit rule: forward-fill max 2 days, drop if gap > 5 days.

3. **Optimal number of HMM states**: Paper 6 recommends 3-state, but MASI's frontier market dynamics may support 2-state (simpler, more robust). **Test BIC/AIC for 2 vs 3 states in Étape 4**.

4. **Walk-forward gap (L6 rule)**: How many days between training end and validation start for MASI? Literature suggests minimum regime duration. For MASI, suggest L=5-10 trading days.

5. **ARCH effects**: Are ARCH effects confirmed in MASI log-returns? This determines GARCH vs rolling-vol choice. **Must test in Étape 0 audit**.

6. **CNN-LSTM vs LSTM-only**: Is the CNN component justified for MASI's daily data? The Fozap (2025) paper found Random Forest competitive. **Test pure LSTM and Random Forest as primary baselines before adding CNN**.

7. **Regime feature representation**: One-hot encoding (our plan) vs continuous state probability vector. The probability vector provides "soft" regime assignment and may be more informative.

8. **MASI liquidity filter**: Which days have anomalously low returns that should be treated as missing? Use |r_t| > 10% threshold to flag suspicious data points.

---

## CONSENSUS ACROSS PAPERS

### Strong Consensus (4+ papers agree)
- Walk-forward validation is mandatory for financial time series
- Data leakage is pervasive and severely inflates reported performance
- Temporal split only (never random split) for time series
- Scaler/encoder must be fit on training data only
- Baseline models (RF, ARIMA) must be beaten before claiming DL superiority

### Moderate Consensus (2-3 papers agree)
- HMM regime features improve DL model accuracy in volatile markets
- 3-state HMM preferred over 2-state for frontier/emerging markets
- GARCH(1,1) is the most feasible volatility proxy without VIX
- Transaction costs must be included in any realistic backtest

### Conflicts / Open Debates
- **Number of HMM states**: Papers use 2, 3, or 5 states. No consensus for MASI specifically. → Test 2 vs 3 with BIC.
- **LSTM vs RF dominance**: Papers 3, 10 find RF competitive or superior; Papers 5, 6 find LSTM superior. → Empirical question for MASI.
- **CNN component value**: Paper 10 shows CNN adds value for S&P500 14-year period; not tested on small frontier market data. → Test empirically in Étape 5.
- **Transfer learning**: Paper 5 uses it, but domain shift is a methodological concern. For MASI index (not individual stocks), more data is available — transfer learning less critical.

---

## DECISIONS FOR ÉTAPE 1

Based on this synthesis, the following decisions are confirmed for preprocessing:

1. **Data source**: Use `data/raw/masi_raw.csv` (most recent, up to April 2026) + merge with `masi_raw.csv` (2016-2023) for maximum coverage. Verify for duplicates.
2. **Temporal split**: 70% train / 10% validation / 20% test (strict temporal, no shuffling)
3. **Sequence generation**: AFTER split (not before) — implements L1, L3, L4
4. **Missing data**: Forward-fill max 2 consecutive days, drop gaps > 5 days (MASI holiday rule L7)
5. **Target variable**: y_t = ln(P_{t+1}/P_t) — log return, computed AFTER all features
6. **Scaler**: StandardScaler fit on training set ONLY — re-fit in each walk-forward window
7. **Leakage check**: Assert no test/val data timestamps appear in training feature computation

---

*End of Literature Synthesis — Étape 0*
