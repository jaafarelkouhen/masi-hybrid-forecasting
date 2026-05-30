"""
================================================================================
ÉTAPE 1 (v2) — Multi-Factor Data Preprocessing
MASI Hybrid Forecasting System (HMM + CNN-LSTM)
================================================================================

INPUT: data/interim/masi_merged.csv (created by scripts/00_data_audit.py)
  - 4,786 obs, 14 raw columns, 2007-01-31 → 2026-04-20
  - Multi-factor: 4 stocks (ATW, IAM, LHM, MNG) + macro (Brent, Gold, EUR/MAD,
    GPR, BAM rate) + MASI OHLC

OUTPUT: 4 CSVs in outputs/etape1/splits/ + scaler stats + 3 plots
  - masi_clean_full.csv
  - masi_train.csv  (~3,350 obs, 2007-2020)
  - masi_val.csv    (~478 obs, 2020-2022)
  - masi_test.csv   (~948 obs, 2022-2026)
  - scaler_stats_train_only.json
  - reports/figures/etape1/*.png

Pipeline scope (Étape 1 — raw inputs only, no derivative features):
    1. Load merged multi-factor dataset
    2. Clean: drop duplicates, handle missing macro tail (ffill ≤ 30 days),
       handle MASI gaps (ffill ≤ 2 days per L7)
    3. Compute target y_t = ln(P_{t+1}/P_t) on masi_close AFTER cleaning (L4)
    4. Temporal split 70/10/20 with 5-day gap (L6)
    5. Fit StandardScaler on TRAIN partition only (L1)

Feature engineering (lags, rolling stats, GARCH, technicals, HMM states)
is DEFERRED to Étape 3 — strict shift(1) enforcement.

Anti-leakage rules enforced in this script:
    L1 — StandardScaler fit ONLY on TRAIN
    L4 — Target y_t computed AFTER all preprocessing
    L6 — Strict temporal split with 5-day buffer
    L7 — Forward-fill max 2 consecutive days for MASI

Author: Quantitative Research Lab
Date:   2026-05-19
================================================================================
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "etape1")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "etape1")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

MERGED_FILE = os.path.join(DATA_DIR, "masi_merged.csv")

# Columns expected in masi_merged.csv (14 total)
EXPECTED_COLUMNS = [
    "date",
    # MASI core
    "masi_close", "masi_open", "masi_high", "masi_low",
    # 4 most liquid Moroccan stocks
    "atw_close", "iam_close", "lhm_close", "mng_close",
    # Macro factors
    "brent_close", "gold_close",   # commodities
    "eur_mad",                     # FX
    "gpr_index",                   # geopolitical risk
    "bam_policy_rate",             # monetary policy
]

# Split configuration (per Étape 0 v2 decisions)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
GAP_DAYS = 5

# Cleaning thresholds
FFILL_LIMIT_MASI = 2          # L7: MASI gaps ≤ 2 days
FFILL_LIMIT_MACRO = 30        # Macro/FX series can have weekend/holiday gaps
SEGMENT_GAP_DAYS = 5          # L7: gap > this on MASI = structural break
EXTREME_RETURN_THRESHOLD = 0.50  # data-integrity guard

# Columns to scale (input features). Target excluded (it's the prediction).
COLUMNS_TO_SCALE = [
    "masi_close",
    "atw_close", "iam_close", "lhm_close", "mng_close",
    "brent_close", "gold_close", "eur_mad", "gpr_index", "bam_policy_rate",
    "log_return",
]


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_merged(filepath: str = MERGED_FILE) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Merged dataset not found: {filepath}\n"
            f"Run scripts/00_data_audit.py first to generate it."
        )
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    print(f"  Loaded {len(df)} rows × {len(df.columns)} cols from {os.path.basename(filepath)}")
    print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
    return df


# =============================================================================
# 2. CLEANING — L7 COMPLIANT
# =============================================================================

@dataclass
class CleaningReport:
    n_input: int = 0
    n_duplicate_dates: int = 0
    n_neg_masi_prices: int = 0
    n_extreme_corrupt: int = 0
    n_masi_ffilled: int = 0
    n_macro_ffilled: int = 0
    structural_breaks: list = None
    n_output: int = 0

    def __post_init__(self):
        if self.structural_breaks is None:
            self.structural_breaks = []


def clean_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, CleaningReport]:
    """Apply cleaning rules. Returns clean df + report."""
    report = CleaningReport(n_input=len(df))
    df = df.copy()

    # --- 1. Duplicates
    dup_mask = df.index.duplicated(keep="last")
    report.n_duplicate_dates = int(dup_mask.sum())
    df = df[~df.index.duplicated(keep="last")]

    # --- 2. Non-positive MASI prices
    neg_mask = (df["masi_close"] <= 0)
    report.n_neg_masi_prices = int(neg_mask.sum())
    if neg_mask.any():
        df = df.loc[~neg_mask]

    # --- 3. Extreme moves on MASI (data corruption guard)
    pct_change = df["masi_close"].pct_change().abs()
    corrupt_mask = pct_change > EXTREME_RETURN_THRESHOLD
    report.n_extreme_corrupt = int(corrupt_mask.sum())
    if corrupt_mask.any():
        if verbose:
            print(f"  [WARN] {report.n_extreme_corrupt} extreme MASI moves > 50% removed")
        df = df.loc[~corrupt_mask]

    # --- 4. Forward-fill MASI gaps (L7: max 2 days)
    masi_cols = ["masi_close", "masi_open", "masi_high", "masi_low"]
    before_masi = df[masi_cols].isna().sum().sum()
    df[masi_cols] = df[masi_cols].ffill(limit=FFILL_LIMIT_MASI)
    after_masi = df[masi_cols].isna().sum().sum()
    report.n_masi_ffilled = int(before_masi - after_masi)

    # --- 5. Forward-fill macro factors (more generous — they have weekend gaps)
    macro_cols = ["atw_close", "iam_close", "lhm_close", "mng_close",
                  "brent_close", "gold_close", "eur_mad", "gpr_index", "bam_policy_rate"]
    macro_cols = [c for c in macro_cols if c in df.columns]
    before_macro = df[macro_cols].isna().sum().sum()
    df[macro_cols] = df[macro_cols].ffill(limit=FFILL_LIMIT_MACRO)
    after_macro = df[macro_cols].isna().sum().sum()
    report.n_macro_ffilled = int(before_macro - after_macro)

    # --- 6. Drop rows where masi_close is still NaN (gap > 2 days = segment break)
    n_before_drop = len(df)
    df = df.dropna(subset=["masi_close"])
    if (n_before_drop - len(df)) > 0 and verbose:
        print(f"  Dropped {n_before_drop - len(df)} rows with masi_close still NaN after ffill")

    # --- 7. Drop early rows where multi-factor data is sparse
    # (Some macro series start later than 2007 → those rows are not usable for DL)
    initial_nans = df[macro_cols].isna().sum(axis=1)
    # Keep rows where at least 80% of macro features are present
    threshold = int(0.8 * len(macro_cols))
    df = df[initial_nans <= (len(macro_cols) - threshold)]

    report.n_output = len(df)

    if verbose:
        print(f"\n  Cleaning summary:")
        print(f"    Input rows:                  {report.n_input}")
        print(f"    Duplicates removed:          {report.n_duplicate_dates}")
        print(f"    Non-positive MASI prices:    {report.n_neg_masi_prices}")
        print(f"    Corrupt extreme moves:       {report.n_extreme_corrupt}")
        print(f"    MASI forward-filled cells:   {report.n_masi_ffilled}")
        print(f"    Macro forward-filled cells:  {report.n_macro_ffilled}")
        print(f"    Output rows (clean):         {report.n_output}")

    return df, report


# =============================================================================
# 3. TARGET CONSTRUCTION — L4 COMPLIANT
# =============================================================================

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute target y_t = ln(P_{t+1}/P_t) on masi_close.
    Also add contemporaneous log_return = ln(P_t/P_{t-1}) for diagnostics.

    L4: target is computed AFTER cleaning is complete.
    """
    df = df.copy()
    df["log_return"]    = np.log(df["masi_close"] / df["masi_close"].shift(1))
    df["target_y_next"] = np.log(df["masi_close"].shift(-1) / df["masi_close"])
    # Drop first row (no lag) and last row (no future target)
    df = df.dropna(subset=["log_return", "target_y_next"])
    return df


# =============================================================================
# 4. TEMPORAL SPLIT — L6 COMPLIANT
# =============================================================================

@dataclass
class SplitInfo:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    n_train: int
    n_val: int
    n_test: int
    gap_train_val_days: int
    gap_val_test_days: int


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    gap_days: int = GAP_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    df = df.sort_index()
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_start_idx = n_train + gap_days
    val_df = df.iloc[val_start_idx: val_start_idx + n_val]
    test_start_idx = val_start_idx + n_val + gap_days
    test_df = df.iloc[test_start_idx:]

    # L6 assertions
    assert train_df.index.max() < val_df.index.min(), "L6 VIOLATION: train >= val"
    assert val_df.index.max() < test_df.index.min(), "L6 VIOLATION: val >= test"
    assert (val_df.index.min() - train_df.index.max()).days >= gap_days
    assert (test_df.index.min() - val_df.index.max()).days >= gap_days

    info = SplitInfo(
        train_start=str(train_df.index.min().date()),
        train_end=str(train_df.index.max().date()),
        val_start=str(val_df.index.min().date()),
        val_end=str(val_df.index.max().date()),
        test_start=str(test_df.index.min().date()),
        test_end=str(test_df.index.max().date()),
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        gap_train_val_days=(val_df.index.min() - train_df.index.max()).days,
        gap_val_test_days=(test_df.index.min() - val_df.index.max()).days,
    )
    return train_df.copy(), val_df.copy(), test_df.copy(), info


# =============================================================================
# 5. SCALER — L1 COMPLIANT
# =============================================================================

@dataclass
class ScalerStats:
    feature: str
    mean: float
    std: float
    n_train: int
    source: str = "TRAIN ONLY"


def fit_scaler_on_train(
    train_df: pd.DataFrame, columns: list = None
) -> Tuple[StandardScaler, dict]:
    if columns is None:
        columns = [c for c in COLUMNS_TO_SCALE if c in train_df.columns]
    assert len(train_df) > 0
    scaler = StandardScaler()
    scaler.fit(train_df[columns].values)
    stats = {
        col: ScalerStats(
            feature=col, mean=float(scaler.mean_[i]),
            std=float(scaler.scale_[i]), n_train=len(train_df),
        )
        for i, col in enumerate(columns)
    }
    return scaler, stats


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, columns: list,
                 suffix: str = "_scaled") -> pd.DataFrame:
    df = df.copy()
    scaled = scaler.transform(df[columns].values)
    for i, col in enumerate(columns):
        df[col + suffix] = scaled[:, i]
    return df


# =============================================================================
# 6. PERSISTENCE
# =============================================================================

def save_splits(df_full, train_df, val_df, test_df, output_dir: str = SPLITS_DIR) -> dict:
    paths = {
        "full":  os.path.join(output_dir, "masi_clean_full.csv"),
        "train": os.path.join(output_dir, "masi_train.csv"),
        "val":   os.path.join(output_dir, "masi_val.csv"),
        "test":  os.path.join(output_dir, "masi_test.csv"),
    }
    df_full.to_csv(paths["full"])
    train_df.to_csv(paths["train"])
    val_df.to_csv(paths["val"])
    test_df.to_csv(paths["test"])
    return paths


def save_scaler_stats(stats: dict, output_dir: str = SPLITS_DIR) -> str:
    path = os.path.join(output_dir, "scaler_stats_train_only.json")
    with open(path, "w") as f:
        json.dump({col: asdict(s) for col, s in stats.items()}, f, indent=2)
    return path


# =============================================================================
# 7. PLOTS
# =============================================================================

def plot_split_overview(train_df, val_df, test_df, save_path: str = None) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(train_df.index, train_df["masi_close"], label=f"TRAIN ({len(train_df)})", color="#1f77b4")
    axes[0].plot(val_df.index, val_df["masi_close"], label=f"VAL ({len(val_df)})", color="#ff7f0e")
    axes[0].plot(test_df.index, test_df["masi_close"], label=f"TEST ({len(test_df)})", color="#2ca02c")
    axes[0].set_title("MASI Close — Temporal Split (70/10/20 + 5d gap)  [2007-2026]")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3); axes[0].set_ylabel("MASI")

    axes[1].plot(train_df.index, train_df["log_return"], color="#1f77b4", linewidth=0.5)
    axes[1].plot(val_df.index, val_df["log_return"], color="#ff7f0e", linewidth=0.6)
    axes[1].plot(test_df.index, test_df["log_return"], color="#2ca02c", linewidth=0.6)
    axes[1].axhline(0, color="black", linewidth=0.4)
    axes[1].set_title("MASI Daily Log-Returns by Partition")
    axes[1].set_ylabel("log-return"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_path = save_path or os.path.join(PLOTS_DIR, "etape1_split_overview.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


def plot_factor_overview(df: pd.DataFrame, save_path: str = None) -> str:
    """Visualize all 11 raw multi-factor series."""
    factors = ["masi_close", "atw_close", "iam_close", "lhm_close", "mng_close",
               "brent_close", "gold_close", "eur_mad", "gpr_index", "bam_policy_rate"]
    factors = [c for c in factors if c in df.columns]
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, factors):
        ax.plot(df.index, df[col], linewidth=0.8)
        ax.set_title(col); ax.grid(alpha=0.3)
    # Hide unused axes
    for ax in axes[len(factors):]:
        ax.set_visible(False)
    plt.suptitle("Multi-Factor Input Series (raw, 2007-2026)", fontsize=14, y=1.00)
    plt.tight_layout()
    save_path = save_path or os.path.join(PLOTS_DIR, "etape1_factor_overview.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


def plot_target_distribution(train_df, val_df, test_df, save_path: str = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.08, 0.08, 100)
    ax.hist(train_df["target_y_next"], bins=bins, alpha=0.6, label=f"TRAIN ({len(train_df)})", density=True, color="#1f77b4")
    ax.hist(val_df["target_y_next"],   bins=bins, alpha=0.6, label=f"VAL ({len(val_df)})", density=True, color="#ff7f0e")
    ax.hist(test_df["target_y_next"],  bins=bins, alpha=0.6, label=f"TEST ({len(test_df)})", density=True, color="#2ca02c")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Target y_t = ln(P_{t+1}/P_t) — Distribution by Partition")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("Next-day log-return"); ax.set_ylabel("Density")
    plt.tight_layout()
    save_path = save_path or os.path.join(PLOTS_DIR, "etape1_target_distribution.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


def plot_scaler_diagnostic(train_df, val_df, test_df, scaler, columns: list,
                            save_path: str = None) -> str:
    """Verify scaler: TRAIN should be µ≈0,σ≈1; val/test may drift."""
    train_s = apply_scaler(train_df, scaler, columns)
    val_s = apply_scaler(val_df, scaler, columns)
    test_s = apply_scaler(test_df, scaler, columns)

    cols_to_plot = ["masi_close", "log_return", "brent_close", "eur_mad"]
    cols_to_plot = [c for c in cols_to_plot if c in columns]
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
    if len(cols_to_plot) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols_to_plot):
        sc = col + "_scaled"
        ax.hist(train_s[sc], bins=60, alpha=0.6, label=f"TRAIN µ={train_s[sc].mean():.2f}σ={train_s[sc].std():.2f}", density=True, color="#1f77b4")
        ax.hist(val_s[sc],   bins=60, alpha=0.6, label=f"VAL   µ={val_s[sc].mean():.2f}σ={val_s[sc].std():.2f}", density=True, color="#ff7f0e")
        ax.hist(test_s[sc],  bins=60, alpha=0.6, label=f"TEST  µ={test_s[sc].mean():.2f}σ={test_s[sc].std():.2f}", density=True, color="#2ca02c")
        ax.set_title(col); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle("StandardScaler diagnostic (fit on TRAIN only — L1)", y=1.02)
    plt.tight_layout()
    save_path = save_path or os.path.join(PLOTS_DIR, "etape1_scaler_diagnostic.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return save_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(verbose: bool = True) -> dict:
    print("=" * 72)
    print(" ÉTAPE 1 (v2) — Multi-Factor Preprocessing on masi_merged.csv")
    print("=" * 72)

    # --- 1. Load
    print("\n[1/6] Loading merged dataset...")
    df_raw = load_merged()

    # --- 2. Clean (L7)
    print("\n[2/6] Cleaning (L7: ffill MASI≤2d, macro≤30d)...")
    df_clean, clean_report = clean_data(df_raw, verbose=verbose)

    # --- 3. Target (L4)
    print("\n[3/6] Computing target y_t = ln(P_{t+1}/P_t) AFTER cleaning (L4)...")
    df_target = compute_target(df_clean)
    print(f"  Final dataset: {df_target.shape}   {df_target.index.min().date()} -> {df_target.index.max().date()}")

    # --- 4. Split (L6)
    print("\n[4/6] Temporal split (70/10/20 + 5d gap, L6)...")
    train_df, val_df, test_df, split_info = temporal_split(df_target)
    print(f"  TRAIN: {split_info.train_start} -> {split_info.train_end}  (n={split_info.n_train})")
    print(f"  VAL:   {split_info.val_start} -> {split_info.val_end}  (n={split_info.n_val})")
    print(f"  TEST:  {split_info.test_start} -> {split_info.test_end}  (n={split_info.n_test})")
    print(f"  Gap train->val: {split_info.gap_train_val_days} cal days")
    print(f"  Gap val->test:  {split_info.gap_val_test_days} cal days")
    print(f"  L6 assertions: PASSED")

    # --- 5. Scaler (L1)
    print("\n[5/6] Fitting StandardScaler on TRAIN ONLY (L1)...")
    cols = [c for c in COLUMNS_TO_SCALE if c in train_df.columns]
    scaler, scaler_stats = fit_scaler_on_train(train_df, cols)
    for col, st in scaler_stats.items():
        print(f"  {col:18s}  mean={st.mean:12.4f}  std={st.std:12.4f}")

    # --- 6. Save artifacts
    print("\n[6/6] Saving artifacts...")
    paths = save_splits(df_target, train_df, val_df, test_df)
    scaler_path = save_scaler_stats(scaler_stats)
    for k, v in paths.items():
        print(f"  CSV [{k}]: {os.path.relpath(v, PROJECT_ROOT)}")
    print(f"  Scaler:  {os.path.relpath(scaler_path, PROJECT_ROOT)}")

    plot_paths = {
        "split_overview":      plot_split_overview(train_df, val_df, test_df),
        "factor_overview":     plot_factor_overview(df_target),
        "target_distribution": plot_target_distribution(train_df, val_df, test_df),
        "scaler_diagnostic":   plot_scaler_diagnostic(train_df, val_df, test_df, scaler, cols),
    }
    for k, v in plot_paths.items():
        print(f"  Plot [{k}]: {os.path.relpath(v, PROJECT_ROOT)}")

    print("\n" + "=" * 72)
    print(" ÉTAPE 1 (v2) COMPLETE")
    print(f"   Clean obs:   {len(df_target)}")
    print(f"   Splits:      TRAIN={len(train_df)}  VAL={len(val_df)}  TEST={len(test_df)}")
    print(f"   Features:    {len(cols)} raw scaled (target + 10 inputs)")
    print(f"   Anti-leak:   L1✓ L4✓ L6✓ L7✓")
    print("=" * 72)

    return {
        "df_raw": df_raw, "df_clean": df_clean, "df_target": df_target,
        "train": train_df, "val": val_df, "test": test_df,
        "split_info": split_info, "clean_report": clean_report,
        "scaler": scaler, "scaler_stats": scaler_stats,
        "csv_paths": paths, "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    artifacts = run_pipeline(verbose=True)
