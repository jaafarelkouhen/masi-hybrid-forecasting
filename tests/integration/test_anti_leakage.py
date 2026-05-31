"""Integration tests for the L1-L8 anti-leakage rules.

These tests read the canonical artifacts under outputs/etape{N}/ and assert
that the pipeline produced them without temporal leakage. They auto-skip when
outputs/ has not been populated.

Anti-leakage rules (from docs/anti_leakage.md):
  L1 — scaler / quantiles fit on TRAIN only
  L3 — rolling features strictly causal (shift(1).rolling(window))
  L6 — walk-forward gap between TRAIN/VAL/TEST splits
  L8 — GARCH parameters frozen from TRAIN
"""
import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# L1 — scaler stats and GARCH params fit on TRAIN only
# ---------------------------------------------------------------------------
def test_L1_scaler_stats_match_train_mean(require_outputs):
    """L1: scaler mean for log_return must equal TRAIN mean (not full / val / test)."""
    features_train = pd.read_csv(
        require_outputs / "etape3" / "features" / "masi_features_train.csv"
    )
    with open(
        require_outputs / "etape3" / "features" / "scaler_stats_etape3_train_only.json"
    ) as f:
        scaler_stats = json.load(f)

    train_mean = features_train["log_return"].mean()
    train_std = features_train["log_return"].std(ddof=0)

    assert scaler_stats["log_return"]["mean"] == pytest.approx(train_mean, rel=1e-6)
    assert scaler_stats["log_return"]["std"] == pytest.approx(train_std, rel=1e-3)


def test_L1_scaler_stats_dont_match_test_mean(require_outputs):
    """L1 negative control: scaler mean must NOT match TEST mean (would be leakage)."""
    features_test = pd.read_csv(
        require_outputs / "etape3" / "features" / "masi_features_test.csv"
    )
    with open(
        require_outputs / "etape3" / "features" / "scaler_stats_etape3_train_only.json"
    ) as f:
        scaler_stats = json.load(f)

    test_mean = features_test["log_return"].mean()
    train_scaler_mean = scaler_stats["log_return"]["mean"]
    # The two means are not identical — if they happened to be, that's a coincidence,
    # but for MASI 2007-2020 vs 2022-2026 they really differ.
    assert train_scaler_mean != pytest.approx(test_mean, rel=1e-3)


def test_L8_garch_params_fit_on_train_only(require_outputs):
    """L8: garch_params_train.json was fit on the TRAIN partition only."""
    with open(
        require_outputs / "etape3" / "features" / "garch_params_train.json"
    ) as f:
        garch = json.load(f)

    # n_train must match the TRAIN row count from feature_metadata
    with open(require_outputs / "etape3" / "features" / "feature_metadata.json") as f:
        meta = json.load(f)

    # garch fits on raw returns BEFORE warmup-row drop — accept ±100 rows tolerance
    train_n_features = meta["splits"]["train"]["n"]
    assert abs(garch["n_train"] - train_n_features) <= 100

    # Parameters must satisfy GARCH(1,1) stationarity: alpha + beta < 1
    assert 0.0 < garch["alpha"] < 1.0
    assert 0.0 < garch["beta"] < 1.0
    assert garch["alpha"] + garch["beta"] < 1.0
    assert garch["omega"] > 0.0


# ---------------------------------------------------------------------------
# L3 — rolling features strictly causal (warmup NaN expected)
# ---------------------------------------------------------------------------
def test_L3_only_one_contemporaneous_feature(require_outputs):
    """L3 / D2: exactly ONE feature is contemporaneous (log_return). All others lagged."""
    with open(require_outputs / "etape3" / "features" / "feature_metadata.json") as f:
        meta = json.load(f)
    assert meta["contemporaneous_features"] == ["log_return"]


def test_L3_lag_features_truly_shifted_in_train(require_outputs):
    """L3: ret_lag1[t] must equal log_return[t-1] (after warmup-row drop)."""
    df = pd.read_csv(
        require_outputs / "etape3" / "features" / "masi_features_train.csv"
    )
    diff = (df["ret_lag1"].iloc[1:].values - df["log_return"].iloc[:-1].values)
    # After warmup-drop, rows are contiguous, so the equality should hold exactly
    # (or within fp tolerance from scaling roundtrips)
    assert abs(diff).max() < 1e-6, (
        f"ret_lag1 doesn't equal log_return.shift(1) — max abs diff = {abs(diff).max()}"
    )


# ---------------------------------------------------------------------------
# L6 — walk-forward gap between splits
# ---------------------------------------------------------------------------
def test_L6_walk_forward_positive_gap(require_outputs):
    """L6: TRAIN end < VAL start, VAL end < TEST start, with positive gap."""
    with open(require_outputs / "etape3" / "features" / "feature_metadata.json") as f:
        meta = json.load(f)

    train_end = pd.Timestamp(meta["splits"]["train"]["end"])
    val_start = pd.Timestamp(meta["splits"]["val"]["start"])
    val_end = pd.Timestamp(meta["splits"]["val"]["end"])
    test_start = pd.Timestamp(meta["splits"]["test"]["start"])

    gap_train_val = (val_start - train_end).days
    gap_val_test = (test_start - val_end).days

    assert gap_train_val > 0, f"TRAIN→VAL gap is {gap_train_val} days (expected > 0)"
    assert gap_val_test > 0, f"VAL→TEST gap is {gap_val_test} days (expected > 0)"


def test_L6_no_date_overlap_between_splits(require_outputs):
    """L6: no date appears in two splits at once."""
    base = require_outputs / "etape3" / "features"
    dates_tr = set(pd.read_csv(base / "masi_features_train.csv")["date"])
    dates_va = set(pd.read_csv(base / "masi_features_val.csv")["date"])
    dates_te = set(pd.read_csv(base / "masi_features_test.csv")["date"])

    assert dates_tr & dates_va == set(), "TRAIN ∩ VAL is non-empty"
    assert dates_va & dates_te == set(), "VAL ∩ TEST is non-empty"
    assert dates_tr & dates_te == set(), "TRAIN ∩ TEST is non-empty"


# ---------------------------------------------------------------------------
# Compute-then-split (D1) — recorded in feature_metadata.leakage_test
# ---------------------------------------------------------------------------
def test_D1_compute_then_split_no_numerical_leakage(require_outputs):
    """D1: leakage_test entries logged at étape 3 must all show max_abs_diff == 0.0."""
    with open(require_outputs / "etape3" / "features" / "feature_metadata.json") as f:
        meta = json.load(f)

    leakage_entries = meta.get("leakage_test", [])
    assert len(leakage_entries) > 0, "No leakage_test entry found in feature_metadata"

    for entry in leakage_entries:
        assert entry["passed"], f"Leakage test at cut {entry['cut_date']} did not pass"
        assert entry["max_abs_diff"] == pytest.approx(0.0, abs=1e-12), (
            f"Cut {entry['cut_date']}: max_abs_diff = {entry['max_abs_diff']} != 0"
        )
