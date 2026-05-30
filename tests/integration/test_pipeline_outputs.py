"""Integration tests for the consistency of pipeline outputs."""
import pandas as pd


# ---------------------------------------------------------------------------
# Alignment between étape 5 (predictions) and étape 4 (regimes)
# ---------------------------------------------------------------------------
def test_predictions_aligned_with_regimes(require_outputs):
    preds = pd.read_csv(require_outputs / "etape5" / "predictions_test.csv")
    regs = pd.read_csv(require_outputs / "etape4" / "regimes" / "masi_regimes_test.csv")
    assert len(preds) == len(regs), (
        f"predictions_test ({len(preds)} rows) != regimes_test ({len(regs)} rows)"
    )


def test_test_split_has_expected_size(require_outputs):
    """Canonical TEST size from feature_metadata = 948 rows (étape 1 split)."""
    preds = pd.read_csv(require_outputs / "etape5" / "predictions_test.csv")
    assert len(preds) == 948


# ---------------------------------------------------------------------------
# Canonical export — schema required by the dashboard
# ---------------------------------------------------------------------------
EXPECTED_CANONICAL_COLS = {
    "date", "actual_return", "predicted_return", "signal_raw",
    "regime", "regime_name", "position", "strategy_return", "equity",
    "strategy_name", "mode", "cost_bps",
}


def test_canonical_export_has_required_columns(require_outputs):
    """outputs/etape6/etape6_final_predictions.csv must expose the dashboard schema."""
    canonical = require_outputs / "etape6" / "etape6_final_predictions.csv"
    if not canonical.exists():
        import pytest
        pytest.skip("Run `python -m masi_hybrid_forecasting.pipeline export` first.")
    df = pd.read_csv(canonical)
    missing = EXPECTED_CANONICAL_COLS - set(df.columns)
    assert not missing, f"Canonical export missing columns: {missing}"


def test_canonical_export_equity_starts_near_one(require_outputs):
    """First equity value must be close to 1.0 ± first-day strategy return."""
    canonical = require_outputs / "etape6" / "etape6_final_predictions.csv"
    if not canonical.exists():
        import pytest
        pytest.skip("canonical CSV not generated yet")
    df = pd.read_csv(canonical)
    # cumprod-of-exp starts near 1.0
    assert 0.9 < df["equity"].iloc[0] < 1.1
