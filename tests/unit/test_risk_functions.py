"""Pure unit tests for the risk math (no I/O, no data files)."""
import numpy as np
import pandas as pd
import pytest

from masi_hybrid_forecasting.pipeline.risk import (
    christoffersen_indep,
    kupiec_pof,
    parametric_var_es,
    rolling_var_es_hist,
)


# ---------------------------------------------------------------------------
# parametric_var_es
# ---------------------------------------------------------------------------
def test_parametric_var_es_matches_normal_formula():
    """At alpha=5%, normal VaR = mu + sigma*z_0.05 ; ES = mu - sigma*phi(z)/alpha."""
    mu = pd.Series([0.0, 0.001])
    sigma = pd.Series([0.01, 0.02])
    var, es = parametric_var_es(mu, sigma, alpha=0.05)

    # z_0.05 ≈ -1.6449
    expected_var_0 = 0.0 + 0.01 * (-1.6449)
    expected_var_1 = 0.001 + 0.02 * (-1.6449)
    assert var.iloc[0] == pytest.approx(expected_var_0, abs=1e-3)
    assert var.iloc[1] == pytest.approx(expected_var_1, abs=1e-3)

    # ES is always more negative (or equal) than VaR for a normal left tail
    assert (es <= var).all()


def test_parametric_var_es_zero_sigma_gives_mu():
    """sigma=0 → VaR = mu (degenerate point mass)."""
    var, es = parametric_var_es(pd.Series([0.5]), pd.Series([0.0]), alpha=0.05)
    assert var.iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rolling_var_es_hist — must be CAUSAL (L3)
# ---------------------------------------------------------------------------
def test_rolling_var_es_hist_is_causal():
    """L3: the rolling window at row t must use rows [t-win, t-1] only.

    Replace the last row with garbage; the rolling stats for all earlier rows
    must be untouched. If they shift, the window is peeking into the future.
    """
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 0.01, size=300))
    var_a, es_a = rolling_var_es_hist(s, win=50, alpha=0.05)

    s_perturbed = s.copy()
    s_perturbed.iloc[-1] = 999.0   # ridiculous value in the future
    var_b, es_b = rolling_var_es_hist(s_perturbed, win=50, alpha=0.05)

    # All rows except the last MUST be identical (causal window)
    pd.testing.assert_series_equal(var_a.iloc[:-1], var_b.iloc[:-1])
    pd.testing.assert_series_equal(es_a.iloc[:-1], es_b.iloc[:-1])


def test_rolling_var_es_hist_warmup_is_nan():
    """First `win` rows must be NaN (insufficient history)."""
    s = pd.Series(np.linspace(-1, 1, 100))
    var, es = rolling_var_es_hist(s, win=20, alpha=0.05)
    # shift(1) + rolling(20, min_periods=20) ⇒ first 20 rows NaN
    assert var.iloc[:20].isna().all()
    assert es.iloc[:20].isna().all()
    assert var.iloc[20:].notna().all()


# ---------------------------------------------------------------------------
# kupiec_pof
# ---------------------------------------------------------------------------
def test_kupiec_pof_accepts_when_breach_rate_matches_alpha():
    """Exactly alpha% breaches → high p-value, verdict OK."""
    T = 1000
    breaches = np.zeros(T, dtype=bool)
    breaches[:50] = True   # 5% breaches, matches alpha=0.05
    rng = np.random.default_rng(0)
    rng.shuffle(breaches)
    res = kupiec_pof(breaches, alpha=0.05)
    assert res["verdict"] == "OK"
    assert res["pvalue"] > 0.5


def test_kupiec_pof_rejects_when_breach_rate_too_high():
    """20% breaches with alpha=5% → verdict REJETÉ."""
    T = 1000
    breaches = np.zeros(T, dtype=bool)
    breaches[:200] = True
    res = kupiec_pof(breaches, alpha=0.05)
    assert res["verdict"] == "REJETÉ"
    assert res["pvalue"] < 0.001


def test_kupiec_pof_degenerate_when_no_breach():
    res = kupiec_pof(np.zeros(100, dtype=bool), alpha=0.05)
    assert res["verdict"] == "DEGENERATE"


# ---------------------------------------------------------------------------
# christoffersen_indep
# ---------------------------------------------------------------------------
def test_christoffersen_indep_accepts_independent_breaches():
    """Bernoulli(p) breaches → independence holds, verdict OK."""
    rng = np.random.default_rng(123)
    breaches = (rng.random(2000) < 0.05).astype(int)
    res = christoffersen_indep(breaches)
    assert res["verdict"] == "OK"


def test_christoffersen_indep_rejects_clustered_breaches():
    """Breaches in a block (clustered) → independence rejected."""
    T = 1000
    breaches = np.zeros(T, dtype=int)
    breaches[100:200] = 1   # 100 consecutive breaches = strong cluster
    res = christoffersen_indep(breaches)
    assert res["verdict"] == "REJETÉ"
