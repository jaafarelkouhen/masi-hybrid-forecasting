"""Unit tests for transaction-cost conventions."""

import numpy as np

from masi_hybrid_forecasting.pipeline.strategies import strategy_returns


def test_binary_strategy_costs_are_turnover_proportional():
    """A direct -1 -> +1 flip closes and reopens exposure, so it costs 2x."""
    positions = np.array([-1.0, 1.0, 0.0, 1.0])
    y_true = np.zeros_like(positions)
    cost = 0.001

    returns = strategy_returns(positions, y_true, mode="binary", cost_dec=cost)

    np.testing.assert_allclose(returns, np.array([-0.001, -0.002, -0.001, -0.001]))


def test_continuous_strategy_costs_match_binary_for_same_turnover():
    positions = np.array([-1.0, 1.0, 0.0, 0.5])
    y_true = np.zeros_like(positions)
    cost = 0.001

    binary_like = strategy_returns(positions, y_true, mode="binary", cost_dec=cost)
    continuous = strategy_returns(positions, y_true, mode="continuous", cost_dec=cost)

    np.testing.assert_allclose(binary_like, continuous)
