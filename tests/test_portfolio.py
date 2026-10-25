import numpy as np
import pandas as pd

from lab.portfolio import simulate


def test_one_winning_trade():
    dates = pd.bdate_range("2024-01-01", periods=5)
    op = np.array([[10.0, 10.0, 11.0, 11.0, 11.0]])
    cl = op.copy()
    eq = simulate(dates, op, cl, np.array([0.0005]), np.array([0]), np.array([1]), np.array([1]))
    # buy $250 at 10 on day 1, sell at 11 on day 2: +$25, minus 5bp on the $275 sale
    assert eq.iloc[0] == 1000
    assert abs(eq.iloc[-1] - (1025 - 275 * 0.0005)) < 1e-9


def test_slots_cap_at_four():
    dates = pd.bdate_range("2024-01-01", periods=4)
    op = np.full((6, 4), 10.0)
    sym = np.arange(6)
    eq = simulate(dates, op, op, np.zeros(6), sym, np.zeros(6, dtype=int), np.full(6, 2))
    assert (eq == 1000).all()     # flat prices, no costs: equity unchanged, nothing crashes


def test_slot_survives_a_loss():
    """Regression: a losing trade used to leave the slot under $250 and unable to buy again."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    op = np.array([[10.0, 10.0, 9.0, 9.0, 9.9, 9.9]])
    eq = simulate(dates, op, op, np.zeros(1), np.array([0, 0]), np.array([1, 3]), np.array([1, 1]))
    # lose 10% of $250, then the second trade still happens (+10% on the new quarter-size slot)
    first = 1000 - 25
    assert abs(eq.iloc[-1] - (first + 0.1 * first / 4)) < 1e-9
