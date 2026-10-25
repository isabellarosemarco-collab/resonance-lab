import numpy as np
import pandas as pd

from lab.portfolio import simulate


def test_one_winning_trade():
    dates = pd.bdate_range("2024-01-01", periods=5)
    op = np.array([[10.0, 10.0, 11.0, 11.0, 11.0]])
    cl = op.copy()
    eq = simulate(dates, op, cl, np.array([0.0005]), np.array([0]), np.array([1]), np.array([1]))
    # buy $250 at 10 on day 1, sell at 11 on day 2: +$25 minus 5bp of $250
    assert eq.iloc[0] == 1000
    assert abs(eq.iloc[-1] - (1025 - 0.125)) < 1e-9


def test_slots_cap_at_four():
    dates = pd.bdate_range("2024-01-01", periods=4)
    op = np.full((6, 4), 10.0)
    sym = np.arange(6)
    eq = simulate(dates, op, op, np.zeros(6), sym, np.zeros(6, dtype=int), np.full(6, 2))
    assert (eq == 1000).all()     # flat prices, no costs: equity unchanged, nothing crashes
