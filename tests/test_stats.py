import numpy as np
import pandas as pd

from lab.stats import Market, Trades, block_consistency, excess, shift_p


def market(n_sym=3, n=3000, seed=0, plant=None):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.01, n)
    r = common + rng.normal(0, 0.01, (n_sym, n))
    if plant is not None:                 # add +0.5% on chosen days for every symbol
        r[:, plant] += 0.005
    op = 100 * np.exp(np.cumsum(r, axis=1))
    dates = pd.bdate_range("2000-01-03", periods=n)
    return Market(dates, [f"S{i}" for i in range(n_sym)], op, np.zeros(n_sym), disc_end=n - 1)


def day_trades(mkt, days):
    sym = np.repeat(np.arange(len(mkt.symbols)), len(days))
    pos = np.tile(days, len(mkt.symbols))
    return Trades(sym, pos, np.ones(len(pos), dtype=int))


def test_excess_is_return_minus_baseline():
    mkt = market()
    tr = day_trades(mkt, np.array([10, 20]))
    net, exc = excess(mkt, tr)
    assert np.allclose(exc, net - mkt.mu(1)[tr.sym])


def test_planted_edge_is_found():
    days = np.arange(100, 2900, 37)
    # the planted jump lands between open[d] and open[d+1], i.e. log-return index d+1
    mkt = market(plant=days + 1)
    p = shift_p(mkt, day_trades(mkt, days), 0, mkt.disc_end, 500, 30, seed=1)
    assert p < 0.01


def test_random_days_are_not_significant():
    mkt = market(seed=3)
    days = np.random.default_rng(7).choice(np.arange(50, 2900), 80, replace=False)
    p = shift_p(mkt, day_trades(mkt, np.sort(days)), 0, mkt.disc_end, 500, 30, seed=2)
    assert p > 0.01


def test_block_consistency_all_positive():
    days = np.arange(100, 2900, 37)
    mkt = market(plant=days + 1)
    tr = day_trades(mkt, days)
    _, exc = excess(mkt, tr)
    assert block_consistency(mkt, tr, exc) == 1.0
