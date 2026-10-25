import numpy as np
import pandas as pd
import pytest

from lab.hypotheses import (HYPOTHESES, Trade, drop_overlaps, n2_22_days, n4_222_prices,
                            t3_369_levels, t4_369_streaks, t7_oscillation)


def frame(close, raw=None, start="2020-01-01"):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1e6, "raw_close": close if raw is None else raw}, index=idx)


def random_walk(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))


def test_drop_overlaps():
    trades = [Trade(0, 3), Trade(1, 3), Trade(3, 1), Trade(4, 2)]
    assert drop_overlaps(trades) == [Trade(0, 3), Trade(3, 1), Trade(4, 2)]


def test_t4_streak_of_three_enters_next_bar():
    sig = t4_369_streaks(frame([10, 9, 8, 7, 8, 9]))     # 3 down closes ending at bar 3
    assert Trade(4, 3) in sig.trades
    assert Trade(3, 3) in sig.twin                       # streak of 2 at bar 2 -> twin


def test_t3_crossing_a_369_level():
    # 31.40 -> 29.80 crosses 31 and 30 (two levels) -> ignored; 30.40 -> 29.90 crosses 30 (root 3)
    df = frame([31.4, 29.8, 30.4, 29.9, 29.95])
    sig = t3_369_levels(df)
    assert sig.trades == [Trade(4, 3)]


def test_n2_enters_on_the_22nd():
    df = frame(np.full(40, 100.0), start="2026-01-01")
    entries = [df.index[t.entry] for t in n2_22_days(df).trades]
    assert entries == [pd.Timestamp("2026-01-22")]


def test_n4_uses_raw_price():
    df = frame([10.0, 10.0, 10.0], raw=[12.22, 13.00, 14.00])
    assert n4_222_prices(df).trades == [Trade(1, 1)]


def test_t7_exits_on_reversion():
    c = [100.0] * 8 + [101, 99, 100, 101, 80, 85, 102, 103, 104, 105]
    sig = t7_oscillation(frame(c))
    assert sig.trades, "expected a snap-back entry after the drop to 80"
    tr = next(t for t in sig.trades if t.entry == 13)   # the 80 close is bar 12
    assert tr.hold == 2                                  # 102 at bar 14 is back above the mean


@pytest.mark.parametrize("h", [h for h in HYPOTHESES], ids=lambda h: h.code)
def test_no_lookahead(h):
    """Entries decided using data up to bar m must not change when later bars are revealed."""
    df = frame(random_walk(), raw=np.round(random_walk(seed=1), 2))
    m = 1100
    full = {t.entry for t in h.fn(df).trades if t.entry < m}
    cut = {t.entry for t in h.fn(df.iloc[:m]).trades if t.entry < m}
    assert full == cut
