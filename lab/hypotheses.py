"""The pre-registered hypotheses. Each one maps a symbol's daily bars to trades.

A trade is (entry bar, hold): buy at the OPEN of `entry`, sell at the OPEN of `entry + hold`.
Signals computed from a bar's close enter on the next bar, so nothing peeks at the future.
Calendar signals (dates known in advance) enter on the special day itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from lab.cycles import band, dominant_cycle, trough_at, turns_up
from lab.numerology import date_root, digit_root, has_222, is_369, personal_day


@dataclass(frozen=True)
class Trade:
    entry: int
    hold: int


@dataclass
class Signal:
    trades: list[Trade]
    twin: list[Trade] = field(default_factory=list)   # non-magic control for attribution


@dataclass(frozen=True)
class Hypothesis:
    code: str
    name: str
    fn: Callable[[pd.DataFrame], Signal]
    per_day: bool = False     # compare returns per day held (T2 compares hold lengths)
    counted: bool = True      # counts toward the Bonferroni family; controls don't


def after_close(mask: np.ndarray, hold: int) -> list[Trade]:
    """Signal known at bar t's close -> enter at bar t+1's open."""
    return [Trade(int(t) + 1, hold) for t in np.flatnonzero(mask) if t + 1 < len(mask)]


def on_day(mask: np.ndarray, hold: int) -> list[Trade]:
    """Calendar signal: enter at the open of the flagged day."""
    return [Trade(int(t), hold) for t in np.flatnonzero(mask)]


def drop_overlaps(trades: list[Trade]) -> list[Trade]:
    """One position per symbol at a time; a new trade may start at the open the last one exits."""
    out, free_at = [], -1
    for tr in sorted(trades, key=lambda x: x.entry):
        if tr.entry >= free_at:
            out.append(tr)
            free_at = tr.entry + tr.hold
    return out


def down_streak(close: np.ndarray) -> np.ndarray:
    s = np.zeros(len(close), dtype=int)
    for i in range(1, len(close)):
        s[i] = s[i - 1] + 1 if close[i] < close[i - 1] else 0
    return s


# ------------------------------------------------------------------ Tesla group

def t1_369_days(df: pd.DataFrame) -> Signal:
    m = np.array([is_369(d.day) for d in df.index])
    return Signal(on_day(m, 1), on_day(~m, 1))


def t2_369_holds(df: pd.DataFrame) -> Signal:
    c = df["close"].to_numpy()
    hi20 = pd.Series(c).rolling(20).max().to_numpy()
    prev_hi = np.concatenate([[np.nan], hi20[:-1]])
    breakout = (c >= hi20) & (c > prev_hi)
    magic = [tr for h in (3, 6, 9) for tr in drop_overlaps(after_close(breakout, h))]
    other = [tr for h in (4, 5, 7, 8) for tr in drop_overlaps(after_close(breakout, h))]
    return Signal(magic, other)


def t3_369_levels(df: pd.DataFrame) -> Signal:
    raw = df["raw_close"].to_numpy()
    prev = np.concatenate([[np.nan], raw[:-1]])
    hi = np.floor(prev)                 # highest whole-dollar level that could be crossed
    one_level = (raw < hi) & (raw >= hi - 1) & (prev >= hi)   # crossed exactly one level
    roots = np.array([is_369(int(h)) if np.isfinite(h) and h > 0 else False for h in hi])
    return Signal(after_close(one_level & roots, 3), after_close(one_level & ~roots, 3))


def t4_369_streaks(df: pd.DataFrame) -> Signal:
    s = down_streak(df["close"].to_numpy())
    return Signal(after_close(np.isin(s, (3, 6, 9)), 3), after_close(np.isin(s, (2, 4, 5, 7, 8)), 3))


def t5_frequency(df: pd.DataFrame, window: int = 252) -> Signal:
    x = np.log(df["close"].to_numpy())
    trades = []
    for t in range(window - 1, len(x) - 1):
        period, phase = dominant_cycle(x[t - window + 1:t + 1])
        if trough_at(period, phase, window):          # next bar is the projected low
            trades.append(Trade(t + 1, max(1, int(round(period / 2)))))
    return Signal(trades)


def t6_resonance(df: pd.DataFrame) -> Signal:
    c = df["close"].to_numpy()
    m = turns_up(band(c, 3)) & turns_up(band(c, 6)) & turns_up(band(c, 9))
    return Signal(after_close(m, 3))


def t7_oscillation(df: pd.DataFrame, n: int = 9) -> Signal:
    c = df["close"]
    mean, sd = c.rolling(n).mean(), c.rolling(n).std()
    z = ((c - mean) / sd).to_numpy()
    above = (c >= mean).to_numpy()
    trades = []
    for t in np.flatnonzero(z < -2):
        if t + 1 >= len(c):
            continue
        hold = n
        for u in range(t + 1, min(t + n, len(c) - 1) + 1):
            if above[u]:                       # reverted at u's close -> exit next open
                hold = int(u - t)
                break
        trades.append(Trade(int(t) + 1, hold))
    return Signal(trades)


# ------------------------------------------------------------------ numerology / personal

def n1_date_root(df: pd.DataFrame) -> Signal:
    m = np.array([date_root(d) in (3, 6, 9) for d in df.index])
    return Signal(on_day(m, 1), on_day(~m, 1))


def n2_22_days(df: pd.DataFrame) -> Signal:
    m = np.array([d.day == 22 for d in df.index])
    return Signal(on_day(m, 1), on_day(~m, 1))


def n3_personal_day(df: pd.DataFrame) -> Signal:
    m = np.array([personal_day(d) in (9, 22) for d in df.index])
    return Signal(on_day(m, 1), on_day(~m, 1))


def n4_222_prices(df: pd.DataFrame) -> Signal:
    m = np.array([has_222(p) for p in df["raw_close"].to_numpy()])
    return Signal(after_close(m, 1), after_close(~m, 1))


# ------------------------------------------------------------------ pattern recognition

def p1_echo(df: pd.DataFrame, window: int = 9, k: int = 9, horizon: int = 3,
            min_history: int = 504, chunk: int = 512) -> Signal:
    """Find the 9 past 9-day shapes most like the last 9 days; follow their next-3-day result."""
    c = df["close"].to_numpy()
    n = len(c)
    r = np.concatenate([[np.nan], np.diff(np.log(c))])
    ends = np.arange(window, n)                       # window ending at t uses r[t-8..t]
    W = np.stack([r[t - window + 1:t + 1] for t in ends])
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
    outcome = np.full(n, np.nan)
    outcome[:n - horizon] = c[horizon:] / c[:-horizon] - 1
    sq = (W ** 2).sum(1)

    signal = np.zeros(n, dtype=bool)
    queries = np.flatnonzero(ends >= min_history)
    for lo in range(0, len(queries), chunk):
        q = queries[lo:lo + chunk]
        d2 = sq[q, None] + sq[None, :] - 2 * W[q] @ W.T
        # a candidate's outcome must be known by the query's close, and it must not overlap it
        latest_ok = ends[q] - horizon - window
        d2[ends[None, :] > latest_ok[:, None]] = np.inf
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        res = outcome[ends[nn]]
        ok = (np.nanmean(res, 1) > 0) & ((res > 0).sum(1) >= 6)
        signal[ends[q][ok]] = True
    return Signal(after_close(signal, horizon))


# ------------------------------------------------------------------ reference control

def c2_momentum(df: pd.DataFrame) -> Signal:
    """Time-series momentum: hold for a month when the 12-month-minus-last-month return is up."""
    c = df["close"]
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    idx = df.index
    first_of_month = np.r_[False, idx.month[1:] != idx.month[:-1]]
    # momentum is measured at the prior close, so this is known at the open
    prev_mom = np.r_[np.nan, mom[:-1]]
    return Signal(on_day(first_of_month & (prev_mom > 0), 21))


HYPOTHESES: list[Hypothesis] = [
    Hypothesis("T1", "3-6-9 days", t1_369_days),
    Hypothesis("T2", "3-6-9 holds", t2_369_holds, per_day=True),
    Hypothesis("T3", "3-6-9 price levels", t3_369_levels),
    Hypothesis("T4", "3-6-9 streaks", t4_369_streaks),
    Hypothesis("T5", "Frequency (dominant cycle)", t5_frequency),
    Hypothesis("T6", "Resonance (3/6/9 cycles align)", t6_resonance),
    Hypothesis("T7", "Oscillation (snap-back)", t7_oscillation),
    Hypothesis("N1", "Date root 3/6/9", n1_date_root),
    Hypothesis("N2", "22nd of the month", n2_22_days),
    Hypothesis("N3", "Personal day 9 or 22", n3_personal_day),
    Hypothesis("N4", "222 prices", n4_222_prices),
    Hypothesis("P1", "Echo (pattern match)", p1_echo),
    Hypothesis("C2", "Momentum (reference control)", c2_momentum, counted=False),
]
