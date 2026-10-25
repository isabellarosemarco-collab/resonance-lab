"""Scoring: excess return over random same-length windows, calendar-shift luck test, consistency.

Everything is on one master calendar (SPY's trading days) so that a calendar shift moves every
symbol's trades by the same number of days. That keeps the fact that stocks move together inside
the null distribution, instead of pretending 60 same-day trades are 60 independent bets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Market:
    dates: pd.DatetimeIndex          # master calendar
    symbols: list[str]
    open: np.ndarray                 # (n_symbols, n_dates), NaN where a symbol didn't trade
    cost: np.ndarray                 # (n_symbols,) round-trip cost as a fraction
    disc_end: int                    # last master index inside discovery (inclusive)

    _fwd: dict | None = None
    _mu: dict | None = None

    def fwd(self, hold: int) -> np.ndarray:
        """Open-to-open return from each bar to `hold` bars later. (n_symbols, n_dates)."""
        self._fwd = self._fwd or {}
        if hold not in self._fwd:
            r = np.full(self.open.shape, np.nan)
            r[:, :-hold] = self.open[:, hold:] / self.open[:, :-hold] - 1
            self._fwd[hold] = r
        return self._fwd[hold]

    def mu(self, hold: int) -> np.ndarray:
        """Each symbol's average same-length return over discovery: the 'random window' baseline."""
        self._mu = self._mu or {}
        if hold not in self._mu:
            self._mu[hold] = np.nanmean(self.fwd(hold)[:, :self.disc_end + 1 - hold], axis=1)
        return self._mu[hold]


@dataclass
class Trades:
    sym: np.ndarray                  # symbol row index
    pos: np.ndarray                  # master-calendar entry index
    hold: np.ndarray

    def __len__(self) -> int:
        return len(self.pos)

    def subset(self, mask: np.ndarray) -> "Trades":
        return Trades(self.sym[mask], self.pos[mask], self.hold[mask])


def excess(mkt: Market, tr: Trades, pos: np.ndarray | None = None,
           per_day: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(net return, excess over baseline) per trade; NaN where the trade can't be priced."""
    pos = tr.pos if pos is None else pos
    net = np.full(len(tr), np.nan)
    base = np.full(len(tr), np.nan)
    for h in np.unique(tr.hold):
        m = tr.hold == h
        net[m] = mkt.fwd(int(h))[tr.sym[m], pos[m]] - mkt.cost[tr.sym[m]]
        base[m] = mkt.mu(int(h))[tr.sym[m]]
    exc = net - base
    if per_day:
        net, exc = net / tr.hold, exc / tr.hold
    return net, exc


def shift_p(mkt: Market, tr: Trades, lo: int, hi: int, n_shifts: int, min_shift: int,
            seed: int, per_day: bool = False) -> float:
    """One-sided p: share of random calendar shifts whose mean excess >= the real one."""
    _, exc = excess(mkt, tr, per_day=per_day)
    obs = np.nanmean(exc)
    span = hi - lo + 1
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, span - min_shift, size=n_shifts)
    beats = 0
    for s in shifts:
        p2 = lo + (tr.pos - lo + s) % span
        # keep the exit inside the period so a shifted trade can't use holdout prices
        p2 = np.where(p2 + tr.hold > hi, p2 - span // 2, p2)
        _, e2 = excess(mkt, tr, pos=p2, per_day=per_day)
        if np.nanmean(e2) >= obs:
            beats += 1
    return (1 + beats) / (1 + n_shifts)


def block_consistency(mkt: Market, tr: Trades, exc: np.ndarray, years: int = 2) -> float:
    """Share of `years`-long calendar blocks whose mean excess is positive."""
    yr = mkt.dates.year.to_numpy()[tr.pos]
    first = yr.min() if len(yr) else 0
    block = (yr - first) // years
    vals = [np.nanmean(exc[block == b]) for b in np.unique(block)
            if np.isfinite(exc[block == b]).sum() >= 10]
    return float(np.mean(np.array(vals) > 0)) if vals else float("nan")


def summarize(mkt: Market, tr: Trades, per_day: bool) -> dict:
    net, exc = excess(mkt, tr, per_day=per_day)
    ok = np.isfinite(net)
    yrs = max((mkt.dates[tr.pos].max() - mkt.dates[tr.pos].min()).days / 365.25, 1e-9) if len(tr) else 1
    return {
        "trades": int(ok.sum()),
        "trades_per_yr": round(ok.sum() / yrs, 1),
        "avg_net_pct": round(100 * np.nanmean(net), 3) if ok.any() else np.nan,
        "excess_bp": round(1e4 * np.nanmean(exc), 2) if ok.any() else np.nan,
        "win_pct": round(100 * np.mean(net[ok] > 0), 1) if ok.any() else np.nan,
        "_exc": exc,
    }
