"""The $1,000 lab account: 4 slots of $250, buys and sells at the open, marked at the close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.config import SLOT_SIZE, SLOTS


def simulate(dates: pd.DatetimeIndex, open_: np.ndarray, close: np.ndarray, cost: np.ndarray,
             sym: np.ndarray, pos: np.ndarray, hold: np.ndarray,
             capital: float = SLOT_SIZE * SLOTS) -> pd.Series:
    """Daily closing equity. Signals beyond 4 open slots are skipped (first come, by symbol order).

    Each new position gets a quarter of current equity (what Bella would do by hand), so slots
    compound and a losing trade doesn't permanently shrink a slot. Cash earns nothing.
    """
    order = np.lexsort((sym, pos))
    sym, pos, hold = sym[order], pos[order], hold[order]
    by_day: dict[int, list[int]] = {}
    for i, p in enumerate(pos):
        by_day.setdefault(int(p), []).append(i)

    cash, open_pos = capital, []          # open_pos: (symbol row, shares, exit index)
    held = set()
    eq = np.empty(len(dates))
    for t in range(len(dates)):
        still = []
        for s, sh, ex in open_pos:        # exits first, at today's open
            px = open_[s, t]
            if t >= ex and np.isfinite(px):
                cash += sh * px * (1 - cost[s])
                held.discard(s)
            else:
                still.append((s, sh, ex))
        open_pos = still
        todays = by_day.get(t, [])
        if todays:
            at_open = cash + sum(sh * open_[s, t] for s, sh, _ in open_pos if np.isfinite(open_[s, t]))
            size = at_open / SLOTS
        for i in todays:
            s = int(sym[i])
            px = open_[s, t]
            if len(open_pos) < SLOTS and s not in held and np.isfinite(px) and cash > 1.0:
                spend = min(size, cash)
                open_pos.append((s, spend / px, t + int(hold[i])))
                held.add(s)
                cash -= spend
        marks = [sh * (close[s, t] if np.isfinite(close[s, t]) else open_[s, t]) for s, sh, _ in open_pos]
        eq[t] = cash + float(np.nansum(marks))
    return pd.Series(eq, index=dates)


def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def cagr(equity: pd.Series) -> float:
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else float("nan")
