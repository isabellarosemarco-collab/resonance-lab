"""Fibonacci retracement test. Rules fixed before running.

Swing: on day t, H = highest close of the last 60 days; Lo = lowest close of the 120 days before
H's date. If H/Lo - 1 >= 20%, the stock is in a pullback from a qualifying run-up.
Level r is "touched" on the first close at or below H - r*(H - Lo) after H (and above Lo).
Event: buy that close, measure the 10-day forward return minus SPY (net 5 bp).
Fibonacci levels: 23.6, 38.2, 50, 61.8, 78.6.  Controls: 30, 45, 55, 70.
Each (stock, swing high, level) counts once. Discovery through 2023-10-31; holdout after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import edge_lab as L
from lab.config import ETFS, STOCKS, DISCOVERY_END, HOLDOUT_START

FIB = [0.236, 0.382, 0.5, 0.618, 0.786]
CTRL = [0.30, 0.45, 0.55, 0.70]
FWD = 10


def events(close: pd.Series, spy: pd.Series) -> list[dict]:
    c = close.dropna().values
    dates = close.dropna().index
    sp = spy.reindex(dates).values
    out, seen = [], set()
    for t in range(180, len(c) - FWD):
        win = c[t - 59:t + 1]
        hi_i = t - 59 + int(np.argmax(win))
        if hi_i == t:
            continue
        H = c[hi_i]
        lo = c[max(0, hi_i - 120):hi_i + 1].min()
        if H / lo - 1 < 0.20:
            continue
        since = c[hi_i + 1:t + 1]
        for r in FIB + CTRL:
            key = (hi_i, r)
            if key in seen:
                continue
            lvl = H - r * (H - lo)
            if c[t] <= lvl and c[t] > lo and (since[:-1] > lvl).all():
                seen.add(key)
                ret = c[t + FWD] / c[t] - 1 - (sp[t + FWD] / sp[t] - 1) - L.STK_COST
                out.append({"date": dates[t], "level": r, "fib": r in FIB, "ex10": ret})
    return out


def main():
    px = {s: L.load(s) for s in ETFS + STOCKS}
    spy = px["SPY"]["close"]
    rows = []
    for s in ETFS + STOCKS:
        for e in events(px[s]["close"], spy):
            e["symbol"] = s; rows.append(e)
    df = pd.DataFrame(rows)
    df.to_csv("results/fib_events.csv", index=False)
    for pname, (a, b) in {"Discovery": ("1999-01-01", DISCOVERY_END), "Holdout": (HOLDOUT_START, "2100-01-01")}.items():
        d = df[(df.date >= a) & (df.date <= b)]
        g = d.groupby("level")["ex10"].agg(["count", "mean", lambda x: (x > 0).mean()])
        g.columns = ["n", "avg 10d vs SPY", "bounce rate"]
        print(f"\n{pname}")
        for lvl, r in g.iterrows():
            tag = "FIB " if lvl in FIB else "ctrl"
            print(f"  {tag} {lvl*100:5.1f}%  n={int(r['n']):5d}  avg {r['avg 10d vs SPY']*100:+.2f}%  bounce {r['bounce rate']*100:.1f}%")
        f, c = d[d.fib].ex10, d[~d.fib].ex10
        diff = f.mean() - c.mean(); se = np.sqrt(f.var() / len(f) + c.var() / len(c))
        print(f"  Fibonacci levels {f.mean()*100:+.2f}% vs control levels {c.mean()*100:+.2f}%  diff {diff*100:+.2f}% (t {diff/se:.2f})")


if __name__ == "__main__":
    main()
