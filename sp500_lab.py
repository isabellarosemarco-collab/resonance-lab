"""S&P 500 holdout test. Rules: docs/prereg/2026-11-11-sp500-holdout.md (committed before data pull).

Universe = S&P 500 on 2023-11-01 (Wikipedia revision of 2023-10-25). Window = 2023-11-01 .. latest.
T4b / T7 / T6 use the original signal code from lab/hypotheses.py, unchanged.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from lab.hypotheses import t4b_streak_3plus, t6_resonance, t7_oscillation
from lab.stats import Market, Trades, excess, shift_p

D = "data_cache/sp500/"
START = pd.Timestamp("2023-11-01")
COST = 5.0 / 1e4
BAR = 0.05 / 4
N_SHIFTS, MIN_SHIFT, SEED = 1000, 30, 20261111


def load():
    px = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in glob.glob(D + "*.parquet") if not f.endswith(".earn.parquet")}
    spy = px.pop("SPY")
    return px, spy


def one_sided_p(t: float, n: int) -> float:
    from scipy import stats
    return float(stats.t.sf(t, max(n - 1, 1)))


def trade_tests(px, spy):
    cal = spy.index
    lo = int(cal.searchsorted(START))
    syms = sorted(px)
    opens = np.full((len(syms), len(cal)), np.nan)
    for k, s in enumerate(syms):
        opens[k] = px[s]["open"].reindex(cal).to_numpy()
    win = cal[lo:]
    mkt = Market(win, syms, opens[:, lo:], np.full(len(syms), COST), disc_end=len(win) - 1)
    out = []
    for code, fn in [("T4b losing streak 3+", t4b_streak_3plus), ("T7 snap-back", t7_oscillation), ("T6 cycles align", t6_resonance)]:
        sym, pos, hold = [], [], []
        for k, s in enumerate(syms):
            df = px[s].reindex(cal).dropna()
            if len(df) < 30:
                continue
            m = cal.get_indexer(df.index)                       # map the symbol's bars to the master calendar
            for tr in fn(df).trades:
                if tr.entry + tr.hold >= len(df):
                    continue
                p = m[tr.entry] - lo
                if p < 0 or p + tr.hold >= len(win):
                    continue
                sym.append(k); pos.append(p); hold.append(tr.hold)
        tr = Trades(np.array(sym), np.array(pos), np.array(hold))
        _, exc = excess(mkt, tr)
        ok = np.isfinite(exc)
        daily = pd.Series(exc[ok]).groupby(tr.pos[ok]).mean()          # one number per entry day
        t = daily.mean() / daily.std() * np.sqrt(len(daily))
        yrs = pd.Series(exc[ok], index=win[tr.pos[ok]].year)
        y = yrs.groupby(level=0).mean()
        luck = shift_p(mkt, tr.subset(ok), 0, len(win) - 1, N_SHIFTS, MIN_SHIFT, SEED)
        p = one_sided_p(t, len(daily))
        passed = p < BAR and luck < BAR and y.get(2024, -1) > 0 and y.get(2025, -1) > 0
        out.append({"test": code, "trades": int(ok.sum()), "avg excess": exc[ok].mean(), "t (daily)": t, "p": p,
                    "luck p": luck, "2024": y.get(2024, np.nan), "2025": y.get(2025, np.nan), "2026": y.get(2026, np.nan),
                    "PASS": passed})
    return out


def pead(px, spy):
    rows = []
    cal = spy.index
    for f in glob.glob(D + "*.earn.parquet"):
        e = pd.read_parquet(f).dropna(subset=["surprise_pct"])
        s = e["symbol"].iloc[0] if len(e) else None
        if s not in px:
            continue
        c = px[s]["close"].reindex(cal)
        ts = pd.to_datetime(e["date"], utc=True).dt.tz_convert("America/New_York")
        day = ts.dt.tz_localize(None).dt.normalize()
        pos = cal.searchsorted(day.values) + (ts.dt.hour.values >= 16)       # after-close reports react next day
        for p, sp in zip(pos, e["surprise_pct"].values):
            if p >= len(cal) or cal[p] < START or p + 60 >= len(cal):
                continue
            p0, p1 = c.iloc[p], c.iloc[p + 60]
            if np.isnan(p0) or np.isnan(p1):
                continue
            rows.append({"symbol": s, "react_day": cal[p], "surprise": sp,
                         "drift_ex": (p1 / p0 - 1) - (spy["close"].iloc[p + 60] / spy["close"].iloc[p] - 1) - COST})
    df = pd.DataFrame(rows).drop_duplicates(["symbol", "react_day"])
    df["year"] = df["react_day"].dt.year
    df["q"] = df.groupby("year")["surprise"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
    top, bot = df[df.q == 5].drift_ex, df[df.q == 1].drift_ex
    diff = top.mean() - bot.mean(); se = np.sqrt(top.var() / len(top) + bot.var() / len(bot)); t = diff / se
    y = df[df.q.isin([1, 5])].groupby(["year", "q"]).drift_ex.mean().unstack()
    ys = (y[5] - y[1]) if 5 in y and 1 in y else pd.Series(dtype=float)
    p = one_sided_p(t, min(len(top), len(bot)))
    return df, {"test": "PEAD top minus bottom quintile (60d vs SPY)", "trades": len(top) + len(bot), "avg excess": diff,
                "t (daily)": t, "p": p, "luck p": np.nan, "2024": ys.get(2024, np.nan), "2025": ys.get(2025, np.nan),
                "2026": ys.get(2026, np.nan), "PASS": p < BAR and ys.get(2024, -1) > 0 and ys.get(2025, -1) > 0,
                "_q": df.groupby("q").drift_ex.agg(["count", "mean"])}


def main():
    px, spy = load()
    print(f"Universe: {len(px)} stocks priced (of 503 on the 2023-11-01 list). Window {START.date()} .. {spy.index[-1].date()}")
    res = trade_tests(px, spy)
    ev, pr = pead(px, spy)
    qtab = pr.pop("_q"); res.append(pr)
    out = pd.DataFrame(res)
    f = out.copy()
    for c in ["avg excess", "2024", "2025", "2026"]:
        f[c] = (f[c] * 100).map(lambda v: f"{v:+.2f}%")
    for c in ["t (daily)"]:
        f[c] = f[c].round(2)
    for c in ["p", "luck p"]:
        f[c] = f[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    pd.set_option("display.width", 220)
    print(f.to_string(index=False))
    print(f"\nPass bar: p < {BAR} (and luck p < {BAR} for the trade signals), positive in 2024 and 2025.")
    print("\nPEAD by surprise quintile (60-day drift vs SPY, net):")
    print((qtab.assign(mean=lambda d: (d["mean"] * 100).round(2).astype(str) + "%")).to_string())
    out.to_csv("results/sp500_holdout_scoreboard.csv", index=False)
    ev.to_csv("results/sp500_pead_events.csv", index=False)


if __name__ == "__main__":
    main()
