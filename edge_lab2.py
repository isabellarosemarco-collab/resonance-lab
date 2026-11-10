"""Edge Lab round 2: better data.

Adds (rules fixed before running):
  * Cash earns the 3-month T-bill (FRED DTB3) instead of 0%.
  * R1 Sector momentum: each month hold the top 3 of the 9 original sector SPDRs by 12-1 month
    return, equal weight; if SPY's 12-1 return is below the T-bill, hold T-bills instead.
  * R2/R3 ETF dual momentum and 200-day trend filter, re-run with T-bill cash.
  * R4 Post-earnings drift (real EPS surprises, Yahoo): enter at the CLOSE of the first trading
    day that reacts to the report (after-close reports react next day), so the day-one jump is
    excluded. Hold 60 trading days. Measured as return minus SPY over the same window.
    Bias control: top-quintile surprises minus bottom-quintile surprises on the same 56 stocks.
  * R5 Tradable PEAD portfolio: hold every stock in its 60-day post-beat window (top-quintile
    surprise by trailing-1-year cutoffs, no look-ahead), equal weight; cash in T-bills.
Split: discovery through 2023-10-31, holdout from 2023-11-01 (same as the Resonance Lab).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import edge_lab as L
from lab.config import ETFS, STOCKS, DISCOVERY_END, HOLDOUT_START

E2 = "data_cache/edge2/"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def tbill_daily(idx: pd.DatetimeIndex) -> pd.Series:
    tb = pd.read_csv(E2 + "DTB3.csv", parse_dates=["observation_date"]).set_index("observation_date")["DTB3"]
    tb = pd.to_numeric(tb, errors="coerce").ffill()
    return (tb.reindex(idx, method="ffill") / 100 / 252).fillna(0.0)


def rotation(close: pd.DataFrame, picks: dict, cost: float, cash: pd.Series) -> pd.Series:
    ret = close.pct_change()
    cur, port, tov = [], [], []
    for d in close.index:
        r = ret.loc[d, cur].mean() if cur else cash.loc[d]
        port.append(r if pd.notna(r) else 0.0)
        t = 0.0
        if d in picks and set(picks[d]) != set(cur):
            t = len(set(picks[d]) ^ set(cur)) / max(len(picks[d]), len(cur), 1)
            cur = picks[d]
        tov.append(t)
    p = pd.Series(port, index=close.index)
    return p - pd.Series(tov, index=close.index).shift(1).fillna(0) * cost


def r1_sector_momentum(spy: pd.DataFrame, cash: pd.Series) -> pd.Series:
    close = pd.DataFrame({s: pd.read_parquet(E2 + f"{s}.parquet")["close"] for s in SECTORS}).dropna()
    me = L.month_ends(close.index)
    mom = L.momentum_12_1(close, me)
    spy_m = L.momentum_12_1(spy[["close"]].reindex(close.index), me)["close"]
    tb_12m = cash.rolling(252).sum().reindex(me)
    picks = {}
    for d, row in mom.iterrows():
        if row.isna().any() or pd.isna(spy_m.get(d)):
            continue
        picks[d] = row.nlargest(3).index.tolist() if spy_m[d] > tb_12m.get(d, 0) else []
    return rotation(close, picks, L.ETF_COST, cash.reindex(close.index).fillna(0))


def r2_etf_dual_momentum(px: dict, cash: pd.Series) -> pd.Series:
    close = pd.DataFrame({s: px[s]["close"] for s in ETFS}).dropna()
    me = L.month_ends(close.index)
    mom = L.momentum_12_1(close, me)
    tb_12m = cash.rolling(252).sum().reindex(me)
    picks = {}
    for d, row in mom.iterrows():
        if row.isna().all():
            continue
        best = row.idxmax()
        picks[d] = [best] if row[best] > tb_12m.get(d, 0) else []
    return rotation(close, picks, L.ETF_COST, cash.reindex(close.index).fillna(0))


def r3_trend(spy: pd.DataFrame, cash: pd.Series) -> pd.Series:
    c = spy["close"]
    held = (c > c.rolling(200).mean()).shift(1, fill_value=False)
    r = c.pct_change()
    sw = (held != held.shift(1, fill_value=False)).astype(float)
    return r.where(held, cash.reindex(c.index).fillna(0)) - sw * L.ETF_COST / 2


def earnings_events(close: pd.DataFrame) -> pd.DataFrame:
    e = pd.read_parquet(E2 + "earnings.parquet").dropna(subset=["surprise_pct"])
    e["ts"] = pd.to_datetime(e["date"], utc=True).dt.tz_convert("America/New_York")
    after_close = e["ts"].dt.hour >= 16
    day = e["ts"].dt.tz_localize(None).dt.normalize()
    idx = close.index
    pos = idx.searchsorted(day.values)                     # first trading day on/after report date
    pos = np.where(after_close.values, pos + 1, pos)       # after-close reports react the next day
    pos = np.where((pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == day.values) & after_close.values, pos, pos)
    e["react_i"] = pos
    e = e[(e["react_i"] >= 0) & (e["react_i"] < len(idx))].copy()
    e["react_day"] = idx[e["react_i"].values]
    e = e.drop_duplicates(["symbol", "react_day"])
    return e


def r4_pead_event_study(px: dict, hold: int = 60) -> pd.DataFrame:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    spy = px["SPY"]["close"].reindex(close.index)
    ev = earnings_events(close)
    out = []
    for _, r in ev.iterrows():
        i, s = int(r["react_i"]), r["symbol"]
        j = i + hold
        if j >= len(close):
            continue
        p0, p1 = close[s].iloc[i], close[s].iloc[j]
        if pd.isna(p0) or pd.isna(p1):
            continue
        out.append({"symbol": s, "react_day": close.index[i], "surprise": r["surprise_pct"],
                    "drift_ex": (p1 / p0 - 1) - (spy.iloc[j] / spy.iloc[i] - 1) - L.STK_COST})
    df = pd.DataFrame(out)
    df["year"] = df["react_day"].dt.year
    # quintiles within each calendar year (no look-ahead across years)
    df["q"] = df.groupby("year")["surprise"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
    return df


def r5_pead_portfolio(px: dict, cash: pd.Series, hold: int = 60) -> pd.Series:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    ev = earnings_events(close).sort_values("react_day")
    # top-quintile cutoff from the trailing 4 quarters of ALL events (known at the time)
    cut = []
    for _, r in ev.iterrows():
        past = ev[(ev["react_day"] < r["react_day"]) & (ev["react_day"] >= r["react_day"] - pd.Timedelta(days=365))]
        cut.append(past["surprise_pct"].quantile(0.8) if len(past) >= 50 else np.nan)
    ev["cut"] = cut
    picks = ev[ev["surprise_pct"] >= ev["cut"]]
    held = np.zeros(close.shape, dtype=bool)
    entries = np.zeros(close.shape, dtype=bool)
    cols = {s: k for k, s in enumerate(close.columns)}
    for _, r in picks.iterrows():
        i, k = int(r["react_i"]), cols[r["symbol"]]
        held[i + 1:i + 1 + hold, k] = True                  # bought at the reaction-day close
        if i + 1 < len(close):
            entries[i + 1, k] = True
    ret = close.pct_change().values
    n = held.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(n > 0, np.nansum(np.where(held, ret, 0), axis=1) / np.maximum(n, 1), cash.reindex(close.index).fillna(0).values)
        cost = np.where(n > 0, entries.sum(axis=1) / np.maximum(n, 1), 0) * L.STK_COST
    return pd.Series(port - cost, index=close.index)


def main():
    px = {s: L.load(s) for s in ETFS + STOCKS}
    spy = px["SPY"]
    bench = spy["close"].pct_change()
    cash = tbill_daily(spy.index)
    ew = pd.DataFrame({s: px[s]["close"] for s in STOCKS}).pct_change().mean(axis=1)
    strategies = {
        "R1 Sector momentum (top 3 of 9, T-bill filter)": r1_sector_momentum(spy, cash),
        "R2 ETF dual momentum (T-bill cash)": r2_etf_dual_momentum(px, cash),
        "R3 Trend filter 200d (T-bill cash)": r3_trend(spy, cash),
        "R5 PEAD portfolio (real surprises)": r5_pead_portfolio(px, cash),
    }
    periods = {"Discovery": ("2002-01-01", DISCOVERY_END), "Holdout": (HOLDOUT_START, None)}
    rows = []
    for pname, (a, b) in periods.items():
        bm = bench.loc[a:b]
        rows.append({"Period": pname, "Strategy": "Buy & hold SPY", **L.stats(bm), "t vs SPY": np.nan, "t vs EW56": np.nan})
        rows.append({"Period": pname, "Strategy": "Equal-weight hold of the 56 stocks", **L.stats(ew.loc[a:b]),
                     "t vs SPY": L.excess_t(ew.loc[a:b], bm), "t vs EW56": np.nan})
        for name, r in strategies.items():
            rr = r.reindex(bm.index).fillna(0.0)
            rows.append({"Period": pname, "Strategy": name, **L.stats(rr), "t vs SPY": L.excess_t(rr, bm),
                         "t vs EW56": L.excess_t(rr, ew.loc[a:b].reindex(bm.index)) if name.startswith("R5") else np.nan})
    out = pd.DataFrame(rows)
    f = out.copy()
    for c in ["CAGR", "MaxDD", "Vol"]:
        f[c] = (f[c] * 100).round(1).astype(str) + "%"
    for c in ["Sharpe", "t vs SPY", "t vs EW56"]:
        f[c] = f[c].round(2)
    pd.set_option("display.width", 200)
    print(f.to_string(index=False))
    out.to_csv("results/edge_lab2_scoreboard.csv", index=False)

    df = r4_pead_event_study(px)
    print("\nR4 Post-earnings drift, 60 trading days after the reaction-day close, minus SPY (net of 5 bp):")
    for pname, (a, b) in periods.items():
        d = df[(df["react_day"] >= pd.Timestamp(a)) & ((b is None) | (df["react_day"] <= pd.Timestamp(b or "2100")))]
        g = d.groupby("q")["drift_ex"].agg(["count", "mean"])
        top, bot = d[d.q == 5]["drift_ex"], d[d.q == 1]["drift_ex"]
        diff = top.mean() - bot.mean()
        se = np.sqrt(top.var() / len(top) + bot.var() / len(bot))
        print(f"  {pname}: " + "  ".join(f"Q{int(q)} n={int(r['count'])} {r['mean']*100:+.2f}%" for q, r in g.iterrows())
              + f"   | top-minus-bottom {diff*100:+.2f}% (t {diff/se:.2f})")
    df.to_csv("results/edge_lab2_pead_events.csv", index=False)


if __name__ == "__main__":
    main()
