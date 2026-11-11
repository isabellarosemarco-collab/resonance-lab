"""Edge Lab round 4: hindsight-proof universes only (broad ETFs and asset classes).

Rules fixed before running. Split: discovery through 2023-10-31, holdout from 2023-11-01.
Cash earns the 3-month T-bill. ETF cost 2 bp round trip.

V1 Volatility-managed SPY: each day hold SPY at weight min(1.5, target_vol / trailing 20d vol),
   target_vol = SPY's long-run 16%; rest in T-bills (weights >1 are 1.5x max, financed at T-bill).
V2 T4 on ETFs: 13 ETFs (SPY QQQ IWM DIA + 9 sectors); 3+ down closes -> buy next open, sell 3 opens
   later; equal weight across open positions, T-bills otherwise.
V3 Low-volatility sectors: each month hold the 3 of 9 sectors with the lowest 60d volatility.
V4 Global asset momentum (GEM-style): each month hold whichever of SPY / EFA has the higher 12-1
   return if it beats T-bills; otherwise hold IEF (intermediate Treasuries). Starts 2003.
V5 Pre-holiday: hold SPY only on the trading day before a market holiday (weekday the market is
   closed), T-bills otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import edge_lab as L
import edge_lab2 as L2
import edge_lab3 as L3
from lab.config import ETFS, DISCOVERY_END, HOLDOUT_START

E2 = "data_cache/edge2/"


def ld(s):
    try:
        return L.load(s)
    except Exception:
        return pd.read_parquet(E2 + f"{s}.parquet")[["open", "close"]]


def v1_vol_managed(spy, cash, target=0.16, cap=1.5):
    r = spy["close"].pct_change()
    vol = r.rolling(20).std() * np.sqrt(252)
    w = (target / vol).clip(upper=cap).shift(1)
    c = cash.reindex(r.index).fillna(0)
    turn = w.diff().abs().fillna(0)
    return (w * r + (1 - w) * c - turn * L.ETF_COST / 2).fillna(0)


def v2_t4_etfs(px13, cash):
    px = {s: px13[s] for s in px13}
    old = L3.ETFS, L3.STOCKS
    L3.ETFS, L3.STOCKS = list(px.keys()), []
    try:
        out = L3.t4_portfolio(px, cash)
    finally:
        L3.ETFS, L3.STOCKS = old
    return out + (L.STK_COST - L.ETF_COST) * 0       # costs: t4_portfolio charges stock cost (conservative)


def v3_low_vol_sectors(cash):
    close = pd.DataFrame({s: ld(s)["close"] for s in L2.SECTORS}).dropna()
    vol = close.pct_change().rolling(60).std()
    me = L.month_ends(close.index)
    picks = {d: vol.loc[d].nsmallest(3).index.tolist() for d in me if vol.loc[d].notna().all()}
    return L2.rotation(close, picks, L.ETF_COST, cash.reindex(close.index).fillna(0))


def v4_gem(cash):
    close = pd.DataFrame({s: ld(s)["close"] for s in ["SPY", "EFA", "IEF"]}).dropna()
    me = L.month_ends(close.index)
    mom = L.momentum_12_1(close, me)
    tb12 = cash.rolling(252).sum().reindex(me)
    picks = {}
    for d, row in mom.iterrows():
        if row.isna().any():
            continue
        best = "SPY" if row["SPY"] >= row["EFA"] else "EFA"
        picks[d] = [best] if row[best] > tb12.get(d, 0) else ["IEF"]
    return L2.rotation(close, picks, L.ETF_COST, cash.reindex(close.index).fillna(0))


def v5_pre_holiday(spy, cash):
    idx = spy.index
    bdays = pd.bdate_range(idx[0], idx[-1])
    holidays = bdays.difference(idx)
    nxt = pd.Series(idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=1)])), index=idx)
    pre = pd.Series([any((h > d) and (h < n) for h in holidays[(holidays > d) & (holidays < n)]) for d, n in nxt.items()], index=idx)
    r = spy["close"].pct_change()
    held = pre                                    # return earned on the pre-holiday day itself (bought prior close)
    entries = held.astype(float)
    return (r.where(held, cash.reindex(idx).fillna(0)) - entries * L.ETF_COST).fillna(0)


def main():
    spy = L.load("SPY"); bench = spy["close"].pct_change()
    cash = L2.tbill_daily(spy.index)
    px13 = {s: ld(s) for s in ETFS + L2.SECTORS}
    series = {
        "SPY": bench,
        "V1 Volatility-managed SPY": v1_vol_managed(spy, cash),
        "V2 T4 on 13 ETFs": v2_t4_etfs(px13, cash),
        "V3 Low-vol sectors (3 of 9)": v3_low_vol_sectors(cash),
        "V4 Global asset momentum": v4_gem(cash),
        "V5 Pre-holiday SPY": v5_pre_holiday(spy, cash),
    }
    periods = {"Discovery": ("2004-01-01", DISCOVERY_END), "Holdout": (HOLDOUT_START, None)}
    rows = []
    for pname, (a, b) in periods.items():
        bm = bench.loc[a:b]
        for name, r in series.items():
            rr = r.reindex(bm.index).fillna(0.0)
            s = L.stats(rr); y = L3.yearly(rr)
            rows.append({"Period": pname, "Strategy": name, "CAGR": s["CAGR"], "Sharpe": s["Sharpe"],
                         "MaxDD": s["MaxDD"], "Losing yrs": f"{(y < 0).sum()}/{len(y)}",
                         "Yrs beat SPY": f"{(y > L3.yearly(bm)).sum()}/{len(y)}" if name != "SPY" else "",
                         "t vs SPY": L.excess_t(rr, bm) if name != "SPY" else np.nan})
    out = pd.DataFrame(rows); f = out.copy()
    for c in ["CAGR", "MaxDD"]:
        f[c] = (f[c] * 100).round(1).astype(str) + "%"
    f["Sharpe"] = f["Sharpe"].round(2); f["t vs SPY"] = f["t vs SPY"].round(2)
    pd.set_option("display.width", 200)
    print(f.to_string(index=False))
    out.to_csv("results/edge_lab4_scoreboard.csv", index=False)


if __name__ == "__main__":
    main()
