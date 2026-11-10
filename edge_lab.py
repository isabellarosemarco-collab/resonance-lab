"""Edge Lab: test 5 documented market effects against buy-and-hold SPY, net of trading costs.

Same data and split as the Resonance Lab:
  discovery  1999-01-04 .. 2023-10-31
  holdout    2023-11-01 .. latest cached bar
Every strategy is fixed in this file BEFORE looking at results. Nothing is tuned on the holdout.

Strategies
  S1 ETF dual momentum  : each month hold the ETF (SPY/QQQ/IWM/DIA) with the best 12-1 month return,
                          or cash if that return is below 0 (absolute-momentum filter).
  S2 Stock momentum     : each month hold the 10 stocks with the best 12-1 month return, equal weight.
                          (Universe = today's megacaps, so this one has survivorship bias. Treat as an upper bound.)
  S3 Turn of month      : hold SPY from the close of the 2nd-to-last trading day of each month
                          through the close of the 3rd trading day of the next month; cash otherwise.
  S4 Overnight SPY      : buy SPY at the close, sell at the next open, every day.
  S5 Trend filter       : hold SPY when its close is above its 200-day average; cash otherwise.
  S6 Gap drift (proxy for post-earnings drift): when a stock opens >=4% above the prior close,
                          buy that close and hold 20 trading days (equal-weight across open positions).
Cash earns 0 (conservative). Costs: ETFs 2 bp round trip, stocks 5 bp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.config import ETFS, STOCKS, DISCOVERY_END, HOLDOUT_START

CACHE = "data_cache/{}_1999-01-01.parquet"
ETF_COST = 2.0 / 1e4
STK_COST = 5.0 / 1e4


def load(sym: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE.format(sym))[["open", "close"]]


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd, "Vol": vol}


def excess_t(r: pd.Series, b: pd.Series) -> float:
    d = (r - b).dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def momentum_12_1(close: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    m = close.loc[me]
    return m.shift(1) / m.shift(12) - 1          # return from t-12 to t-1 months


def monthly_rotation(close: pd.DataFrame, picks: dict, cost: float) -> pd.Series:
    """picks: {month_end_date: list_of_symbols (empty = cash)} applied from next day."""
    ret = close.pct_change()
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cur, prev = [], []
    turnover = pd.Series(0.0, index=close.index)
    for d in close.index:
        w.loc[d, cur] = 1.0 / len(cur) if cur else 0.0
        if d in picks:
            new = picks[d]
            if set(new) != set(cur):
                changed = len(set(new) ^ set(cur))
                turnover[d] = changed / max(len(new), len(cur), 1)
            cur = new
    port = (w.shift(0) * ret).sum(axis=1)       # weights set at prior close apply to today's return
    port = port.where(w.sum(axis=1) > 0, 0.0)
    return port - turnover.shift(1).fillna(0) * cost


def s1_etf_momentum(px: dict) -> pd.Series:
    close = pd.DataFrame({s: px[s]["close"] for s in ETFS}).dropna()
    me = month_ends(close.index)
    mom = momentum_12_1(close, me)
    picks = {}
    for d, row in mom.iterrows():
        if row.isna().all():
            continue
        best = row.idxmax()
        picks[d] = [best] if row[best] > 0 else []
    return monthly_rotation(close, picks, ETF_COST)


def s2_stock_momentum(px: dict, n: int = 10) -> pd.Series:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    me = month_ends(close.index)
    mom = momentum_12_1(close, me)
    picks = {d: row.dropna().nlargest(n).index.tolist() for d, row in mom.iterrows() if row.notna().sum() >= n}
    return monthly_rotation(close.ffill(), picks, STK_COST)


def s3_turn_of_month(spy: pd.DataFrame) -> pd.Series:
    r = spy["close"].pct_change()
    idx = spy.index
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    pos_in_month = ym.groupby(ym).cumcount()                    # 0 = first day
    from_end = ym.groupby(ym).cumcount(ascending=False)          # 0 = last day
    # return on day d is earned if we held from prior close: in-window days are
    # last trading day (from_end==0) and first 3 days (pos 0,1,2)
    inwin = (from_end == 0) | (pos_in_month <= 2)
    entries = (inwin & ~inwin.shift(1, fill_value=False)).astype(float)
    return r.where(inwin, 0.0) - entries * ETF_COST


def s4_overnight(spy: pd.DataFrame) -> pd.Series:
    r = spy["open"] / spy["close"].shift(1) - 1
    return r - ETF_COST


def s5_trend(spy: pd.DataFrame) -> pd.Series:
    c = spy["close"]
    held = (c > c.rolling(200).mean()).shift(1, fill_value=False)
    r = c.pct_change()
    switches = (held != held.shift(1, fill_value=False)).astype(float)
    return r.where(held, 0.0) - switches * ETF_COST / 2


def s6_gap_drift(px: dict, gap: float = 0.04, hold: int = 20) -> pd.Series:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    open_ = pd.DataFrame({s: px[s]["open"] for s in STOCKS})
    sig = (open_ / close.shift(1) - 1) >= gap                   # gap-up at the open on day t
    held = pd.DataFrame(False, index=close.index, columns=close.columns)
    arr = sig.values
    h = np.zeros_like(arr, dtype=bool)
    for j in range(arr.shape[1]):
        for i in np.where(arr[:, j])[0]:
            h[i + 1:i + 1 + hold, j] = True                    # buy the signal day's close, hold N days
    held[:] = h
    ret = close.pct_change()
    n = held.sum(axis=1)
    port = (ret.where(held)).sum(axis=1) / n.replace(0, np.nan)
    entries = pd.DataFrame(arr, index=close.index, columns=close.columns).shift(1, fill_value=False).sum(axis=1)
    cost = (entries / n.replace(0, np.nan)).fillna(0) * STK_COST
    return port.fillna(0.0) - cost


def main():
    px = {s: load(s) for s in ETFS + STOCKS}
    spy = px["SPY"]
    bench = spy["close"].pct_change()
    strategies = {
        "S1 ETF dual momentum": s1_etf_momentum(px),
        "S2 Stock momentum (biased universe)": s2_stock_momentum(px),
        "S3 Turn of month": s3_turn_of_month(spy),
        "S4 Overnight SPY": s4_overnight(spy),
        "S5 Trend filter (200d)": s5_trend(spy),
        "S6 Gap drift (PEAD proxy)": s6_gap_drift(px),
    }
    periods = {"Discovery 1999-2023": (None, DISCOVERY_END), "Holdout Nov23-Oct26": (HOLDOUT_START, None)}
    rows = []
    for pname, (a, b) in periods.items():
        bm = bench.loc[a:b].iloc[1:] if a is None else bench.loc[a:b]
        bs = stats(bm)
        rows.append({"Period": pname, "Strategy": "Buy & hold SPY", **bs, "Excess t": np.nan,
                     "Time in market": 1.0})
        for name, r in strategies.items():
            rr = r.reindex(bm.index).fillna(0.0)
            s = stats(rr)
            tim = (rr != 0).mean()
            rows.append({"Period": pname, "Strategy": name, **s, "Excess t": excess_t(rr, bm),
                         "Time in market": tim})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    fmt = out.copy()
    for c in ["CAGR", "MaxDD", "Vol", "Time in market"]:
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    fmt["Sharpe"] = fmt["Sharpe"].round(2)
    fmt["Excess t"] = fmt["Excess t"].round(2)
    print(fmt.to_string(index=False))
    out.to_csv("results/edge_lab_scoreboard.csv", index=False)


if __name__ == "__main__":
    main()
