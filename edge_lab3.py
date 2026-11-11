"""Edge Lab round 3: stacking signals and a walk-forward prediction model.

Rules fixed before running. Same split: discovery through 2023-10-31, holdout from 2023-11-01.

C1  Earnings beat + near 52-week high: among top-quintile surprises (Edge Lab 2 events), compare
    the 60-day drift when the stock closed within 5% of its 252-day high on the reaction day vs not.
C2  T4 + PEAD blend: daily T4 portfolio (3+ down closes -> buy next open, sell 3 opens later) and
    the PEAD portfolio, 50/50. Count losing years and compare to SPY and equal-weight hold.
W1  Walk-forward model ("predict before it happens"): each month-end, for the 56 stocks, compute
    features known at that date, then predict next-month return minus the stock average.
    Model: ridge regression on cross-sectionally ranked features, retrained every January on ALL
    prior months only (expanding window). Trade the top 10 predicted stocks, equal weight, monthly.
    Features: 12-1m momentum, 1m return (reversal), distance to 52w high, 60d volatility,
    distance to 200d average, current down streak, latest EPS surprise if reported in the last 90 days.
    Scored vs equal-weight hold of the same 56 (bias control) and vs a plain momentum top-10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import edge_lab as L
import edge_lab2 as L2
from lab.config import ETFS, STOCKS, DISCOVERY_END, HOLDOUT_START


def down_streak(close: pd.DataFrame) -> pd.DataFrame:
    """Consecutive down closes ending on each day, per column."""
    down = (close.diff() < 0).values
    out = np.zeros(down.shape, dtype=int)
    for i in range(1, len(down)):
        out[i] = np.where(down[i], out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def t4_portfolio(px: dict, cash: pd.Series, streak_n: int = 3, hold: int = 3) -> pd.Series:
    syms = ETFS + STOCKS
    close = pd.DataFrame({s: px[s]["close"] for s in syms})
    open_ = pd.DataFrame({s: px[s]["open"] for s in syms})
    sig = (down_streak(close) >= streak_n).values                       # known at close of day t
    o = open_.values
    n_days, n_sym = o.shape
    # position earns open-to-open returns from open t+1 to open t+1+hold
    oo = np.full_like(o, np.nan)
    oo[1:] = o[1:] / o[:-1] - 1                             # return from open d-1 to open d, booked on day d
    held = np.zeros_like(sig, dtype=bool)
    for j in range(n_sym):
        for i in np.where(sig[:, j])[0]:
            held[i + 2:i + 2 + hold, j] = True               # open t+1 -> open t+1+hold
    n = held.sum(axis=1)
    entries = np.zeros_like(held); entries[2:] = sig[:-2]
    port = np.where(n > 0, np.nansum(np.where(held, oo, 0), axis=1) / np.maximum(n, 1),
                    cash.reindex(close.index).fillna(0).values)
    cost = np.where(n > 0, entries.sum(axis=1) / np.maximum(n, 1), 0) * L.STK_COST
    return pd.Series(port - cost, index=close.index)


def c1_beat_near_high(px: dict) -> pd.DataFrame:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    hi = close.rolling(252, min_periods=200).max()
    ev = L2.r4_pead_event_study(px)
    ev = ev[ev.q == 5].copy()
    ev["near_high"] = [close.at[d, s] >= 0.95 * hi.at[d, s] if pd.notna(hi.at[d, s]) else np.nan
                       for s, d in zip(ev.symbol, ev.react_day)]
    return ev.dropna(subset=["near_high"])


def features(px: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS})
    ret = close.pct_change()
    me = L.month_ends(close.index)
    f = {}
    m = close.loc[me]
    f["mom12_1"] = m.shift(1) / m.shift(12) - 1
    f["rev1"] = m / m.shift(1) - 1
    f["hi52"] = (close / close.rolling(252, min_periods=200).max()).loc[me]
    f["vol60"] = (ret.rolling(60).std()).loc[me]
    f["ma200"] = (close / close.rolling(200).mean()).loc[me]
    f["streak"] = down_streak(close).loc[me]
    # latest EPS surprise reported in the prior 90 days (known at month-end)
    ev = L2.earnings_events(close)
    sup = pd.DataFrame(np.nan, index=me, columns=STOCKS)
    for s, g in ev.groupby("symbol"):
        g = g.sort_values("react_day")
        for d in me:
            past = g[(g.react_day <= d) & (g.react_day > d - pd.Timedelta(days=90))]
            if len(past):
                sup.at[d, s] = past.surprise_pct.iloc[-1]
    f["eps_surprise"] = sup
    fwd = (m.shift(-1) / m - 1)                               # next month's return (target)
    target = fwd.sub(fwd.mean(axis=1), axis=0)              # relative to the cross-section
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    panel["target"] = target.stack(future_stack=True)
    panel.index.names = ["date", "symbol"]
    return panel, fwd


def rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="date").rank(pct=True) - 0.5


def walk_forward(px: dict, alpha: float = 10.0, top: int = 10) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    panel, fwd = features(px)
    X = rank_norm(panel.drop(columns="target")).fillna(0.0)   # missing feature = neutral
    y = panel["target"]
    dates = sorted(set(panel.index.get_level_values("date")))
    picks, picks_mom, coefs = {}, {}, {}
    beta = None
    for d in dates:
        if d < pd.Timestamp("2004-12-31"):
            continue
        if beta is None or d.month == 1 and d.year not in coefs:
            train = X.index.get_level_values("date") < d
            tr = train & y.notna().values
            Xt, yt = X[tr].values, y[tr].values
            beta = np.linalg.solve(Xt.T @ Xt + alpha * len(Xt) / 100 * np.eye(Xt.shape[1]), Xt.T @ yt)
            coefs[d.year] = beta
        xd = X.xs(d, level="date")
        valid = panel.xs(d, level="date")["mom12_1"].notna()
        score = pd.Series(xd.values @ beta, index=xd.index)[valid]
        picks[d] = score.nlargest(top).index.tolist()
        picks_mom[d] = panel.xs(d, level="date")["mom12_1"].dropna().nlargest(top).index.tolist()
    close = pd.DataFrame({s: px[s]["close"] for s in STOCKS}).ffill()
    model = L.monthly_rotation(close, picks, L.STK_COST)
    mom = L.monthly_rotation(close, picks_mom, L.STK_COST)
    coef_df = pd.DataFrame(coefs, index=X.columns).T
    return model, mom, coef_df


def main():
    px = {s: L.load(s) for s in ETFS + STOCKS}
    spy = px["SPY"]; bench = spy["close"].pct_change()
    cash = L2.tbill_daily(spy.index)
    ew = pd.DataFrame({s: px[s]["close"] for s in STOCKS}).pct_change().mean(axis=1)
    periods = {"Discovery": ("2005-01-01", DISCOVERY_END), "Holdout": (HOLDOUT_START, None)}

    print("C1 Earnings beats: near 52-week high vs not (60-day drift minus SPY, net)")
    ev = c1_beat_near_high(px)
    for pname, (a, b) in periods.items():
        d = ev[(ev.react_day >= a) & (ev.react_day <= pd.Timestamp(b or "2100"))]
        hi, lo = d[d.near_high == True].drift_ex, d[d.near_high == False].drift_ex
        diff = hi.mean() - lo.mean(); se = np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
        print(f"  {pname}: near high {hi.mean()*100:+.2f}% (n={len(hi)}) | not near {lo.mean()*100:+.2f}% (n={len(lo)}) | diff {diff*100:+.2f}% t {diff/se:.2f}")

    t4 = t4_portfolio(px, cash)
    pead = L2.r5_pead_portfolio(px, cash)
    blend = 0.5 * t4 + 0.5 * pead
    model, mom, coefs = walk_forward(px)
    series = {"SPY": bench, "Equal-weight 56": ew, "T4 portfolio": t4, "PEAD portfolio": pead,
              "T4+PEAD 50/50": blend, "Plain momentum top-10": mom, "Walk-forward model top-10": model}
    rows = []
    for pname, (a, b) in periods.items():
        for name, r in series.items():
            rr = r.loc[a:b].fillna(0.0)
            s = L.stats(rr)
            y = yearly(rr)
            rows.append({"Period": pname, "Strategy": name, "CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"],
                         "Losing yrs": f"{(y < 0).sum()}/{len(y)}",
                         "t vs EW56": L.excess_t(rr, ew.loc[a:b].reindex(rr.index).fillna(0)) if name not in ("SPY", "Equal-weight 56") else np.nan})
    out = pd.DataFrame(rows)
    f = out.copy()
    for c in ["CAGR", "MaxDD"]:
        f[c] = (f[c] * 100).round(1).astype(str) + "%"
    f["Sharpe"] = f["Sharpe"].round(2); f["t vs EW56"] = f["t vs EW56"].round(2)
    pd.set_option("display.width", 200)
    print("\n" + f.to_string(index=False))
    out.to_csv("results/edge_lab3_scoreboard.csv", index=False)

    full = {k: yearly(v.loc["2005":].fillna(0)) for k, v in series.items()}
    yt = pd.DataFrame(full)
    beat_ew = (yt.drop(columns=["SPY", "Equal-weight 56"]).sub(yt["Equal-weight 56"], axis=0) > 0).sum()
    print("\nYears beating equal-weight hold (2005-2026, of", len(yt), "):")
    print(beat_ew.to_string())
    print("\nModel weights by retrain year (positive = higher feature rank predicts higher return):")
    print(coefs.round(4).iloc[[0, len(coefs)//2, -1]].to_string())
    yt.to_csv("results/edge_lab3_yearly.csv")


if __name__ == "__main__":
    main()
