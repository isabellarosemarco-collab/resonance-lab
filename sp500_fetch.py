"""Download prices (2023-06-01 on) and earnings dates for the 2023-11-01 S&P 500 list."""
import os, sys, time
import pandas as pd, yfinance as yf
OUT = "data_cache/sp500"; os.makedirs(OUT, exist_ok=True)
syms = open("data_cache/sp500_members_2023-11-01.txt").read().split() + ["SPY"]
fails, earn = [], []
for k, s in enumerate(syms):
    f = f"{OUT}/{s}.parquet"
    try:
        if not os.path.exists(f):
            t = yf.Ticker(s)
            h = t.history(start="2023-06-01", auto_adjust=True)
            if h.empty: raise RuntimeError("no prices")
            h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
            h = h[["Open", "Close"]].rename(columns=str.lower); h.to_parquet(f)
            if s != "SPY":
                e = t.get_earnings_dates(limit=16)
                if e is not None and len(e):
                    e = e.reset_index().rename(columns={"Earnings Date": "date", "EPS Estimate": "eps_est",
                                                        "Reported EPS": "eps_act", "Surprise(%)": "surprise_pct"})
                    e["symbol"] = s; e[["date","eps_est","eps_act","surprise_pct","symbol"]].to_parquet(f"{OUT}/{s}.earn.parquet")
            time.sleep(0.3)
    except Exception as ex:
        fails.append((s, str(ex)[:80]))
    if k % 50 == 0: print(k, len(fails), flush=True)
pd.DataFrame(fails, columns=["symbol", "error"]).to_csv(f"{OUT}/_fails.csv", index=False)
print("done", len(syms), "fails", len(fails))
