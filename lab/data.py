"""Daily bars: adjusted OHLC for returns, plus the price as actually traded for digit signals."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lab.config import START

CACHE = Path(__file__).resolve().parent.parent / "data_cache"


def unsplit(split_adjusted: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo split adjustment. `splits` holds the ratio (4.0 for 4:1) indexed by ex-date.

    Before an N:1 split the stock really traded N times higher than the adjusted series shows.
    """
    factor = pd.Series(1.0, index=split_adjusted.index)
    for ex_date, ratio in splits.items():
        if ratio and ratio > 0:
            factor[split_adjusted.index < ex_date] *= ratio
    return split_adjusted * factor


def load_daily(symbol: str, start: str = START, use_cache: bool = True) -> pd.DataFrame:
    """Columns: open, high, low, close (dividend+split adjusted), volume, raw_close (as traded)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{symbol}_{start}.parquet"
    if use_cache and f.exists():
        return pd.read_parquet(f)

    import yfinance as yf

    t = yf.Ticker(symbol)
    adj = t.history(start=start, auto_adjust=True, actions=True)
    raw = t.history(start=start, auto_adjust=False, actions=True)
    if adj.empty or raw.empty:
        raise RuntimeError(f"no data for {symbol}")
    for frame in (adj, raw):
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()

    df = adj[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    splits = raw["Stock Splits"]
    splits = splits[splits > 0]
    # yfinance's unadjusted Close is still split-adjusted, so undo the splits.
    df["raw_close"] = unsplit(raw["Close"], splits).reindex(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["open", "close", "raw_close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df.to_parquet(f)
    return df


def load_universe(symbols: list[str], start: str = START) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        try:
            out[s] = load_daily(s, start)
        except Exception as e:  # one bad ticker shouldn't sink the battery
            print(f"  {s}: load failed ({type(e).__name__}: {e})")
    return out
